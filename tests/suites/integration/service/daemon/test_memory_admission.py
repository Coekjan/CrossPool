from __future__ import annotations

from dataclasses import fields
from http import HTTPStatus
from pathlib import Path

import pytest

from xpool.fabric import FabricDeploymentPlan
from xpool.memory import (
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    XpoolMemoryCalibrationProfile,
)
from xpool.runtime.ffnagent.device_memory import DeviceMemoryEstimator
from xpool.service.wire import FfnAgentRegistration
from xtest.harness.support.config import reset_global_config, synthetic_config, synthetic_memory_context
from xtest.harness.support.service.daemon import (
    atnagent_registration,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    register,
    request,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def test_unmodeled_overhead_allows_unprofiled_http_plan_publication() -> None:
    """Use real analytic admission rather than replacing the memory ledger."""

    app = create_app(synthetic_config())
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
    response = request(app, "GET", "/fabric/plan")
    assert response.status_code == HTTPStatus.OK
    plan = FabricDeploymentPlan.model_validate_json(response.content)
    assert plan.ffn_execution_contracts == (FfnAgentRegistration.model_validate(ffnagent).execution_contract,)
    assert len(plan.ffn_memory_estimates) == 1
    assert plan.ffn_memory_estimates[0].peak_bytes > 0
    assert request(app, "GET", "/ready").json()["ready"] is False


def test_unprofiled_registration_omits_compatibility_observations() -> None:
    app = create_app(synthetic_config())
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    atnagent["calibration_environment"] = None
    ffnagent["calibration_environment"] = None
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
    response = request(app, "GET", "/fabric/plan")
    assert response.status_code == HTTPStatus.OK
    plan = response.json()
    assert plan["ffn_execution_contracts"] == [ffnagent["execution_contract"]]
    assert len(plan["ffn_memory_estimates"]) == 1


@pytest.mark.parametrize("missing_role", ("atn", "ffn"))
def test_partial_compatibility_observation_cannot_publish_a_plan(missing_role: str) -> None:
    app = create_app(synthetic_config())
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    (atnagent if missing_role == "atn" else ffnagent)["calibration_environment"] = None
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    response = register(app, "/instance/register", instance_registration())
    assert response.status_code == HTTPStatus.CONFLICT
    assert "partial memory compatibility" in response.text
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE


@pytest.mark.parametrize("base_bytes", (0, 32 << 20))
def test_published_correction_survives_profile_removal(tmp_path: Path, base_bytes: int) -> None:
    config = synthetic_config()
    path = tmp_path / "memory-profile.json"
    config = config.model_copy(update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": path})})
    coefficients = FfnMemoryCalibrationCoefficients(
        **{
            field.name: base_bytes if field.name == "base_bytes" else 0
            for field in fields(FfnMemoryCalibrationCoefficients)
        }
    )
    profile = XpoolMemoryCalibrationProfile(
        memory_context=synthetic_memory_context(),
        ffn=FfnMemoryCalibration(
            executor_lane_count=config.scheduler.ffn_concurrency,
            minimum_held_out_headroom_bytes=0,
            coefficients=coefficients,
        ),
    )
    path.write_text(profile.model_dump_json())
    app = create_app(config)
    assert register(app, "/atnagent/register", atnagent_registration(device=0)).status_code == HTTPStatus.NO_CONTENT
    ffnagent = ffnagent_registration()
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
    response = request(app, "GET", "/fabric/plan")
    assert response.status_code == HTTPStatus.OK
    plan = FabricDeploymentPlan.model_validate_json(response.content)
    path.unlink()
    assert request(app, "GET", "/fabric/plan").json() == response.json()
    estimator = DeviceMemoryEstimator(
        model_specs=FfnAgentRegistration.model_validate(ffnagent).model_specs,
        instance_profiles=tuple(instance.ffn_profile for instance in plan.instance_plans),
        ffn_execution_contracts=plan.ffn_execution_contracts,
        coefficients=coefficients,
    )
    assert plan.ffn_memory_estimates == (
        estimator.estimate_model_plans(model_plans=plan.model_plans, ffnagent_index=0),
    )


def test_invalid_configured_profile_cannot_fall_back_to_analytic(tmp_path: Path) -> None:
    config = synthetic_config()
    path = tmp_path / "invalid-profile.json"
    path.write_text("invalid json")
    config = config.model_copy(update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": path})})
    app = create_app(config)
    assert register(app, "/atnagent/register", atnagent_registration(device=0)).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent_registration()).status_code == HTTPStatus.NO_CONTENT
    response = register(app, "/instance/register", instance_registration())
    assert response.status_code == HTTPStatus.CONFLICT
    assert "failed to load memory calibration Profile" in response.text
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
