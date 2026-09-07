from __future__ import annotations

from types import SimpleNamespace

import pytest

import xpool.service.daemon.ffn_placement
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.native.sizing import install_native_allocation_sizing
from tests.harness.support.service.daemon import ffn_model_spec, ffn_profile
from xpool.config import XpoolConfig
from xpool.fabric import FabricInstancePlan, InstanceFfnProfile, InstanceRankTopology
from xpool.ffn import FfnModelSpec
from xpool.memory import FfnMemoryCalibrationCoefficients
from xpool.runtime.ffnagent import device_memory
from xpool.service.daemon.ffn_placement import place_ffn_models
from xpool.service.errors import XpoolDaemonError

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def native_allocation_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_allocation_sizing(monkeypatch)


def config(*, tp_size: int | None = None) -> XpoolConfig:
    """Build one two-FfnAgent configuration."""

    model: dict[str, object] = {"id": "m", "path": "/models/m"}
    if tp_size is not None:
        model["ffn_tp_size"] = tp_size
    return XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [model],
        }
    )


def instance_plan(instance_id: str = "m") -> FabricInstancePlan:
    """Build the co-indexed Instance declaration."""

    return FabricInstancePlan(
        instance_id=instance_id,
        ffn_profile=InstanceFfnProfile.model_validate(ffn_profile()),
        instance_rank_topology=InstanceRankTopology(atn_tp_size=1, atn_dp_size=1, atnagent_indices=(0,)),
    )


def test_structural_placement_uses_one_common_lowest_index_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id="m"))
    install_test_config(config())
    coefficients = FfnMemoryCalibrationCoefficients(
        base_bytes=10,
        bytes_per_tensor_storage_mib=0,
        bytes_per_tensor_storage_allocation=0,
        bytes_per_dense_graph_capture=0,
        bytes_per_moe_graph_capture=0,
        bytes_per_executor_lane=0,
        bytes_per_compute_branch=0,
        dense_implementation_bytes=0,
        moe_implementation_bytes=0,
        joined_ffnagent_bytes=0,
    )
    monkeypatch.setattr(
        device_memory,
        "load_memory_calibration_profile",
        lambda: SimpleNamespace(ffn=SimpleNamespace(coefficients=coefficients)),
    )

    plans = place_ffn_models(
        model_specs=(spec,),
        instance_plans=(instance_plan(),),
        ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
    )

    assert plans[0].tp_size == 2
    assert plans[0].layers[0].ffnagent_indices == (0, 1)
    assert plans[0].layers[0].local_intermediate_size == 4


def test_equal_optimum_placement_uses_lowest_index_groups() -> None:
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "atn": {"devices": [0]},
                "ffn": {"devices": [1, 2, 3, 4]},
                "models": [
                    {"id": "wide", "path": "/models/wide", "ffn_tp_size": 4},
                    {"id": "narrow", "path": "/models/narrow", "ffn_tp_size": 2},
                ],
            }
        )
    )
    plans = place_ffn_models(
        model_specs=tuple(
            FfnModelSpec.model_validate(ffn_model_spec(model_id=model_id)) for model_id in ("wide", "narrow")
        ),
        instance_plans=(instance_plan("wide"), instance_plan("narrow")),
        ffnagent_free_memory_bytes=(1 << 40,) * 4,
    )

    assert tuple(frozenset(plan.layers[0].ffnagent_indices) for plan in plans) == (
        frozenset((0, 1, 2, 3)),
        frozenset((0, 1)),
    )


def test_structural_placement_rejects_tp_wider_than_fleet() -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id="m"))
    install_test_config(config(tp_size=3))

    with pytest.raises(XpoolDaemonError, match="exceeds the FfnAgent Fleet"):
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
        )


def test_placement_rejects_insufficient_memory_without_fallback() -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id="m"))
    install_test_config(config())

    with pytest.raises(XpoolDaemonError, match="INFEASIBLE"):
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1, 1),
        )


def test_placement_accounts_for_fabric_observer_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id="m"))
    install_test_config(config())
    install_native_allocation_sizing(monkeypatch, fabric_observer_bytes=1 << 30)

    with pytest.raises(XpoolDaemonError, match="INFEASIBLE"):
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
        )


def test_placement_uses_one_deadline_before_solver_work(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = FfnModelSpec.model_validate(ffn_model_spec(model_id="m"))
    install_test_config(config())
    times = iter((0.0, 61.0))
    monkeypatch.setattr(xpool.service.daemon.ffn_placement, "monotonic", lambda: next(times))

    with pytest.raises(XpoolDaemonError, match="deadline expired during resource calculation"):
        place_ffn_models(
            model_specs=(spec,),
            instance_plans=(instance_plan(),),
            ffnagent_free_memory_bytes=(1 << 30, 1 << 30),
        )
