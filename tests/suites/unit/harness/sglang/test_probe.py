from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest

import tests.harness.sglang.probe
from tests.harness.sglang.e2e import E2eLaunch, E2eLaunchModel
from tests.harness.sglang.graph import SglangGraphMode, SglangGraphSettings
from tests.harness.sglang.manifest import (
    E2eLoopbackServingCase,
    E2eManifest,
    E2eModel,
    E2eModelPlacement,
    E2eServingCase,
)
from tests.harness.sglang.probe import ProbeRun, run_probe, run_probe_attempt, wait_for_system_readiness
from tests.harness.sglang.server import SglangEndpointConflict, SglangServerProcess
from xpool.config import LoopbackSite, XpoolConfig


def test_probe_retries_only_complete_endpoint_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[Path] = []
    expected = cast(ProbeRun, object())

    def attempt(*args: object, workdir: Path, **kwargs: object) -> ProbeRun:
        attempts.append(workdir)
        if len(attempts) < 3:
            raise SglangEndpointConflict(20_000 + len(attempts), "metrics_port")
        return expected

    monkeypatch.setattr(tests.harness.sglang.probe, "run_probe_attempt", attempt)

    result = run_probe(
        probe_manifest(),
        probe_case(),
        base_config=probe_launch(tmp_path).config,
        graph_settings=SglangGraphSettings(False, False),
        workdir=tmp_path / "run",
    )

    assert result is expected
    assert attempts == [tmp_path / "run" / f"attempt-{attempt}" for attempt in range(1, 4)]


@pytest.mark.parametrize("loopback_site", [LoopbackSite.INSTANCE, LoopbackSite.ATNAGENT])
def test_non_fabric_loopback_readiness_stops_at_http_health(
    loopback_site: LoopbackSite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = probe_launch(tmp_path, loopback_site=loopback_site)
    server = cast(SglangServerProcess, SimpleNamespace(healthy=lambda: True))
    monkeypatch.setattr(
        tests.harness.sglang.probe.httpx,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("daemon final readiness is unreachable")),
    )

    wait_for_system_readiness(launch, [server])


def test_probe_does_not_retry_non_collision_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fail(*args: object, **kwargs: object) -> ProbeRun:
        nonlocal attempts
        attempts += 1
        raise AssertionError("startup failure")

    monkeypatch.setattr(tests.harness.sglang.probe, "run_probe_attempt", fail)

    with pytest.raises(AssertionError, match="startup failure"):
        run_probe(
            probe_manifest(),
            probe_case(),
            base_config=probe_launch(tmp_path).config,
            graph_settings=SglangGraphSettings(False, False),
            workdir=tmp_path / "run",
        )

    assert attempts == 1


def test_probe_attempt_does_not_expose_retryable_conflict_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = probe_launch(tmp_path)
    cluster = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(RuntimeError("cluster cleanup failed")),
        diagnostics=lambda: "cluster diagnostics",
    )
    server = SimpleNamespace(close=lambda: None, diagnostics=lambda: "server diagnostics")

    monkeypatch.setattr(tests.harness.sglang.probe, "materialize", lambda *args, **kwargs: launch)
    monkeypatch.setattr(
        tests.harness.sglang.probe.XpoolCluster,
        "start",
        classmethod(lambda cls, launch, endpoint: cluster),
    )
    monkeypatch.setattr(
        tests.harness.sglang.probe.SglangServerProcess,
        "start",
        classmethod(lambda cls, **kwargs: server),
    )

    def fail_readiness(launch: E2eLaunch, servers: list[SglangServerProcess]) -> None:
        raise SglangEndpointConflict(20_000, "metrics_port")

    monkeypatch.setattr(tests.harness.sglang.probe, "wait_for_system_readiness", fail_readiness)

    with pytest.raises(AssertionError, match="cleanup failures: cluster cleanup failed"):
        run_probe_attempt(
            probe_manifest(),
            probe_case(),
            base_config=launch.config,
            graph_settings=SglangGraphSettings(False, False),
            workdir=tmp_path / "attempt",
            loopback_site=LoopbackSite.FFNAGENT,
        )


def probe_launch(tmp_path: Path, *, loopback_site: LoopbackSite = LoopbackSite.FFNAGENT) -> E2eLaunch:
    config_path = tmp_path / "xpool.toml"
    config_path.write_text("", encoding="utf-8")
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19810},
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "model-a"}],
        },
        env={},
    )
    return E2eLaunch(
        case_id="probe",
        models=(
            E2eLaunchModel(
                alias="model",
                model_id="model-a",
                architecture="SyntheticForCausalLM",
                max_total_tokens=16_384,
                atn_tp_size=1,
                atn_dp_size=1,
            ),
        ),
        config=config,
        config_path=config_path,
        environment=MappingProxyType({"XPOOL_CONFIG": str(config_path)}),
        observer_outdir=tmp_path / "observers",
        loopback_site=loopback_site,
    )


def probe_manifest() -> E2eManifest:
    model = E2eModel(
        alias="model",
        model_id="model-a",
        architecture="SyntheticForCausalLM",
        max_total_tokens=16_384,
    )
    case = probe_case()
    return E2eManifest(models=(model,), model_serving_cases=(case,), loopback_serving_cases=(loopback_case(),))


def probe_case() -> E2eServingCase:
    return E2eServingCase(
        id="probe",
        models=(E2eModelPlacement(model="model", atn_tp_size=1, atn_dp_size=1),),
        ffnagent_count=1,
        executor_count=1,
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_trace_capacity=1,
        fabric_trace_capacity=1,
    )


def loopback_case() -> E2eLoopbackServingCase:
    values = probe_case().model_dump()
    values["id"] = "probe-sites"
    return E2eLoopbackServingCase.model_validate({**values, "sites": (LoopbackSite.FFNAGENT,)})
