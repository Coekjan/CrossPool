from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import xtest.harness.sglang.serving.attempt
import xtest.harness.sglang.serving.probe
from xkit.network import TcpEndpointConflict
from xkit.process import OwnedProcessGroup
from xkit.serving.cluster import XpoolClusterLaunch
from xkit.serving.sglang.endpoints import SglangEndpointFamily
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.config import XpoolConfig
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.attempt import ProbeAttempt, ProbeRun
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import SglangProbeServer, SglangServerResult
from xtest.harness.support.config import TEST_MODEL_ID


def test_probe_retries_only_complete_endpoint_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdirs: list[Path] = []
    expected = cast(ProbeRun, object())

    class Attempt:
        def __init__(self, *, workdir: Path, **kwargs: object) -> None:
            workdirs.append(workdir)

        def run(self, workload: object = None) -> ProbeRun:
            assert workload is None
            if len(workdirs) < 3:
                raise TcpEndpointConflict((("127.0.0.1", 20_000 + len(workdirs)),)) from RuntimeError("startup")
            return expected

    monkeypatch.setattr(xtest.harness.sglang.serving.probe, "ProbeAttempt", Attempt)

    result = run_probe(
        probe_case(),
        base_config=probe_config(tmp_path),
        graph_settings=SglangGraphSettings("disabled", "disabled"),
        workdir=tmp_path / "run",
    )

    assert result is expected
    assert workdirs == [tmp_path / "run" / f"attempt-{number}" for number in range(1, 4)]


def test_probe_does_not_retry_non_conflict_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class Attempt:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, workload: object = None) -> ProbeRun:
            assert workload is None
            nonlocal attempts
            attempts += 1
            raise AssertionError("startup failure")

    monkeypatch.setattr(xtest.harness.sglang.serving.probe, "ProbeAttempt", Attempt)

    with pytest.raises(AssertionError, match="startup failure"):
        run_probe(
            probe_case(),
            base_config=probe_config(tmp_path),
            graph_settings=SglangGraphSettings("disabled", "disabled"),
            workdir=tmp_path / "run",
        )

    assert attempts == 1


@pytest.mark.parametrize("inference_failure", [False, True])
def test_probe_adapter_uses_shared_system_and_always_closes(
    inference_failure: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = probe_attempt(tmp_path)
    launch = probe_launch(tmp_path)
    events: list[str] = []
    owner = cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test"))
    model = launch.models[0]
    server = SglangServerProcess(model, owner, SglangEndpointFamily("127.0.0.1", 20000, 21000, 22000, 1))
    system = SimpleNamespace(
        launch=XpoolClusterLaunch(
            config=launch.config,
            config_path=tmp_path / "runtime.toml",
            environment=launch.environment,
            cwd=launch.cwd,
        ),
        servers=[server],
        cluster=SimpleNamespace(daemon_startup_seconds=0.5),
        close=lambda: events.append("shared-close"),
        diagnostics=lambda: "test diagnostics",
    )

    monkeypatch.setattr(xtest.harness.sglang.serving.attempt, "prepare", lambda *args, **kwargs: launch)

    def start(cls: type[XpoolServingSystem], serving_launch: ServingLaunch, **kwargs: object) -> XpoolServingSystem:
        assert serving_launch is launch
        events.append("shared-start")
        return cast(XpoolServingSystem, system)

    monkeypatch.setattr(XpoolServingSystem, "start", classmethod(start))
    monkeypatch.setattr(xtest.harness.sglang.serving.attempt, "read_graph_events", lambda path: [])

    def workload(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        assert servers[0].process is server
        assert servers[0].inference_path == attempt.workdir / "models" / TEST_MODEL_ID.uri_encode() / "inference.json"
        events.append("test-request")
        if inference_failure:
            raise RuntimeError("inference failed")
        return (SglangServerResult(model.model_id, model.graph_mode.settings(), tuple(range(8)), None),)

    if inference_failure:
        with pytest.raises(AssertionError, match="inference failed"):
            attempt.run(workload)
    else:
        result = attempt.run(workload)
        assert result.daemon_startup_seconds == 0.5
        assert result.model_ids == (TEST_MODEL_ID,)
        assert result.observer_outdir == (attempt.workdir / "observers").resolve()
    assert events == ["shared-start", "test-request", "shared-close"]


def probe_attempt(tmp_path: Path) -> ProbeAttempt:
    return ProbeAttempt(
        case=probe_case(),
        base_config=probe_config(tmp_path),
        graph_settings=SglangGraphSettings("disabled", "disabled"),
        workdir=tmp_path / "attempt",
    )


def probe_config(tmp_path: Path) -> ResolvedConfig:
    config_path = tmp_path / "xpool.toml"
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19810},
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID)}],
        },
        env={},
    )
    return ResolvedConfig(config_path, config)


def probe_launch(tmp_path: Path) -> ServingLaunch:
    return ServingLaunch(
        cwd=tmp_path,
        models=(SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),),
        config=probe_config(tmp_path).config,
        environment={},
    )


def probe_case() -> E2eServingCase:
    return E2eServingCase(
        id="probe",
        description="Probe lifecycle and endpoint-conflict retry policy.",
        deployment=Path("deployment.toml"),
        models=(TEST_MODEL_ID,),
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
    )
