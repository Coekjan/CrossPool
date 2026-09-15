from __future__ import annotations

import errno
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest

import tests.harness.sglang.serving.attempt
import tests.harness.sglang.serving.probe
from tests.harness.native.cluster import XpoolCluster
from tests.harness.runner.network import TcpEndpointConflict, TcpEndpointReservation
from tests.harness.sglang.manifest import (
    E2eFfnInputMatrix,
    E2eFfnNumericalCase,
    E2eFfnTopologyCase,
    E2eFfnTopologyInstance,
    E2eFfnTopologyRequest,
    E2eManifest,
    E2eModel,
    E2eModelPlacement,
    E2eServingCase,
)
from tests.harness.sglang.serving.attempt import ProbeAttempt, ProbeRun
from tests.harness.sglang.serving.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from tests.harness.sglang.serving.graph import SglangGraphMode, SglangGraphSettings
from tests.harness.sglang.serving.launch import E2eLaunch, E2eLaunchModel
from tests.harness.sglang.serving.probe import run_probe
from tests.harness.sglang.serving.server import SglangServerProcess, SglangServerResult
from xpool.config import XpoolConfig


def test_probe_retries_only_complete_endpoint_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdirs: list[Path] = []
    expected = cast(ProbeRun, object())

    class Attempt:
        def __init__(self, *args: object) -> None:
            workdirs.append(cast(Path, args[4]))

        def run(self, workload: object = None) -> ProbeRun:
            assert workload is None
            if len(workdirs) < 3:
                raise TcpEndpointConflict((("127.0.0.1", 20_000 + len(workdirs)),)) from RuntimeError("startup")
            return expected

    monkeypatch.setattr(tests.harness.sglang.serving.probe, "ProbeAttempt", Attempt)

    result = run_probe(
        probe_manifest(),
        probe_case(),
        base_config=probe_launch(tmp_path).config,
        graph_settings=SglangGraphSettings("disabled", "disabled"),
        workdir=tmp_path / "run",
    )

    assert result is expected
    assert workdirs == [tmp_path / "run" / f"attempt-{number}" for number in range(1, 4)]


def test_probe_does_not_retry_non_conflict_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class Attempt:
        def __init__(self, *args: object) -> None:
            pass

        def run(self, workload: object = None) -> ProbeRun:
            assert workload is None
            nonlocal attempts
            attempts += 1
            raise AssertionError("startup failure")

    monkeypatch.setattr(tests.harness.sglang.serving.probe, "ProbeAttempt", Attempt)

    with pytest.raises(AssertionError, match="startup failure"):
        run_probe(
            probe_manifest(),
            probe_case(),
            base_config=probe_launch(tmp_path).config,
            graph_settings=SglangGraphSettings("disabled", "disabled"),
            workdir=tmp_path / "run",
        )

    assert attempts == 1


def test_attempt_converts_startup_endpoint_occupation_to_retryable_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = probe_attempt(tmp_path)
    address = ("127.0.0.1", 20_000)
    events: list[str] = []

    def fail_start(self: ProbeAttempt) -> E2eLaunch:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(ProbeAttempt, "start_system", fail_start)
    monkeypatch.setattr(ProbeAttempt, "diagnostics", lambda self: "")
    monkeypatch.setattr(ProbeAttempt, "close_processes_and_inspect_endpoints", lambda self: (address,))
    monkeypatch.setattr(ProbeAttempt, "close_endpoints", lambda self: events.append("endpoint-release"))

    with pytest.raises(TcpEndpointConflict) as error:
        attempt.run()

    assert error.value.addresses == (address,)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert events == ["endpoint-release"]
    assert attempt.closed


def test_attempt_classifies_released_family_and_reports_only_occupied_ports(tmp_path: Path) -> None:
    attempt = probe_attempt(tmp_path)
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    endpoint = cast(
        SglangEndpointFamilyLease,
        SimpleNamespace(family=family, tcp_released=True, reacquire_tcp=lambda: (family.nccl_port,)),
    )
    attempt.server_endpoints.append(endpoint)

    assert attempt.classify_released_endpoints() == ((family.host, family.nccl_port),)


def test_attempt_propagates_non_conflict_classification_failure(tmp_path: Path) -> None:
    attempt = probe_attempt(tmp_path)
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)

    def fail() -> tuple[int, ...]:
        raise OSError(errno.EACCES, "inspection denied")

    attempt.server_endpoints.append(
        cast(SglangEndpointFamilyLease, SimpleNamespace(family=family, tcp_released=True, reacquire_tcp=fail))
    )

    with pytest.raises(OSError) as error:
        attempt.classify_released_endpoints()

    assert error.value.errno == errno.EACCES


def test_attempt_closes_processes_before_endpoint_reservations(tmp_path: Path) -> None:
    attempt = probe_attempt(tmp_path)
    events: list[str] = []
    attempt.servers.append(cast(SglangServerProcess, SimpleNamespace(close=lambda: events.append("server"))))
    attempt.cluster = cast(XpoolCluster, SimpleNamespace(close=lambda: events.append("cluster")))
    attempt.daemon_endpoint = cast(
        TcpEndpointReservation,
        SimpleNamespace(close=lambda: events.append("daemon-endpoint")),
    )
    attempt.server_endpoints.append(
        cast(SglangEndpointFamilyLease, SimpleNamespace(close=lambda: events.append("server-endpoint")))
    )

    attempt.close()

    assert events == ["server", "cluster", "daemon-endpoint", "server-endpoint"]


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("startup", RuntimeError),
        ("inference", AssertionError),
        ("success", None),
    ],
)
def test_attempt_terminal_paths_cleanup_classify_and_release_endpoints(
    outcome: str,
    expected_error: type[BaseException] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = probe_attempt(tmp_path)
    launch = probe_launch(tmp_path)
    events: list[str] = []
    endpoint = SimpleNamespace()
    cluster = SimpleNamespace(daemon_startup_seconds=0.5)
    server = SimpleNamespace()

    monkeypatch.setattr(
        TcpEndpointReservation,
        "reserve",
        classmethod(lambda cls, host, *, port_space: SimpleNamespace(port=launch.config.daemon.port)),
    )
    monkeypatch.setattr(
        SglangEndpointFamilyLease,
        "acquire",
        classmethod(lambda cls, host, *, dp_size, port_space: endpoint),
    )
    monkeypatch.setattr(tests.harness.sglang.serving.attempt, "materialize", lambda *args, **kwargs: launch)
    monkeypatch.setattr(XpoolCluster, "start", classmethod(lambda cls, launch, daemon_endpoint: cluster))
    monkeypatch.setattr(SglangServerProcess, "start", classmethod(lambda cls, **kwargs: server))

    def wait_for_readiness(launch: E2eLaunch, servers: list[SglangServerProcess]) -> None:
        if outcome == "startup":
            raise RuntimeError("startup failed")

    def server_result(server: SglangServerProcess, request_barrier: object) -> SglangServerResult:
        del server
        assert request_barrier is not None
        if outcome == "inference":
            raise RuntimeError("inference failed")
        return cast(SglangServerResult, object())

    monkeypatch.setattr(tests.harness.sglang.serving.attempt, "wait_for_system_readiness", wait_for_readiness)
    monkeypatch.setattr(SglangServerProcess, "result", server_result)
    monkeypatch.setattr(tests.harness.sglang.serving.attempt, "read_graph_events", lambda path: [])
    monkeypatch.setattr(ProbeAttempt, "diagnostics", lambda self: "")
    monkeypatch.setattr(
        ProbeAttempt,
        "close_processes",
        lambda self: events.append("process-cleanup") or (),
    )
    monkeypatch.setattr(
        ProbeAttempt,
        "classify_released_endpoints",
        lambda self: events.append("endpoint-classification") or (),
    )
    monkeypatch.setattr(
        ProbeAttempt,
        "close_endpoints",
        lambda self: events.append("endpoint-release"),
    )

    if expected_error is None:
        assert isinstance(attempt.run(), ProbeRun)
    else:
        with pytest.raises(expected_error):
            attempt.run()

    assert events == ["process-cleanup", "endpoint-classification", "endpoint-release"]
    assert attempt.closed


def probe_attempt(tmp_path: Path) -> ProbeAttempt:
    launch = probe_launch(tmp_path)
    return ProbeAttempt(
        probe_manifest(),
        probe_case(),
        launch.config,
        SglangGraphSettings("disabled", "disabled"),
        tmp_path / "attempt",
    )


def probe_launch(tmp_path: Path) -> E2eLaunch:
    config_path = tmp_path / "xpool.toml"
    config_path.write_text("", encoding="utf-8")
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19810},
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "model-a"}],
        },
        env={},
    )
    return E2eLaunch(
        case_id="probe",
        models=(E2eLaunchModel("model", "model-a", "SyntheticForCausalLM", 1, 1),),
        config=config,
        config_path=config_path,
        environment=MappingProxyType({"XPOOL_CONFIG": str(config_path)}),
        observer_outdir=tmp_path / "observers",
    )


def probe_manifest() -> E2eManifest:
    model = E2eModel(
        alias="model",
        model_id="model-a",
        architecture="SyntheticForCausalLM",
    )
    case = probe_case()
    numerical = E2eFfnNumericalCase(
        id="probe-numerical",
        model_placement=case.models[0],
        layer_ids=(0,),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1,)),
        ffnagent_count=1,
        executor_lane_count=1,
        ffn_tp_size=1,
        estimated_duration_seconds=1,
        timeout_seconds=1,
    )
    topology = E2eFfnTopologyCase(
        id="probe-topology",
        atnagent_count=1,
        ffnagent_count=1,
        executor_lane_count=1,
        instances=(
            E2eFfnTopologyInstance(model="model", layer_ordinal=0, atn_tp_size=1, atn_dp_size=1, ffn_tp_size=1),
        ),
        requests=(
            E2eFfnTopologyRequest(
                instance_index=0,
                forward_mode="decode",
                dp_rank_payload_rows=(1,),
                output_requirement="per_rank_complete",
            ),
        ),
        estimated_duration_seconds=1,
        timeout_seconds=1,
    )
    return E2eManifest(
        models=(model,),
        model_serving_cases=(case,),
        ffn_numerical_cases=(numerical,),
        ffn_topology_cases=(topology,),
    )


def probe_case() -> E2eServingCase:
    return E2eServingCase(
        id="probe",
        models=(E2eModelPlacement(model="model", atn_tp_size=1, atn_dp_size=1),),
        ffnagent_count=1,
        executor_lane_count=1,
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_record_capacity=1,
        fabric_record_capacity=1,
    )
