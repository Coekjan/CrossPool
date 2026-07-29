from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import httpx
import pytest
from pydantic import TypeAdapter

import tests.harness.sglang.server
from tests.harness.runner.process import OwnedProcessGroup
from tests.harness.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from tests.harness.sglang.graph import SglangGraphSettings
from tests.harness.sglang.launch import E2eLaunch, E2eLaunchModel
from tests.harness.sglang.readiness import ReadinessEvidence
from tests.harness.sglang.server import (
    PROMPT,
    SglangInferenceRecord,
    SglangServerProcess,
    server_command,
)
from xpool.config import LoopbackSite, XpoolConfig


@pytest.mark.parametrize(
    ("model", "settings", "expected", "absent"),
    [
        (
            E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
            SglangGraphSettings(False, False),
            {"--disable-cuda-graph", "--disable-piecewise-cuda-graph"},
            {"--enable-dp-attention"},
        ),
        (
            E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
            SglangGraphSettings(False, True),
            {"--disable-cuda-graph"},
            {"--disable-piecewise-cuda-graph", "--enable-dp-attention"},
        ),
        (
            E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 2),
            SglangGraphSettings(False, True),
            {"--disable-cuda-graph", "--enable-dp-attention"},
            {"--disable-piecewise-cuda-graph"},
        ),
    ],
)
def test_server_command_projects_pinned_cli_policy(
    model: E2eLaunchModel,
    settings: SglangGraphSettings,
    expected: set[str],
    absent: set[str],
    tmp_path: Path,
) -> None:
    command = server_command(
        launch=server_launch(tmp_path),
        model=model,
        graph_settings=settings,
        endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, model.atn_dp_size),
    )

    assert command[:2] == ["sglang", "serve"]
    assert command[command.index("--max-total-tokens") + 1] == str(model.max_total_tokens)
    assert command[command.index("--nccl-port") + 1] == "19001"
    assert command[command.index("--tensor-parallel-size") + 1] == str(model.atn_tp_size * model.atn_dp_size)
    assert command[command.index("--data-parallel-size") + 1] == str(model.atn_dp_size)
    assert expected <= set(command)
    assert not absent & set(command)


def test_server_start_projects_process_specific_grpc_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = replace(
        server_launch(tmp_path),
        environment=MappingProxyType({"SGLANG_GRPC_PORT": "discarded", "XPOOL_TEST_VALUE": "preserved"}),
    )
    model = launch.models[0]
    family = SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1)
    events: list[str] = []
    endpoint = cast(
        SglangEndpointFamilyLease,
        SimpleNamespace(family=family, release_tcp_for_spawn=lambda: events.append("released")),
    )
    captured: dict[str, object] = {}

    def spawn_logged(
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> OwnedProcessGroup:
        events.append("spawned")
        captured.update(name=name, command=command, cwd=cwd, env=env, log_path=log_path)
        return cast(OwnedProcessGroup, SimpleNamespace())

    monkeypatch.setattr(OwnedProcessGroup, "spawn_logged", staticmethod(spawn_logged))

    server = SglangServerProcess.start(
        launch=launch,
        model=model,
        graph_settings=SglangGraphSettings(False, False),
        endpoint=endpoint,
        workdir=tmp_path,
    )

    assert events == ["released", "spawned"]
    assert captured["env"] == {"SGLANG_GRPC_PORT": "19002", "XPOOL_TEST_VALUE": "preserved"}
    assert launch.environment == {"SGLANG_GRPC_PORT": "discarded", "XPOOL_TEST_VALUE": "preserved"}
    assert server.endpoint is family
    assert server.inference_path == tmp_path / "organization-model.inference.json"


def test_server_startup_blocker_rejects_invalid_owned_tcp_store_peer() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            tail=lambda: (
                "[W TCPStore.cpp:384] TCP client failed to connect/validate to host 127.0.0.1:20237 - "
                "retrying: Ping failed, invalid value returned from server. Expected: 3058192, Got: 759714643"
            ),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 2),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )

    with pytest.raises(RuntimeError, match=r"retrying an invalid TCPStore peer at 127\.0\.0\.1:20237"):
        server.raise_for_startup_blocker()


def test_server_startup_blocker_ignores_peer_outside_owned_family() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            tail=lambda: (
                "[W TCPStore.cpp:384] TCP client failed to connect/validate to host 127.0.0.1:30000 - "
                "retrying: Ping failed, invalid value returned from server. Expected: 3058192, Got: 759714643"
            ),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 2),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )

    server.raise_for_startup_blocker()


def test_server_health_records_tcp_store_blocker_as_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: None),
            tail=lambda: (
                "[W TCPStore.cpp:384] TCP client failed to connect/validate to host 127.0.0.1:20237 - "
                "retrying: Ping failed, invalid value returned from server. Expected: 3058192, Got: 759714643"
            ),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 2),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )

    def connect_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("not ready")

    monkeypatch.setattr(tests.harness.sglang.server.httpx, "get", connect_error)
    evidence = ReadinessEvidence("SGLang health", "http://127.0.0.1:20000/health")

    with pytest.raises(RuntimeError, match="retrying an invalid TCPStore peer"):
        server.healthy(evidence)

    assert evidence.attempt_count == 2
    assert evidence.last_error_type == "RuntimeError"
    assert evidence.last_error_message is not None
    assert "retrying an invalid TCPStore peer" in evidence.last_error_message


def test_server_health_records_early_exit_before_raising() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: 1),
            tail=lambda: "ValueError: metrics_port at 20237 is not available in 30 seconds.",
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )
    evidence = ReadinessEvidence("SGLang health", "http://127.0.0.1:20000/health")

    with pytest.raises(RuntimeError, match="exited before readiness with code 1"):
        server.healthy(evidence)

    assert evidence.attempt_count == 1
    assert evidence.last_error_type == "RuntimeError"
    assert evidence.last_error_message is not None
    assert "exited before readiness with code 1" in evidence.last_error_message


def test_server_close_owns_only_process_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: 0),
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )
    monkeypatch.setattr(tests.harness.sglang.server, "wait_for_process_group", lambda process, timeout: True)

    server.close()
    server.close()

    assert events == ["process-close"]


def test_server_close_signals_only_live_leader_on_orderly_path(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        send_signal=lambda signum: events.append(f"signal:{signum}"),
    )
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=process,
            terminate=lambda: events.append("fallback"),
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )
    monkeypatch.setattr(tests.harness.sglang.server, "wait_for_process_group", lambda process, timeout: True)

    server.close()

    assert events == [f"signal:{tests.harness.sglang.server.signal.SIGTERM}", "process-close"]


def test_server_close_falls_back_to_process_group_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        send_signal=lambda signum: events.append(f"signal:{signum}"),
    )
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=process,
            terminate=lambda: events.append("fallback"),
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=owner,
        endpoint=family,
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )
    monkeypatch.setattr(tests.harness.sglang.server, "wait_for_process_group", lambda process, timeout: False)

    server.close()

    assert events == [f"signal:{tests.harness.sglang.server.signal.SIGTERM}", "fallback", "process-close"]


def test_server_result_reads_resolved_modes_and_exact_output_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "/server_info": FakeResponse({"disable_cuda_graph": False, "disable_piecewise_cuda_graph": True}),
        "/generate": FakeResponse({"output_ids": list(range(8))}),
    }
    monkeypatch.setattr(tests.harness.sglang.server.httpx, "Client", lambda **kwargs: FakeClient(responses))
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
        endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1),
        host="127.0.0.1",
        port=19000,
        inference_path=tmp_path / "inference.json",
    )

    result = server.result()

    assert result.model_id == "organization/model"
    assert result.resolved_graph_settings == SglangGraphSettings(True, False)
    assert result.output_ids == tuple(range(8))
    record = TypeAdapter(SglangInferenceRecord).validate_json(server.inference_path.read_bytes())
    assert record.model_id == "organization/model"
    assert record.url == "http://127.0.0.1:19000/generate"
    assert record.request["text"] == PROMPT
    assert record.response_status_code == 200
    assert record.response == {"output_ids": list(range(8))}


def test_server_result_rejects_boolean_token_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/server_info": FakeResponse({"disable_cuda_graph": True, "disable_piecewise_cuda_graph": True}),
        "/generate": FakeResponse({"output_ids": [0, 1, 2, 3, 4, 5, 6, True]}),
    }
    monkeypatch.setattr(tests.harness.sglang.server.httpx, "Client", lambda **kwargs: FakeClient(responses))
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 1),
        owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
        endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1),
        host="127.0.0.1",
        port=19000,
        inference_path=tmp_path / "inference.json",
    )

    with pytest.raises(RuntimeError, match="invalid output_ids"):
        server.result()

    assert server.inference_path.is_file()


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeClient:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, path: str) -> FakeResponse:
        return self.responses[path]

    def post(self, path: str, *, json: object) -> FakeResponse:
        return self.responses[path]


def server_launch(tmp_path: Path) -> E2eLaunch:
    model_root = tmp_path / "models"
    model_id = "organization/model"
    model_path = model_root / model_id
    model_path.mkdir(parents=True)
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19000},
            "vendor": {"model_base_uri": str(model_root)},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": model_id}],
        },
        env={},
    )
    return E2eLaunch(
        case_id="server",
        models=(E2eLaunchModel("model", model_id, "SyntheticForCausalLM", 16_384, 1, 1),),
        config=config,
        config_path=tmp_path / "xpool.toml",
        environment=MappingProxyType({}),
        observer_outdir=tmp_path / "observers",
        loopback_site=LoopbackSite.FFNAGENT,
    )
