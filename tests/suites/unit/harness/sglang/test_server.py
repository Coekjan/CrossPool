from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest
from pydantic import TypeAdapter

import tests.harness.sglang.server
from tests.harness.process import OwnedProcessGroup
from tests.harness.sglang.e2e import E2eLaunch, E2eLaunchModel
from tests.harness.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from tests.harness.sglang.graph import SglangGraphSettings
from tests.harness.sglang.server import (
    PROMPT,
    SglangEndpointConflict,
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
    assert server.inference_path == tmp_path / "organization-model.inference.json"


@pytest.mark.parametrize(
    ("log", "expected_name", "expected_port"),
    [
        (
            "ValueError: metrics_port at 20237 is not available in 30 seconds.",
            "metrics_port",
            20_237,
        ),
        (
            "ZMQError: Address already in use (addr='tcp://127.0.0.1:20236')",
            "tcp_bind",
            20_236,
        ),
    ],
)
def test_endpoint_conflict_parses_only_owned_family_members(
    log: str,
    expected_name: str,
    expected_port: int,
) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)

    conflict = SglangEndpointConflict.from_log(family, log)

    assert conflict is not None
    assert conflict.endpoint_name == expected_name
    assert conflict.port == expected_port


@pytest.mark.parametrize(
    "log",
    [
        "ValueError: metrics_port at 30000 is not available in 30 seconds.",
        "ZMQError: Address already in use (addr='tcp://127.0.0.1:30000')",
        "SGLang exited during model loading",
    ],
)
def test_endpoint_conflict_rejects_unknown_or_unstructured_diagnostics(log: str) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)

    assert SglangEndpointConflict.from_log(family, log) is None


def test_server_health_classifies_owned_endpoint_after_early_exit() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: 1),
            tail=lambda: "ValueError: metrics_port at 20237 is not available in 30 seconds.",
        ),
    )
    server = SglangServerProcess(
        model=E2eLaunchModel("model", "organization/model", "SyntheticForCausalLM", 16_384, 1, 2),
        owner=owner,
        endpoint=cast(SglangEndpointFamilyLease, SimpleNamespace(family=family)),
        host=family.host,
        port=family.http_port,
        inference_path=Path("inference.json"),
    )

    with pytest.raises(SglangEndpointConflict) as error:
        server.healthy()

    assert error.value.endpoint_name == "metrics_port"
    assert error.value.port == 20_237


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
        endpoint=cast(SglangEndpointFamilyLease, SimpleNamespace(close=lambda: None)),
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
        endpoint=cast(SglangEndpointFamilyLease, SimpleNamespace(close=lambda: None)),
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
