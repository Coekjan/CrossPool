from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from pydantic import JsonValue, TypeAdapter, ValidationError
from safetensors.torch import save_file

import xtest.harness.sglang.serving.server
from xkit.process import OwnedProcessGroup
from xkit.serving.sglang.endpoints import SglangEndpointFamily
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xkit.serving.sglang.launch import SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xtest.harness.sglang.serving.server import PROMPT, SglangInferenceRecord, SglangProbeServer
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.mark.parametrize(
    "graph_settings",
    [
        SglangGraphSettings("full", "disabled"),
        SglangGraphSettings("disabled", "breakable"),
        SglangGraphSettings("breakable", "full"),
    ],
)
def test_server_result_reads_resolved_modes_and_exact_output_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph_settings: SglangGraphSettings,
) -> None:
    responses = {
        "/server_info": FakeResponse(
            {
                "cuda_graph_config": {
                    "decode": {"backend": graph_settings.decode_backend},
                    "prefill": {"backend": graph_settings.prefill_backend},
                }
            }
        ),
        "/generate": FakeResponse({"output_ids": list(range(8))}),
    }
    monkeypatch.setattr(xtest.harness.sglang.serving.server.httpx, "Client", lambda **kwargs: FakeClient(responses))
    prefill_logits_path = tmp_path / "xpool.prefill-logits.123.safetensors"
    save_file(
        {"next_token_logits": torch.ones((1, 3), dtype=torch.float32)},
        prefill_logits_path,
        metadata={"rid": f"xpool-serving-graph-{TEST_MODEL_ID}"},
    )
    server = SglangProbeServer(
        process=SglangServerProcess(
            model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
            owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
            endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1),
        ),
        inference_path=tmp_path / "inference.json",
        prefill_logit_outdir=tmp_path,
    )

    result = server.result()

    assert result.model_id == TEST_MODEL_ID
    assert result.resolved_graph_settings == graph_settings
    assert result.output_ids == tuple(range(8))
    assert result.prefill_logits_path == prefill_logits_path
    record = TypeAdapter(SglangInferenceRecord).validate_json(server.inference_path.read_bytes())
    assert record.model_id == TEST_MODEL_ID
    assert record.url == "http://127.0.0.1:19000/generate"
    assert record.request["text"] == PROMPT
    assert record.request["rid"] == f"xpool-serving-graph-{TEST_MODEL_ID}"
    assert record.response_status_code == 200
    assert record.response == {"output_ids": list(range(8))}


@pytest.mark.parametrize(
    ("payload", "cause"),
    [
        ({}, KeyError),
        (None, TypeError),
        (
            {"cuda_graph_config": {"decode": {"backend": "disabled"}, "prefill": {"backend": "unknown"}}},
            ValidationError,
        ),
    ],
)
def test_server_result_rejects_invalid_graph_settings_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: JsonValue,
    cause: type[Exception],
) -> None:
    client = FakeClient({"/server_info": FakeResponse(payload)})
    monkeypatch.setattr(xtest.harness.sglang.serving.server.httpx, "Client", lambda **kwargs: client)
    server = SglangProbeServer(
        process=SglangServerProcess(
            model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
            owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
            endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1),
        ),
        inference_path=tmp_path / "inference.json",
    )

    with pytest.raises(RuntimeError, match="invalid CUDA Graph settings") as error:
        server.result()

    assert isinstance(error.value.__cause__, cause)
    assert client.post_paths == []
    assert not server.inference_path.exists()


def test_server_result_rejects_boolean_token_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/server_info": FakeResponse(
            {
                "cuda_graph_config": {
                    "decode": {"backend": "disabled"},
                    "prefill": {"backend": "disabled"},
                }
            }
        ),
        "/generate": FakeResponse({"output_ids": [0, 1, 2, 3, 4, 5, 6, True]}),
    }
    monkeypatch.setattr(xtest.harness.sglang.serving.server.httpx, "Client", lambda **kwargs: FakeClient(responses))
    server = SglangProbeServer(
        process=SglangServerProcess(
            model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
            owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
            endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1),
        ),
        inference_path=tmp_path / "inference.json",
    )

    with pytest.raises(RuntimeError, match="invalid output_ids"):
        server.result()

    assert server.inference_path.is_file()


class FakeResponse:
    def __init__(self, payload: JsonValue) -> None:
        self.payload = payload
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> JsonValue:
        return self.payload


class FakeClient:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.post_paths: list[str] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, path: str) -> FakeResponse:
        return self.responses[path]

    def post(self, path: str, *, json: object) -> FakeResponse:
        self.post_paths.append(path)
        return self.responses[path]
