from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import torch
from pydantic import JsonValue, TypeAdapter
from safetensors.torch import load_file

import xtest.harness.sglang.serving.server
from xkit.process import OwnedProcessGroup
from xkit.serving.sglang.endpoints import SglangEndpointFamily
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xtest.harness.sglang.serving.server import SglangInferenceRecord, SglangProbeServer, generation_request
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.fixture
def server() -> SglangProbeServer:
    return SglangProbeServer(
        SglangServerProcess(
            model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.DECODE_FULL.settings()),
            owner=cast(OwnedProcessGroup, SimpleNamespace(name="sglang-test")),
            endpoint=SglangEndpointFamily("127.0.0.1", 19000, 19001, 19002, 1),
        )
    )


def install_response(monkeypatch: pytest.MonkeyPatch, payload: JsonValue, *, status: int = 200) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/server_info":
            return httpx.Response(
                200, json={"cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}}
            )
        return httpx.Response(status, json=payload)

    client = httpx.Client(base_url="http://127.0.0.1:19000", transport=httpx.MockTransport(respond))
    monkeypatch.setattr(xtest.harness.sglang.serving.server.httpx, "Client", lambda **kwargs: client)


def test_generation_retains_conditioning_and_full_vocabulary_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: SglangProbeServer
) -> None:
    request = generation_request("test", input_ids=(1, 2), vocab_size=3, max_new_tokens=2)
    rows: JsonValue = [
        [[-3.0, 2, None], [-1.0, 0, None], [-2.0, 1, None]],
        [[-4.0, 0, None], [-5.0, 1, None], [-6.0, 2, None]],
    ]
    response: JsonValue = {
        "prompt_token_ids": [1, 2],
        "output_ids": [0, 1],
        "meta_info": {"cached_tokens": 1, "output_token_ids_logprobs": rows},
    }
    install_response(monkeypatch, response)
    path = tmp_path / "request.json"

    result = server.generate(request, record_path=path)

    assert result.model_id == TEST_MODEL_ID
    assert result.resolved_graph_settings == SglangGraphMode.DECODE_FULL.settings()
    assert result.request_id == "test" and result.cached_tokens == 1
    assert result.input_ids + result.output_ids == (1, 2, 0, 1)
    assert result.prefix_lengths == (2, 3)
    assert result.logprobs_path is not None
    torch.testing.assert_close(
        load_file(result.logprobs_path)["logprobs"], torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    )
    record = TypeAdapter(SglangInferenceRecord).validate_json(path.read_bytes())
    assert record.request == request and record.response == response
    assert record.started_ns <= record.finished_ns


@pytest.mark.parametrize(
    "bad_rows", [None, [], [[[0.0, 0, None], [0.0, 0, None]]], [[[1e100, 0, None], [0.0, 1, None]]]]
)
def test_rejects_missing_or_malformed_distributions_after_retaining_http_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: SglangProbeServer, bad_rows: JsonValue
) -> None:
    request = generation_request("invalid", input_ids=(1,), vocab_size=2, max_new_tokens=1)
    payload: JsonValue = {
        "prompt_token_ids": [1],
        "output_ids": [0],
        "meta_info": {"output_token_ids_logprobs": bad_rows},
    }
    install_response(monkeypatch, payload)
    path = tmp_path / "invalid.json"

    with pytest.raises((ValueError, RuntimeError)):
        server.generate(request, record_path=path)

    assert path.is_file()
    assert not path.with_suffix(".logprobs.safetensors").exists()


def test_retains_failed_http_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: SglangProbeServer
) -> None:
    install_response(monkeypatch, {"error": "rejected"}, status=400)
    path = tmp_path / "failed.json"

    with pytest.raises(httpx.HTTPStatusError):
        server.generate(generation_request("failed"), record_path=path)

    record = TypeAdapter(SglangInferenceRecord).validate_json(path.read_bytes())
    assert record.response_status_code == 400 and record.response == {"error": "rejected"}


@pytest.mark.parametrize("invalid_id", [True, -1])
def test_rejects_invalid_token_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: SglangProbeServer, invalid_id: bool | int
) -> None:
    install_response(monkeypatch, {"prompt_token_ids": [1], "output_ids": [invalid_id], "meta_info": {}})
    request = generation_request("invalid-ids", max_new_tokens=1)
    with pytest.raises(ValueError):
        server.generate(request, record_path=tmp_path / "ids.json")
