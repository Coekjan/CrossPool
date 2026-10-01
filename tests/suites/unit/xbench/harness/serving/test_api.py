import json
from pathlib import Path

import pytest

from xbench.harness.serving.api import ResponseProtocolError, create_api_adapter
from xbench.harness.serving.case import ClientTarget, JsonlPrompts
from xbench.harness.serving.workload import ResolvedPrompt, ScheduledRequest
from xtest.harness.support.config import TEST_MODEL_ID


def payload(count: int, *, finish: str | None = None) -> bytes:
    return json.dumps(
        {
            "text": "",
            "meta_info": {
                "completion_tokens": count,
                "prompt_tokens": 4,
                "cached_tokens": 2,
                "finish_reason": {"type": finish} if finish is not None else None,
            },
        }
    ).encode()


def test_native_adapter_constructs_endpoint_and_prepared_token_payload() -> None:
    target = ClientTarget(
        model_id=TEST_MODEL_ID,
        base_url="http://localhost:8000/base",
        prompts=JsonlPrompts(kind="jsonl", path=Path("p.jsonl")),
    )
    request = ScheduledRequest(
        request_id="r", model_id=TEST_MODEL_ID, arrival_seconds=0.0, prompt_id="p", max_new_tokens=10
    )
    prompt = ResolvedPrompt(model_id=TEST_MODEL_ID, prompt_id="p", input_ids=(1, 2))
    adapter = create_api_adapter(target.api)
    assert adapter.endpoint(target.base_url) == "http://localhost:8000/base/generate"
    assert adapter.payload(request, prompt, target) == {
        "rid": "r",
        "stream": True,
        "input_ids": [1, 2],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 10, "ignore_eos": False, "stream_interval": 1},
    }


def test_native_stream_accepts_empty_text_progress_and_valid_done() -> None:
    adapter = create_api_adapter("sglang")
    frames = (payload(3), payload(3), payload(6, finish="length"), b"[DONE]")
    events = [
        adapter.consume(frame, request_id="r", sequence=index, observed_at_seconds=0.1 * index)
        for index, frame in enumerate(frames)
    ]
    assert [event.completion_tokens for event in events] == [3, 3, 6, None]
    assert all(event.accepted for event in events)
    assert events[-1].kind == "done"


@pytest.mark.parametrize(
    "invalid", [payload(2), payload(True), payload(-1), payload(4, finish="abort"), b"null", b'{"error":{"code":400}}']
)
def test_invalid_stream_carries_the_rejected_event(invalid: bytes) -> None:
    adapter = create_api_adapter("sglang")
    adapter.consume(payload(3), request_id="r", sequence=0, observed_at_seconds=0.1)
    with pytest.raises(ResponseProtocolError) as caught:
        adapter.consume(invalid, request_id="r", sequence=1, observed_at_seconds=0.2)
    event = caught.value.event
    assert event is not None and not event.accepted
    assert event.error_kind == "protocol" and event.request_id == "r"
    assert event.sequence == 1 and event.observed_at_seconds == 0.2


def test_done_requires_normal_terminal_usage_and_accepts_zero_tokens() -> None:
    adapter = create_api_adapter("sglang")
    with pytest.raises(ResponseProtocolError, match="final usage"):
        adapter.consume(b"[DONE]", request_id="r", sequence=0, observed_at_seconds=0.1)
    event = adapter.consume(payload(0, finish="stop"), request_id="r", sequence=1, observed_at_seconds=0.2)
    assert event.completion_tokens == 0
    assert adapter.consume(b"[DONE]", request_id="r", sequence=2, observed_at_seconds=0.3).kind == "done"
