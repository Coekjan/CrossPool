import json
from pathlib import Path

import pytest

from xbench.harness.serving.api import OpenaiAdapter, ResponseProtocolError
from xbench.harness.serving.case import ClientTarget, JsonlPrompts, ServingEndpoint
from xbench.harness.serving.workload import ResolvedPrompt, ScheduledRequest
from xtest.harness.support.config import TEST_MODEL_ID


def frame(count: int, *, finish: str | None = None, final: bool = False) -> bytes:
    usage: dict[str, object] = {"completion_tokens": count, "prompt_tokens": 4}
    if final:
        usage["prompt_tokens_details"] = {"cached_tokens": 2}
    return json.dumps({"choices": [] if final else [{"text": "", "finish_reason": finish}], "usage": usage}).encode()


def test_openai_adapter_constructs_endpoint_and_prepared_token_payload() -> None:
    target = ClientTarget(
        model_id=TEST_MODEL_ID,
        base_url="http://localhost:8000/",
        served_model_name="baseline",
        prompts=JsonlPrompts(kind="jsonl", path=Path("p.jsonl")),
    )
    endpoint = ServingEndpoint(base_url=target.base_url, served_model_name="baseline")
    request = ScheduledRequest(
        request_id="r", model_id=TEST_MODEL_ID, arrival_seconds=0.0, prompt_id="p", max_new_tokens=10
    )
    prompt = ResolvedPrompt(model_id=TEST_MODEL_ID, prompt_id="p", input_ids=(1, 2))
    assert OpenaiAdapter.payload(request, prompt, target, endpoint) == {
        "model": "baseline",
        "prompt": [1, 2],
        "max_tokens": 10,
        "temperature": 0.0,
        "ignore_eos": False,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }


def test_stream_accepts_empty_text_progress_and_separate_final_usage() -> None:
    adapter = OpenaiAdapter()
    frames = (frame(3), frame(3), frame(6, finish="length"), frame(6, final=True), b"[DONE]")
    events = [
        adapter.consume(value, request_id="r", sequence=index, observed_at_seconds=0.1 * index)
        for index, value in enumerate(frames)
    ]
    assert [event.completion_tokens for event in events] == [3, 3, 6, 6, None]
    assert [event.kind for event in events] == ["data", "data", "data", "usage", "done"]
    assert events[-2].cached_tokens == 2


@pytest.mark.parametrize(
    "invalid",
    [
        frame(2),
        frame(True),
        frame(-1),
        frame(4, finish="abort"),
        frame(3, final=True),
        b"null",
        b'{"error":{"code":400}}',
        b'{"choices":[{"text":"answer"}]}',
        b'{"choices":[{}],"usage":{"completion_tokens":4,"extra":NaN}}',
    ],
)
def test_invalid_stream_carries_the_rejected_event(invalid: bytes) -> None:
    adapter = OpenaiAdapter()
    adapter.consume(frame(3), request_id="r", sequence=0, observed_at_seconds=0.1)
    with pytest.raises(ResponseProtocolError) as caught:
        adapter.consume(invalid, request_id="r", sequence=1, observed_at_seconds=0.2)
    event = caught.value.event
    assert event is not None and not event.accepted
    assert event.error_kind == "protocol" and event.request_id == "r"
    assert event.sequence == 1 and event.observed_at_seconds == 0.2


@pytest.mark.parametrize("invalid", [b"[DONE]", frame(1, final=True)])
def test_finish_requires_matching_final_usage_before_done(invalid: bytes) -> None:
    adapter = OpenaiAdapter()
    assert (
        adapter.consume(frame(0, finish="stop"), request_id="r", sequence=0, observed_at_seconds=0.1).completion_tokens
        == 0
    )
    with pytest.raises(ResponseProtocolError, match="final usage"):
        adapter.consume(invalid, request_id="r", sequence=1, observed_at_seconds=0.2)
    adapter.consume(frame(0, final=True), request_id="r", sequence=2, observed_at_seconds=0.3)
    assert adapter.consume(b"[DONE]", request_id="r", sequence=3, observed_at_seconds=0.4).kind == "done"
