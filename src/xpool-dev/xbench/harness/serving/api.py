"""Request-local OpenAI Completions protocol for benchmark endpoints."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import JsonValue

from xbench.harness.serving.case import BenchTarget, ServingEndpoint
from xbench.harness.serving.measure import StreamEvent, StreamProgress
from xbench.harness.serving.workload import ResolvedPrompt, ScheduledRequest


class ResponseProtocolError(ValueError):
    """Invalid response evidence with a rejected frame when available."""

    def __init__(self, message: str, event: StreamEvent | None = None) -> None:
        super().__init__(message)
        self.event = event


class OpenaiAdapter:
    """Encode one Completions request and validate its timestamped SSE frames."""

    def __init__(self) -> None:
        self.progress = StreamProgress()

    @staticmethod
    def payload(
        request: ScheduledRequest, prompt: ResolvedPrompt, target: BenchTarget, endpoint: ServingEndpoint
    ) -> dict[str, JsonValue]:
        return {
            "model": endpoint.served_model_name,
            "prompt": list(prompt.input_ids) if prompt.input_ids is not None else prompt.text,
            "max_tokens": request.max_new_tokens,
            "temperature": target.sampling.temperature,
            "ignore_eos": target.ignores_eos(),
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }

    def consume(self, data: bytes, *, request_id: str, sequence: int, observed_at_seconds: float) -> StreamEvent:
        """Validate one SSE data value after the client assigns its observation time."""
        kind: Literal["data", "usage", "done"] = "done" if data == b"[DONE]" else "data"
        completion = prompt = cached = None
        finish: JsonValue = None
        reported: JsonValue = None
        try:
            if kind != "done":
                payload: JsonValue = json.loads(data, parse_constant=reject_json_constant)
                if not isinstance(payload, dict):
                    raise ValueError("Completions frame must be a JSON object")
                if "error" in payload:
                    reported = payload["error"]
                    raise ValueError("Completions stream contains a server error")
                choices = payload.get("choices")
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("Completions frame requires a single choice or final usage")
                if choices:
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise ValueError("Completions choice must be an object")
                    finish = choice.get("finish_reason")
                else:
                    kind = "usage"
                usage = payload.get("usage")
                reported = {"usage": usage, "finish_reason": finish}
                if not isinstance(usage, dict):
                    raise ValueError("Completions frame lacks usage")
                completion = usage.get("completion_tokens")
                prompt = usage.get("prompt_tokens")
                if kind == "usage":
                    details = usage.get("prompt_tokens_details")
                    if isinstance(details, dict):
                        cached = details.get("cached_tokens")
            event = StreamEvent.model_validate(
                {
                    "request_id": request_id,
                    "sequence": sequence,
                    "observed_at_seconds": observed_at_seconds,
                    "kind": kind,
                    "completion_tokens": completion,
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "finish_reason": finish,
                    "reported_meta": reported,
                }
            )
            self.progress.accept(event)
            return event
        except ValueError as error:
            event = StreamEvent(
                request_id=request_id,
                sequence=sequence,
                observed_at_seconds=observed_at_seconds,
                kind=kind,
                accepted=False,
                reported_meta=reported,
                error_kind="protocol",
                error_message=str(error),
            )
            raise ResponseProtocolError(str(error), event) from error


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite Completions JSON constant: {value}")
