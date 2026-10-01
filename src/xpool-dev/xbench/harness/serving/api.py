"""Request-local protocol adapters for benchmark streaming endpoints."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import JsonValue

from xbench.harness.serving.case import BenchTarget
from xbench.harness.serving.measure import StreamEvent
from xbench.harness.serving.workload import ResolvedPrompt, ScheduledRequest


class ResponseProtocolError(ValueError):
    """Invalid response evidence; retain the rejected frame when one is available."""

    def __init__(self, message: str, event: StreamEvent | None = None) -> None:
        super().__init__(message)
        self.event = event


class ApiAdapter(ABC):
    """Protocol construction and validation for one warmup or measured request.

    The client supplies event identity and seconds relative to its measurement
    origin. Adapters retain validation state, while timestamps, transport and
    event history remain client-owned.
    """

    @abstractmethod
    def endpoint(self, base_url: str) -> str:
        """Resolve the protocol endpoint from a declared serving base URL."""
        raise NotImplementedError

    @abstractmethod
    def payload(self, request: ScheduledRequest, prompt: ResolvedPrompt, target: BenchTarget) -> dict[str, JsonValue]:
        """Construct request content from validated, prepared workload values."""
        raise NotImplementedError

    @abstractmethod
    def consume(self, data: bytes, *, request_id: str, sequence: int, observed_at_seconds: float) -> StreamEvent:
        """Validate a timestamped data frame or protocol termination.

        Raises ResponseProtocolError with the rejected event on malformed input.
        The returned event uses the existing observation contract; it contains
        no response-text history or independently observed token timestamps.
        """
        raise NotImplementedError


class SglangAdapter(ApiAdapter):
    """Native /generate parsing with cumulative counts and terminal usage."""

    def __init__(self) -> None:
        self.completion_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.finish_reason: str | None = None

    def endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/generate"

    def payload(self, request: ScheduledRequest, prompt: ResolvedPrompt, target: BenchTarget) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "rid": request.request_id,
            "stream": True,
            "sampling_params": {
                "temperature": target.sampling.temperature,
                "max_new_tokens": request.max_new_tokens,
                "ignore_eos": target.ignores_eos(),
                "stream_interval": target.sampling.stream_interval,
            },
        }
        if prompt.input_ids is not None:
            payload["input_ids"] = list(prompt.input_ids)
        else:
            payload["text"] = prompt.text
        return payload

    def consume(self, data: bytes, *, request_id: str, sequence: int, observed_at_seconds: float) -> StreamEvent:
        reported: JsonValue = None
        kind: Literal["data", "done"] = "done" if data == b"[DONE]" else "data"
        try:
            if kind == "done":
                if (
                    self.finish_reason not in {"stop", "length"}
                    or self.prompt_tokens is None
                    or self.completion_tokens is None
                ):
                    raise ValueError("DONE lacks normal terminal finish reason and valid final usage")
                return StreamEvent(
                    request_id=request_id, sequence=sequence, observed_at_seconds=observed_at_seconds, kind="done"
                )
            payload: JsonValue = json.loads(data, parse_constant=reject_json_constant)
            if not isinstance(payload, dict):
                raise ValueError("native response must be a JSON object")
            if "error" in payload:
                reported = payload["error"]
                raise ValueError("native stream contains a server error")
            meta = payload.get("meta_info")
            if not isinstance(meta, dict):
                raise ValueError("native response lacks object meta_info")
            reported = {
                key: meta[key]
                for key in ("completion_tokens", "prompt_tokens", "cached_tokens", "finish_reason")
                if key in meta
            }

            def count(name: str) -> int | None:
                value = meta.get(name)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise ValueError(f"invalid native {name}")
                return value

            completion = count("completion_tokens")
            prompt = count("prompt_tokens")
            cached = count("cached_tokens")
            if completion is not None and self.completion_tokens is not None and completion < self.completion_tokens:
                raise ValueError("native completion count regressed")
            reason = meta.get("finish_reason")
            finish = reason.get("type") if isinstance(reason, dict) else None
            if reason is not None and (not isinstance(finish, str) or finish not in {"stop", "length"}):
                raise ValueError("native finish reason is not normal stop or length")
            if finish is not None and (completion is None or prompt is None):
                raise ValueError("terminal native response lacks final token usage")
            if cached is not None and prompt is not None and cached > prompt:
                raise ValueError("native cached_tokens exceeds logical prompt_tokens")
        except (ValueError, UnicodeError) as error:
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
        event = StreamEvent(
            request_id=request_id,
            sequence=sequence,
            observed_at_seconds=observed_at_seconds,
            kind="data",
            completion_tokens=completion,
            prompt_tokens=prompt,
            cached_tokens=cached,
            finish_reason=finish,
            reported_meta=reported,
        )
        if completion is not None:
            self.completion_tokens = completion
        if prompt is not None:
            self.prompt_tokens = prompt
        if finish is not None:
            self.finish_reason = finish
        return event


class OpenaiAdapter(ApiAdapter):
    """Explicit boundary for future OpenAI-compatible protocol execution."""

    def __init__(self) -> None:
        raise NotImplementedError("openai-compatible API execution is not implemented")

    def endpoint(self, base_url: str) -> str:
        raise NotImplementedError

    def payload(self, request: ScheduledRequest, prompt: ResolvedPrompt, target: BenchTarget) -> dict[str, JsonValue]:
        raise NotImplementedError

    def consume(self, data: bytes, *, request_id: str, sequence: int, observed_at_seconds: float) -> StreamEvent:
        raise NotImplementedError


def create_api_adapter(api: Literal["sglang", "openai"]) -> ApiAdapter:
    """Create fresh request-local state; unsupported execution fails at preflight."""
    match api:
        case "sglang":
            return SglangAdapter()
        case "openai":
            return OpenaiAdapter()


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite native JSON constant: {value}")
