"""Streaming observation and open-loop FIFO admission without request drops."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import JsonValue

from xbench.harness.serving.api import ApiAdapter, ResponseProtocolError, create_api_adapter
from xbench.harness.serving.case import BenchCase
from xbench.harness.serving.measure import MeasurementOrigin, RepetitionManifest, RequestRecord, StreamEvent
from xbench.harness.serving.workload import PreparedWorkload, ScheduledRequest
from xkit.results import write_json
from xpool.model import ModelId

MAX_SSE_FRAME_BYTES = 16 * 1024 * 1024
CLIENT_CLEANUP_SECONDS = 5.0


class MeasurementRecorder:
    """Buffer raw records and flush terminal requests; I/O loss is infrastructure failure."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.events: list[StreamEvent] = []
        self.requests: list[RequestRecord] = []
        self.event_file = (directory / "events.jsonl").open("x", encoding="utf-8", buffering=64 * 1024)
        try:
            self.request_file = (directory / "requests.jsonl").open("x", encoding="utf-8", buffering=64 * 1024)
        except BaseException:
            self.event_file.close()
            raise
        self.request_ids: set[str] = set()
        self.failed = False
        self.window: RepetitionManifest | None = None

    def seal_window(self, end: float, kind: Literal["complete", "interrupted"]) -> None:
        """Retain the stop before teardown; late terminal evidence can extend an interruption."""

        if self.window is None:
            self.window = RepetitionManifest.model_validate_json((self.directory / "repetition.json").read_bytes())
        if self.window.window_kind == "complete":
            return
        previous = self.window.window_end_seconds
        if previous is not None:
            if previous >= end and self.window.window_kind == kind:
                return
            end = max(previous, end)
        self.window = self.window.model_copy(update={"window_end_seconds": end, "window_kind": kind})
        write_json(self.directory / "repetition.json", self.window.model_dump(mode="json"))

    def event(self, event: StreamEvent) -> None:
        try:
            self.event_file.write(event.model_dump_json() + "\n")
        except (OSError, ValueError):
            self.failed = True
            raise
        self.events.append(event)
        if self.window is not None and self.window.window_kind == "interrupted":
            self.seal_window(event.observed_at_seconds, "interrupted")

    def request(self, request: RequestRecord) -> None:
        if request.request_id in self.request_ids:
            raise ValueError(f"duplicate terminal request record: {request.request_id}")
        try:
            self.request_file.write(request.model_dump_json() + "\n")
            self.event_file.flush()
            self.request_file.flush()
        except (OSError, ValueError):
            self.failed = True
            raise
        self.request_ids.add(request.request_id)
        self.requests.append(request)
        if (
            self.window is not None
            and self.window.window_kind == "interrupted"
            and request.ended_at_seconds is not None
        ):
            self.seal_window(request.ended_at_seconds, "interrupted")

    def close(self) -> None:
        failures = []
        for output in (self.event_file, self.request_file):
            if output.closed:
                continue
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError as error:
                self.failed = True
                failures.append(error)
            finally:
                output.close()
        if failures:
            raise failures[0]


@dataclass(slots=True)
class RequestState:
    """One request's accepted prefix, positive progress anchor and terminal evidence."""

    request: ScheduledRequest
    recorder: MeasurementRecorder
    adapter: ApiAdapter
    enqueued_at: float | None = None
    started_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    first_count: int | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    status: int | None = None
    events: list[StreamEvent] = field(default_factory=list)

    def event(
        self,
        kind: Literal["enqueue", "http_start", "data", "done", "error"],
        when: float,
        *,
        completion_tokens: int | None = None,
        prompt_tokens: int | None = None,
        cached_tokens: int | None = None,
        finish_reason: str | None = None,
        accepted: bool = True,
        reported_meta: JsonValue = None,
        error_kind: str | None = None,
        error_message: str | None = None,
    ) -> None:
        event = StreamEvent(
            request_id=self.request.request_id,
            sequence=len(self.events),
            observed_at_seconds=when,
            kind=kind,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            finish_reason=finish_reason,
            accepted=accepted,
            reported_meta=reported_meta,
            error_kind=error_kind,
            error_message=error_message,
        )
        self.recorder.event(event)
        self.events.append(event)

    def consume(self, data: bytes, when: float) -> bool:
        """Consume one timestamped SSE data event, returning true only at valid DONE."""

        try:
            event = self.adapter.consume(
                data, request_id=self.request.request_id, sequence=len(self.events), observed_at_seconds=when
            )
        except ResponseProtocolError as error:
            if error.event is not None:
                self.recorder.event(error.event)
                self.events.append(error.event)
            raise
        self.recorder.event(event)
        self.events.append(event)
        completion = event.completion_tokens
        if completion is not None:
            previous = self.completion_tokens or 0
            if completion > previous:
                if self.first_token_at is None:
                    self.first_token_at = when
                    self.first_count = completion
                self.last_token_at = when
            self.completion_tokens = completion
        if event.prompt_tokens is not None:
            self.prompt_tokens = event.prompt_tokens
        if event.cached_tokens is not None:
            self.cached_tokens = event.cached_tokens
        if event.finish_reason is not None:
            self.finish_reason = event.finish_reason
        return event.kind == "done"

    def terminal(
        self,
        outcome: Literal["success", "failed", "cancelled", "not_sent"],
        end: float | None,
        *,
        error_kind: str | None = None,
        error_message: str | None = None,
    ) -> RequestRecord:
        return RequestRecord(
            **self.request.model_dump(),
            outcome=outcome,
            enqueued_at_seconds=self.enqueued_at,
            http_started_at_seconds=self.started_at,
            first_token_at_seconds=self.first_token_at,
            last_token_at_seconds=self.last_token_at,
            ended_at_seconds=end,
            first_completion_tokens=self.first_count,
            completion_tokens=self.completion_tokens,
            prompt_tokens=self.prompt_tokens,
            cached_tokens=self.cached_tokens,
            http_status=self.status,
            finish_reason=self.finish_reason,
            error_kind=error_kind,
            error_message=error_message,
        ).measured(self.events)


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, JsonValue],
    state: RequestState,
    *,
    origin: float,
    timeout_seconds: float | None,
) -> RequestRecord:
    """Record termination before closing transport outside the latency clock.

    Response-close failures propagate as infrastructure errors after the terminal
    request has been flushed to its recorder.
    """

    response = None
    record = None
    request = client.build_request("POST", state.adapter.endpoint(base_url), json=payload)
    state.started_at = time.perf_counter() - origin
    state.event("http_start", state.started_at)
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await client.send(request, stream=True)
            state.status = response.status_code
            if response.status_code != 200:
                raise ResponseProtocolError(f"HTTP status is {response.status_code}, expected 200")
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ResponseProtocolError("streaming response must use text/event-stream")
            buffer = bytearray()
            data_lines: list[bytes] = []
            frame_bytes = 0
            async for chunk in response.aiter_raw():
                observed = time.perf_counter() - origin
                offset = 0
                while offset < len(chunk):
                    newline = chunk.find(b"\n", offset)
                    stop = len(chunk) if newline < 0 else newline + 1
                    frame_bytes += stop - offset
                    if frame_bytes > MAX_SSE_FRAME_BYTES:
                        raise ResponseProtocolError("SSE frame exceeds the supported size")
                    buffer.extend(chunk[offset:stop])
                    offset = stop
                    if newline < 0:
                        break
                    line = bytes(buffer[:-1]).removesuffix(b"\r")
                    buffer.clear()
                    if not line:
                        if data_lines and state.consume(b"\n".join(data_lines), observed):
                            record = state.terminal("success", observed)
                            return record
                        data_lines.clear()
                        frame_bytes = 0
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].removeprefix(b" "))
            raise ResponseProtocolError("stream ended without protocol termination")
    except (ResponseProtocolError, httpx.HTTPError, TimeoutError) as error:
        ended = time.perf_counter() - origin
        kind = (
            "timeout"
            if isinstance(error, TimeoutError)
            else "protocol"
            if isinstance(error, ResponseProtocolError)
            else "transport"
        )
        # HTTP diagnostics omit credential-bearing request URLs and response text.
        message = str(error) if isinstance(error, ResponseProtocolError) else type(error).__name__
        state.event("error", ended, error_kind=kind, error_message=message)
        record = state.terminal("failed", ended, error_kind=kind, error_message=message)
        return record
    except asyncio.CancelledError:
        ended = time.perf_counter() - origin
        state.event("error", ended, error_kind="cancelled", error_message="HTTP request cancelled")
        record = state.terminal("cancelled", ended, error_kind="cancelled", error_message="HTTP request cancelled")
        return record
    finally:
        try:
            if record is not None:
                state.recorder.request(record)
        finally:
            if response is not None:
                async with asyncio.timeout(CLIENT_CLEANUP_SECONDS):
                    await response.aclose()


async def observe_system(check_alive: Callable[[], None] | None, stopped: asyncio.Event) -> None:
    while not stopped.is_set():
        if check_alive is not None:
            check_alive()
        try:
            await asyncio.wait_for(stopped.wait(), timeout=0.1)
        except TimeoutError:
            pass


async def run_measurement(
    case: BenchCase,
    workload: PreparedWorkload,
    endpoints: Mapping[ModelId, str],
    recorder: MeasurementRecorder,
    *,
    check_alive: Callable[[], None] | None = None,
) -> float:
    """Dispatch all finite arrivals through one unbounded FIFO and global active limit.

    Warmup is performed separately by the runner before this common origin.
    Optional request deadlines exclude queue waiting. Individual failures do not
    interrupt drain; process/recording failures cancel all workers and account for every
    planned request. The return value is the arrival/drain window, excluding
    HTTP-resource and serving cleanup.
    """

    prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
    targets = {target.model_id: target for target in case.targets}
    adapters = {request.request_id: create_api_adapter(targets[request.model_id].api) for request in workload.requests}
    payloads = {
        request.request_id: adapters[request.request_id].payload(
            request, prompts[request.model_id, request.prompt_id], targets[request.model_id]
        )
        for request in workload.requests
    }
    states: dict[str, RequestState] = {}
    queue: asyncio.Queue[RequestState | None] = asyncio.Queue()
    stopped = asyncio.Event()
    window_end = workload.arrival_horizon_seconds
    origin = time.perf_counter()
    write_json(
        recorder.directory / "measurement.json",
        MeasurementOrigin(monotonic_origin_seconds=origin, wall_clock_origin_seconds=time.time()).model_dump(
            mode="json"
        ),
    )
    limits = httpx.Limits(max_connections=case.max_inflight, max_keepalive_connections=case.max_inflight)
    async with httpx.AsyncClient(timeout=None, limits=limits, follow_redirects=False) as client:

        async def produce() -> None:
            nonlocal window_end
            try:
                for request in workload.requests:
                    delay = origin + request.arrival_seconds - time.perf_counter()
                    while delay > 0:
                        await asyncio.sleep(delay)
                        delay = origin + request.arrival_seconds - time.perf_counter()
                    enqueued_at = time.perf_counter() - origin
                    state = RequestState(request, recorder, adapters[request.request_id], enqueued_at=enqueued_at)
                    states[request.request_id] = state
                    state.event("enqueue", enqueued_at)
                    queue.put_nowait(state)
                await queue.join()
                remaining_horizon = origin + workload.arrival_horizon_seconds - time.perf_counter()
                if remaining_horizon > 0:
                    await asyncio.sleep(remaining_horizon)
            except BaseException:
                # The producer observes cancellation before waiting for HTTP teardown.
                recorder.seal_window(time.perf_counter() - origin, "interrupted")
                raise
            end = max((record.ended_at_seconds or 0.0 for record in recorder.requests), default=0.0)
            window_end = max(workload.arrival_horizon_seconds, end)
            recorder.seal_window(window_end, "complete")
            for _ in range(case.max_inflight):
                queue.put_nowait(None)
            stopped.set()

        async def work() -> None:
            while True:
                state = await queue.get()
                if state is None:
                    queue.task_done()
                    return
                try:
                    record = await send_request(
                        client,
                        endpoints[state.request.model_id],
                        payloads[state.request.request_id],
                        state,
                        origin=origin,
                        timeout_seconds=case.request_timeout_seconds,
                    )
                    if record.outcome == "cancelled":
                        raise asyncio.CancelledError
                finally:
                    queue.task_done()

        async def progress() -> None:
            while not stopped.is_set():
                now = time.perf_counter() - origin
                terminal = recorder.requests
                started = sum(state.started_at is not None for state in states.values())
                provisional_rate = sum(state.completion_tokens or 0 for state in states.values()) / max(now, 1e-9)
                print(
                    f"xbench phase=measure elapsed={now:.1f}s planned={len(workload.requests)} enqueued={len(states)} "
                    f"queued={len(states) - started} in-flight={started - len(terminal)} completed={len(terminal)} "
                    f"successful={sum(record.outcome == 'success' for record in terminal)} "
                    f"failed={sum(record.outcome == 'failed' for record in terminal)} "
                    f"provisional-output-tokens/s={provisional_rate:.1f}",
                    file=sys.stderr,
                )
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=1.0)
                except TimeoutError:
                    pass

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(produce())
                for _ in range(case.max_inflight):
                    group.create_task(work())
                group.create_task(observe_system(check_alive, stopped))
                group.create_task(progress())
        finally:
            stopped.set()
            if not recorder.failed:
                for request in workload.requests:
                    if request.request_id not in recorder.request_ids:
                        state = states.get(request.request_id)
                        if state is None:
                            state = RequestState(request, recorder, adapters[request.request_id])
                        if state.started_at is None:
                            record = state.terminal(
                                "not_sent",
                                None,
                                error_kind="interrupted",
                                error_message="measurement interrupted before HTTP dispatch",
                            )
                        else:
                            record = RequestRecord(
                                **request.model_dump(),
                                outcome="failed",
                                error_kind="evidence_missing",
                                error_message="terminal HTTP evidence unavailable",
                            )
                        recorder.request(record)
    return window_end
