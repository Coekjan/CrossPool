from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType

import httpx
import pytest
from benches.suites.serving.multi_model import run_serving
from tests.suites.integration.xbench.support import FakeServingServer, client_case

import xbench.harness.serving.client
from xbench.harness.serving.api import create_api_adapter
from xbench.harness.serving.case import BenchCase, ClientBenchCase
from xbench.harness.serving.client import MeasurementRecorder, RequestState, run_measurement, send_request
from xbench.harness.serving.execution import warmup
from xbench.harness.serving.measure import BenchCaseManifest, RepetitionManifest, RequestRecord, summarize
from xbench.harness.serving.runner import finalize_repetition
from xbench.harness.serving.workload import PreparedWorkload, prepare_workload, read_jsonl
from xkit.results import write_json, write_jsonl
from xkit.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeState
from xtest.harness.support.config import TEST_MODEL_ID


def test_native_http_stream_empty_text_usage_done_and_final_artifacts(tmp_path: Path) -> None:
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url)
        workload = retain_workload(case, tmp_path / "result")
        recorder = MeasurementRecorder(tmp_path / "result")
        try:
            end = asyncio.run(run_measurement(case, workload, {TEST_MODEL_ID: server.url}, recorder))
        finally:
            recorder.close()
        summary = summarize(
            workload, recorder.requests, recorder.events, window_end_seconds=end, window_kind="complete"
        )
        assert summary.execution_complete and summary.measurement_available
        assert server.received == ["first", "second", "third"]
        assert all(
            request.outcome == "success" and request.metrics.http_ttft_seconds is not None
            for request in recorder.requests
        )
        assert summary.targets["aggregate"].input_tokens == 15
        assert summary.targets["aggregate"].output_tokens == 9
        assert len((tmp_path / "result" / "requests.jsonl").read_text().splitlines()) == 3
        assert "text" not in (tmp_path / "result" / "events.jsonl").read_text()


@pytest.mark.parametrize("oversized_frame", [False, True])
def test_sse_limit_counts_wire_frames_across_chunks_not_transport_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized_frame: bool
) -> None:
    monkeypatch.setattr(xbench.harness.serving.client, "MAX_SSE_FRAME_BYTES", 512)
    frame = (
        b'data: {"text":"","meta_info":{"completion_tokens":3,"prompt_tokens":5,'
        b'"cached_tokens":4,"finish_reason":{"type":"length"}}}\n\n'
    )
    chunks = [b"data:\n"] * 100 + [frame, b"data: [DONE]\n\n"] if oversized_frame else [frame * 5 + b"data: [DONE]\n\n"]

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

    case = client_case(tmp_path, "http://127.0.0.1:1")
    workload = prepare_workload(case)
    recorder = MeasurementRecorder(tmp_path / "result")
    adapter = create_api_adapter("sglang")
    state = RequestState(workload.requests[0], recorder, adapter, enqueued_at=0.0)
    state.event("enqueue", 0.0)

    async def execute() -> RequestRecord:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())
        )
        async with httpx.AsyncClient(transport=transport) as client:
            return await send_request(
                client,
                case.targets[0].base_url,
                adapter.payload(state.request, workload.prompts[0], case.targets[0]),
                state,
                origin=time.perf_counter(),
                timeout_seconds=5.0,
            )

    try:
        record = asyncio.run(execute())
    finally:
        recorder.close()
    assert record.outcome == ("failed" if oversized_frame else "success")
    if oversized_frame:
        assert record.error_kind == "protocol" and record.error_message == "SSE frame exceeds the supported size"


def test_one_timeout_does_not_stop_fifo(tmp_path: Path) -> None:
    with FakeServingServer(block_first=True) as server:
        original = client_case(tmp_path, server.url)
        case = ClientBenchCase.model_validate({**original.model_dump(), "request_timeout_seconds": 2.0})
        workload = retain_workload(case, tmp_path / "result")
        recorder = MeasurementRecorder(tmp_path / "result")
        try:
            asyncio.run(run_measurement(case, workload, {TEST_MODEL_ID: server.url}, recorder))
        finally:
            recorder.close()
        assert server.received == ["first", "second", "third"]
        assert tuple(record.outcome for record in recorder.requests) == ("failed", "success", "success")
        assert recorder.requests[0].error_kind == "timeout"


def test_http_200_without_done_is_a_failed_request_not_an_infrastructure_stop(tmp_path: Path) -> None:
    with FakeServingServer(omit_done=True) as server:
        case = client_case(tmp_path, server.url)
        workload = retain_workload(case, tmp_path / "result")
        recorder = MeasurementRecorder(tmp_path / "result")
        try:
            asyncio.run(run_measurement(case, workload, {TEST_MODEL_ID: server.url}, recorder))
        finally:
            recorder.close()
        assert len(recorder.requests) == 3
        assert all(record.outcome == "failed" and record.completion_tokens == 3 for record in recorder.requests)
        assert all(record.error_kind == "protocol" for record in recorder.requests)


@pytest.mark.parametrize("cleanup_phase", ["warmup", "response", "client"])
def test_http_cleanup_failure_preserves_completed_samples_and_fails_repetition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_phase: str
) -> None:
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url).model_copy(
            update={"warmup_requests_per_target": int(cleanup_phase == "warmup")}
        )
        repetition = tmp_path / "cases/case/repetition-0001"
        workload = retain_workload(case, repetition)
        if cleanup_phase == "client":
            close_client = httpx.AsyncClient.__aexit__

            async def close_client_with_failure(
                client: httpx.AsyncClient,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                await close_client(client, exc_type, exc_value, traceback)
                if (repetition / "measurement.json").is_file():
                    raise RuntimeError("client cleanup failed")

            monkeypatch.setattr(httpx.AsyncClient, "__aexit__", close_client_with_failure)
        else:
            close_response = httpx.Response.aclose

            async def close(response: httpx.Response) -> None:
                await close_response(response)
                raise RuntimeError("response cleanup failed")

            monkeypatch.setattr(httpx.Response, "aclose", close)
        with pytest.raises(RuntimeError):
            asyncio.run(run_serving(case, repetition))
        records = read_jsonl(
            repetition / "warmup/requests.jsonl" if cleanup_phase == "warmup" else repetition / "requests.jsonl",
            RequestRecord,
        )
        assert records[0].outcome == "success"
        assert records[0].metrics.http_ttft_seconds is not None
        assert records[0].completion_tokens == 3
        checkpoint = RepetitionManifest.model_validate_json((repetition / "repetition.json").read_bytes())
        assert checkpoint.infrastructure_error is not None
        if cleanup_phase == "client":
            assert checkpoint.window_kind == "complete"
            assert len(records) == len(workload.requests)
        assert (
            finalize_repetition(repetition, workload, cleanup_verified=True, worker_code=2, infrastructure_error=None)
            == 2
        )


def test_client_initialization_failure_after_origin_keeps_the_window_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = client_case(tmp_path, "http://127.0.0.1:1")
    repetition = tmp_path / "cases/case/repetition-0001"
    workload = retain_workload(case, repetition)
    enter = httpx.AsyncClient.__aenter__

    async def start(client: httpx.AsyncClient) -> httpx.AsyncClient:
        if (repetition / "measurement.json").is_file():
            raise RuntimeError("HTTP client initialization failed")
        return await enter(client)

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", start)
    with pytest.raises(RuntimeError, match="HTTP client initialization failed"):
        asyncio.run(run_serving(case, repetition))
    checkpoint = RepetitionManifest.model_validate_json((repetition / "repetition.json").read_bytes())
    assert checkpoint.window_end_seconds is None and checkpoint.cleanup_verified is None
    assert checkpoint.infrastructure_error is not None
    origin = (repetition / "measurement.json").read_bytes()
    records = read_jsonl(repetition / "requests.jsonl", RequestRecord)
    assert len(records) == len(workload.requests) == 3
    assert all(record.error_kind == "evidence_missing" and record.ended_at_seconds is None for record in records)
    assert (
        finalize_repetition(repetition, workload, cleanup_verified=True, worker_code=2, infrastructure_error=None) == 2
    )
    finalized = RepetitionManifest.model_validate_json((repetition / "repetition.json").read_bytes())
    assert finalized.finished and finalized.cleanup_verified and finalized.window_end_seconds is None
    assert not finalized.raw_evidence_complete
    assert (repetition / "measurement.json").read_bytes() == origin


@pytest.mark.parametrize("infrastructure_failure", [False, True])
def test_cancellation_or_process_failure_accounts_for_dispatched_queued_and_future_requests(
    infrastructure_failure: bool, tmp_path: Path
) -> None:
    with FakeServingServer(block_first=True) as server:
        case = client_case(tmp_path, server.url, future=True)
        workload = retain_workload(case, tmp_path / "result")
        recorder = MeasurementRecorder(tmp_path / "result")

        def check_alive() -> None:
            if infrastructure_failure and server.first_started.is_set():
                raise RuntimeError("serving process exited")

        async def execute() -> None:
            task = asyncio.create_task(
                run_measurement(case, workload, {TEST_MODEL_ID: server.url}, recorder, check_alive=check_alive)
            )
            assert await asyncio.to_thread(server.first_started.wait, 5)
            if not infrastructure_failure:
                task.cancel()
            await task

        try:
            if infrastructure_failure:
                with pytest.raises(ExceptionGroup, match="unhandled errors"):
                    asyncio.run(execute())
            else:
                with pytest.raises(asyncio.CancelledError):
                    asyncio.run(execute())
        finally:
            recorder.close()
        records = {record.request_id: record for record in recorder.requests}
        assert len(records) == 3
        assert records["first"].outcome == "cancelled"
        assert records["second"].outcome == records["third"].outcome == "not_sent"
        assert records["second"].enqueued_at_seconds is not None
        assert records["third"].enqueued_at_seconds is None
        assert records["third"].http_started_at_seconds is None


def test_cancellation_after_requests_finish_does_not_claim_the_unelapsed_horizon(tmp_path: Path) -> None:
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url)
        case = case.model_copy(update={"arrivals": case.arrivals.model_copy(update={"duration_seconds": 1000.0})})
        workload = retain_workload(case, tmp_path / "result")
        recorder = MeasurementRecorder(tmp_path / "result")

        async def execute() -> None:
            task = asyncio.create_task(run_measurement(case, workload, {TEST_MODEL_ID: server.url}, recorder))
            try:
                async with asyncio.timeout(10):
                    while len(recorder.requests) != len(workload.requests):
                        await asyncio.sleep(0.001)
                await asyncio.sleep(0)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        try:
            asyncio.run(execute())
        finally:
            recorder.close()
        checkpoint = RepetitionManifest.model_validate_json((recorder.directory / "repetition.json").read_bytes())
        assert checkpoint.window_kind == "interrupted"
        assert checkpoint.window_end_seconds is not None and checkpoint.window_end_seconds < 1000.0
        summary = summarize(
            workload,
            recorder.requests,
            recorder.events,
            window_end_seconds=checkpoint.window_end_seconds,
            window_kind="interrupted",
        )
        assert summary.outcomes["success"] == 3 and not summary.execution_complete


def test_program_signal_adapter_keeps_async_cleanup_and_original_cancelled_verdict(tmp_path: Path) -> None:
    program = """
import asyncio
import sys
from pathlib import Path
import httpx
from benches.suites.serving.multi_model import run_serving
from xbench.harness.serving.measure import BenchCaseManifest
from xkit.task import TaskRoot

directory = Path(sys.argv[1])
case = BenchCaseManifest.model_validate_json((directory.parent / "case.json").read_bytes()).case
root = TaskRoot.from_environment()
assert root is not None
scope = root.register_scope()
root.activate()
original_close = httpx.AsyncClient.__aexit__

async def close(client, exc_type, exc_value, traceback):
    measured = (directory / "measurement.json").exists()
    if measured:
        (directory / "retiring").touch()
        while not (directory / "allow-cleanup").exists():
            await asyncio.sleep(0.01)
    await original_close(client, exc_type, exc_value, traceback)
    if measured:
        (directory / "client-cleaned").touch()

httpx.AsyncClient.__aexit__ = close
try:
    asyncio.run(run_serving(case, directory))
finally:
    if (directory / "client-cleaned").exists():
        scope.complete()
    root.finish()
"""
    with FakeServingServer(block_first=True) as server:
        case = client_case(tmp_path, server.url, future=True)
        repetition = tmp_path / "case/repetition-0001"
        retain_workload(case, repetition)
        scope = SupervisedTaskScope.start(
            "benchmark-async-cancel",
            [sys.executable, "-c", program, str(repetition)],
            cwd=Path(__file__).resolve().parents[6],
            env=dict(os.environ),
            log_path=tmp_path / "task.log",
            timeout_seconds=30,
        )
        try:
            observation_deadline = time.monotonic() + 10
            while not server.first_started.is_set() and time.monotonic() < observation_deadline:
                assert scope.poll() is None
                time.sleep(0.01)
            assert server.first_started.is_set(), (tmp_path / "task.log").read_text(encoding="utf-8")
            assert scope.root is not None
            os.kill(scope.root.pid, signal.SIGTERM)
            while not (repetition / "retiring").exists() and time.monotonic() < observation_deadline:
                assert scope.poll() is None
                time.sleep(0.01)
            assert (repetition / "retiring").exists()
            # The real benchmark asyncio adapter now owns signals. Its second
            # delivery must leave the awaited HTTP-client cleanup running.
            os.kill(scope.root.pid, signal.SIGINT)
            os.kill(scope.root.pid, signal.SIGTERM)
            observation_deadline = time.monotonic() + 0.2
            while time.monotonic() < observation_deadline:
                assert scope.poll() is None
                assert scope.root.is_alive()
                time.sleep(0.01)
            (repetition / "allow-cleanup").touch()
            completion = scope.wait()
            assert completion.kind is TaskCompletionKind.EXITED and completion.returncode == 143
            assert not scope.is_protected
            assert (repetition / "client-cleaned").is_file()
            records = read_jsonl(repetition / "requests.jsonl", RequestRecord)
            assert records[0].outcome == "cancelled"
            assert all(record.outcome == "not_sent" for record in records[1:])
            assert server.thread.is_alive()
        finally:
            # Only a CPU HTTP peer and an explicit stand-in cleanup scope exist.
            (repetition / "allow-cleanup").touch()
            server.gate.set()
            if scope.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED, TaskScopeState.CLOSED):
                SupervisedTaskScope.terminate_all((scope,))
            if scope.state in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
                scope.close()


def test_warmup_observes_process_loss_while_a_response_is_pending(tmp_path: Path) -> None:
    with FakeServingServer(block_first=True) as server:
        case = client_case(tmp_path, server.url).model_copy(update={"warmup_requests_per_target": 1})

        def check_alive() -> None:
            if server.first_started.is_set():
                raise RuntimeError("serving process exited")

        with pytest.raises(ExceptionGroup) as error:
            asyncio.run(
                warmup(case, prepare_workload(case), {TEST_MODEL_ID: server.url}, tmp_path, check_alive=check_alive)
            )

        assert any(str(cause) == "serving process exited" for cause in error.value.exceptions)
        records = read_jsonl(tmp_path / "warmup/requests.jsonl", RequestRecord)
        assert len(records) == 1 and records[0].outcome == "cancelled"


def retain_workload(case: BenchCase, repetition: Path) -> PreparedWorkload:
    workload = prepare_workload(case)
    repetition.mkdir(parents=True)
    write_json(
        repetition.parent / "case.json",
        BenchCaseManifest(
            case=case,
            prompt_sha256=workload.prompt_sha256,
            trace_sha256=workload.trace_sha256,
            warmup_sha256=workload.warmup_sha256,
            repetitions=(repetition.name,),
        ).model_dump(mode="json"),
    )
    write_json(
        repetition.parent / "workload.json",
        {
            **workload.model_dump(mode="json", exclude={"case_id", "model_ids", "prompts", "requests", "warmup"}),
            "inputs": {"prompts": "prompts.jsonl", "requests": "trace.jsonl", "warmup": "warmup.jsonl"},
        },
    )
    for name, values in (("prompts", workload.prompts), ("trace", workload.requests), ("warmup", workload.warmup)):
        write_jsonl(repetition.parent / f"{name}.jsonl", (value.model_dump(mode="json") for value in values))
    write_json(repetition / "repetition.json", RepetitionManifest(repetition=1).model_dump(mode="json"))
    return workload
