"""Warm up and measure one benchmark workload, starting and stopping managed servers."""

from __future__ import annotations

import asyncio
import os
import platform
import signal
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
from pydantic import JsonValue

from xbench.harness.serving.api import OpenaiAdapter
from xbench.harness.serving.case import BenchCase, ClientBenchCase, OwnedBenchCase, ServingEndpoint
from xbench.harness.serving.client import (
    MeasurementRecorder,
    RequestState,
    observe_system,
    run_measurement,
    send_request,
)
from xbench.harness.serving.environment import (
    ENVIRONMENT_KEYS,
    package_versions,
    write_environment,
)
from xbench.harness.serving.hosted import HostedServingAdapter
from xbench.harness.serving.measure import BenchCaseManifest, RepetitionManifest, RequestRecord
from xbench.harness.serving.workload import PreparedWorkload
from xkit.results import write_json
from xkit.task import TaskCancelled, get_task_root
from xpool.model import ModelId


async def run_serving(case_manifest: BenchCaseManifest, directory: Path) -> None:
    case = case_manifest.case
    workload = PreparedWorkload.load(directory.parent, case=case)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("benchmark worker requires a running task")
    loop = asyncio.get_running_loop()
    root = get_task_root()
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    recorder = MeasurementRecorder(directory)
    cancelled_signal = None
    infrastructure_error = None
    failure: Exception | None = None

    def cancel(signum: int) -> None:
        nonlocal cancelled_signal
        if cancelled_signal is not None or (root is not None and not root.consume_cancellation()):
            return
        cancelled_signal = signum
        task.cancel()

    try:
        for signum in handlers:
            loop.add_signal_handler(signum, cancel, signum)
        capture_errors: list[str] = []
        environment: dict[str, JsonValue] = {
            "mode": case.mode,
            "tool_software": {
                "source": "local_distribution_metadata",
                "python": platform.python_version(),
                "packages": package_versions(("xpool-dev", "xpool", "httpx"), capture_errors),
            },
            "environment": {name: os.environ[name] for name in ENVIRONMENT_KEYS if name in os.environ},
            "capture_errors": capture_errors,
            "cache_policy": {
                "controller": "external" if isinstance(case, ClientBenchCase) else case.mode,
                "flush_performed": False,
                "serving_state_reset": False,
                "warmup_requests_per_target": case.warmup_requests_per_target,
            },
        }
        write_environment(directory, environment)
        try:
            if workload.requests:
                async with AsyncExitStack() as stack:
                    check_alive = None
                    if isinstance(case, ClientBenchCase):
                        endpoints = {
                            target.model_id: ServingEndpoint(
                                base_url=target.base_url,
                                served_model_name=target.served_model_name or str(target.model_id),
                            )
                            for target in case.targets
                        }
                    else:
                        if isinstance(case, OwnedBenchCase):
                            if case_manifest.deployment is None:
                                raise ValueError("owned workload lacks its resolved configuration")
                            from xbench.harness.serving.owned import OwnedServingAdapter

                            serving = OwnedServingAdapter(case, case_manifest.deployment, capture_errors)
                        else:
                            if case_manifest.hosted_launch is None:
                                raise ValueError("hosted workload lacks its launch declaration")
                            serving = HostedServingAdapter(case, case_manifest.hosted_launch, capture_errors)
                        print("xbench phase=startup", file=sys.stderr)
                        ready = await stack.enter_async_context(
                            serving.start(workdir=directory, startup_timeout_seconds=case.startup_timeout_seconds)
                        )
                        endpoints = ready.endpoints
                        environment["serving"] = ready.model_dump(mode="json")
                        environment["cache_policy"] = {
                            "controller": case.mode,
                            "flush_performed": False,
                            "serving_state_reset": True,
                            "warmup_requests_per_target": case.warmup_requests_per_target,
                        }
                        write_environment(directory, environment)
                        print("xbench phase=request-limits", file=sys.stderr)
                        await serving.validate_workload(workload)
                        check_alive = serving.check_alive
                    print("xbench phase=warmup", file=sys.stderr)
                    await warmup(case, workload, endpoints, directory, check_alive=check_alive)
                    await run_measurement(case, workload, endpoints, recorder, check_alive=check_alive)
                    if check_alive is not None:
                        check_alive()
        except TaskCancelled:
            if root is not None:
                root.consume_cancellation()
            cancelled_signal = cancelled_signal or signal.SIGINT
            infrastructure_error = "benchmark repetition cancelled"
        except asyncio.CancelledError:
            infrastructure_error = "benchmark repetition cancelled"
        except Exception as error:
            failure = error
            infrastructure_error = f"{type(error).__name__}: {error}"
            traceback.print_exception(error)
        finally:
            measured = (directory / "measurement.json").is_file()
            if not recorder.failed:
                for request in workload.requests:
                    if request.request_id not in recorder.request_ids:
                        recorder.request(
                            RequestRecord(
                                **request.model_dump(),
                                outcome="failed" if measured else "not_sent",
                                error_kind="evidence_missing" if measured else "before_measurement",
                                error_message="terminal evidence unavailable"
                                if measured
                                else "measurement did not start",
                            )
                        )
            try:
                recorder.close()
            except OSError as error:
                infrastructure_error = infrastructure_error or f"recording failed: {type(error).__name__}"
            checkpoint = RepetitionManifest.model_validate_json(
                (directory / "repetition.json").read_bytes()
            ).model_copy(update={"infrastructure_error": infrastructure_error})
            write_json(directory / "repetition.json", checkpoint.model_dump(mode="json"))
        if cancelled_signal is not None:
            raise SystemExit(128 + cancelled_signal) from failure
        if infrastructure_error is not None or recorder.failed:
            raise RuntimeError(infrastructure_error or "benchmark recording failed") from failure
    finally:
        for signum, handler in handlers.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, handler)


async def warmup(
    case: BenchCase,
    workload: PreparedWorkload,
    endpoints: Mapping[ModelId, ServingEndpoint],
    directory: Path,
    *,
    check_alive: Callable[[], None] | None,
) -> None:
    """Complete every target's independent warmup before creating measured T0."""

    recorder = MeasurementRecorder(directory / "warmup")
    prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
    targets = {target.model_id: target for target in case.targets}
    origin = time.perf_counter()
    stopped = asyncio.Event()
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=case.mode == "client") as client:

            async def dispatch() -> None:
                for request in workload.warmup:
                    now = time.perf_counter() - origin
                    target = targets[request.model_id]
                    endpoint = endpoints[request.model_id]
                    state = RequestState(request, recorder, enqueued_at=now)
                    state.event("enqueue", now)
                    record = await send_request(
                        client,
                        endpoint.base_url,
                        OpenaiAdapter.payload(request, prompts[request.model_id, request.prompt_id], target, endpoint),
                        state,
                        origin=origin,
                        timeout_seconds=case.request_timeout_seconds,
                    )
                    if record.outcome != "success":
                        raise RuntimeError(f"{request.model_id}: warmup failed ({record.error_kind})")
                stopped.set()

            async with asyncio.TaskGroup() as group:
                group.create_task(dispatch())
                group.create_task(observe_system(check_alive, stopped))
    finally:
        recorder.close()
