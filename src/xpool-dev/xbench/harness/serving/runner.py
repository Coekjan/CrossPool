"""Sequential benchmark invocations with proven-empty repetition supervision."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter

from xbench.harness.collection import collect_programs
from xbench.harness.serving.api import create_api_adapter
from xbench.harness.serving.case import (
    BenchCase,
    BenchValue,
    ClientBenchCase,
    OwnedBenchCase,
    ResolvedDeployment,
    resolve_deployment,
)
from xbench.harness.serving.measure import (
    BenchCaseManifest,
    BenchRunManifest,
    MeasurementOrigin,
    RepetitionManifest,
    RequestRecord,
    StreamEvent,
    validate_measurement,
)
from xbench.harness.serving.workload import PreparedWorkload, ScheduledRequest, file_digest, prepare_workload
from xkit.config import resolve_model_weights
from xkit.device import DeviceLease, DevicePool
from xkit.results import RunStore, write_json, write_jsonl
from xkit.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeFailure
from xpool.config import ConfigSourceRecord, MissingRequiredConfig, XpoolConfig


class BenchmarkCancelled(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"benchmark cancelled by signal {signum}")
        self.signum = signum


def redact_case(case: BenchCase) -> BenchCase:
    if not isinstance(case, ClientBenchCase):
        return case
    targets = []
    for target in case.targets:
        url = urlsplit(target.base_url)
        # Credentials reach the worker only in its environment, never the manifest.
        safe = urlunsplit(url._replace(netloc=url.netloc.rsplit("@", 1)[-1]))
        targets.append(target.model_copy(update={"base_url": safe}))
    return case.model_copy(update={"targets": tuple(targets)})


def recover_records[V: BenchValue](path: Path, value_type: type[V]) -> tuple[tuple[V, ...], bool]:
    """Preserve damaged raw files and recover only their validated ordered prefix."""

    values = []
    complete = path.is_file()
    if complete:
        with path.open(encoding="utf-8") as source:
            try:
                for line in source:
                    values.append(value_type.model_validate_json(line))
            except (UnicodeError, ValueError):
                complete = False
    if not complete:
        if path.exists():
            path.rename(path.with_name(path.name + ".partial"))
        write_jsonl(path, (value.model_dump(mode="json") for value in values))
    return tuple(values), complete


def finalize_repetition(
    directory: Path,
    workload: PreparedWorkload,
    *,
    cleanup_verified: bool,
    worker_code: int,
    infrastructure_error: str | None,
) -> int:
    """Reconcile raw evidence after domain drain and seal the measurement checkpoint."""

    requests, requests_valid = recover_records(directory / "requests.jsonl", RequestRecord)
    events, events_valid = recover_records(directory / "events.jsonl", StreamEvent)
    if not requests_valid or not events_valid:
        infrastructure_error = infrastructure_error or "recording evidence missing or damaged"
    recorded = {request.request_id for request in requests}
    missing = tuple(request for request in workload.requests if request.request_id not in recorded)
    if missing:
        infrastructure_error = infrastructure_error or "worker lost before terminal request accounting"
        with (directory / "requests.jsonl").open("a", encoding="utf-8") as output:
            recovered = []
            for request in missing:
                record = RequestRecord(
                    **request.model_dump(),
                    outcome="failed",
                    error_kind="evidence_missing",
                    error_message="owner lost; dispatch and termination times are unknown",
                )
                output.write(record.model_dump_json() + "\n")
                recovered.append(record)
            output.flush()
            os.fsync(output.fileno())
        requests = (*requests, *recovered)
    if not events_valid:
        by_request: dict[str, list[StreamEvent]] = {}
        for event in events:
            by_request.setdefault(event.request_id, []).append(event)
        reconciled = []
        for request in requests:
            try:
                request.validate_evidence(by_request.get(request.request_id, []))
            except ValueError:
                reconciled.append(
                    RequestRecord(
                        **request.model_dump(include=set(ScheduledRequest.model_fields)),
                        outcome="failed",
                        error_kind="evidence_missing",
                        error_message="recording lost; retained events do not prove request termination",
                    )
                )
            else:
                reconciled.append(request)
        if tuple(reconciled) != requests:
            raw_requests = directory / "requests.jsonl"
            raw_requests.rename(directory / "requests.jsonl.unverified")
            requests = tuple(reconciled)
            write_jsonl(raw_requests, (request.model_dump(mode="json") for request in requests))
    manifest = RepetitionManifest.model_validate_json((directory / "repetition.json").read_bytes())
    window = manifest.window_end_seconds
    window_kind = manifest.window_kind
    infrastructure_error = infrastructure_error or manifest.infrastructure_error
    if worker_code not in (0, 130, 143):
        infrastructure_error = infrastructure_error or f"benchmark worker exited with code {worker_code}"
    origin = directory / "measurement.json"
    measured = origin.is_file()
    if measured or window is not None:
        MeasurementOrigin.model_validate_json(origin.read_bytes())
        if window is None:
            last = max(
                (
                    *(event.observed_at_seconds for event in events),
                    *(request.ended_at_seconds for request in requests if request.ended_at_seconds is not None),
                ),
                default=None,
            )
            if last is not None:
                window = last
                window_kind = "observed_prefix"
    validate_measurement(workload, requests, events, window_end_seconds=window, window_kind=window_kind)
    if workload.requests and window_kind != "complete":
        infrastructure_error = infrastructure_error or "benchmark measurement did not complete"
    code = (
        worker_code
        if worker_code in (130, 143)
        else 2
        if infrastructure_error is not None or not cleanup_verified or worker_code != 0
        else 1
        if not requests or any(request.outcome != "success" for request in requests)
        else 0
    )
    manifest = manifest.model_copy(
        update={
            "finished": True,
            "result_code": code,
            "cleanup_verified": cleanup_verified,
            "raw_evidence_complete": requests_valid
            and events_valid
            and not any(request.error_kind == "evidence_missing" for request in requests),
            "window_end_seconds": window,
            "window_kind": window_kind,
            "infrastructure_error": infrastructure_error,
            "artifact_sha256": {
                name: file_digest(directory / name)
                for name in ("requests.jsonl", "events.jsonl", "measurement.json")
                if name != "measurement.json" or measured
            },
        }
    )
    write_json(directory / "repetition.json", manifest.model_dump(mode="json"))
    return code


def run_benchmarks(cases: tuple[BenchCase, ...], *, root: Path, catalogue_path: Path) -> int:
    programs = collect_programs(catalogue_path, cases)
    for case in cases:
        for target in case.targets:
            create_api_adapter(target.api)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    run = RunStore(root.expanduser()).start(run_id)
    manifest = BenchRunManifest(run_id=run_id, selected_cases=tuple(case.id for case in cases))
    pool = None
    result = 0
    infrastructure_error = None
    safe = True
    cancelled = False
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def cancel(signum: int, frame: FrameType | None) -> None:
        nonlocal cancelled
        if cancelled:
            return
        cancelled = True
        raise BenchmarkCancelled(signum)

    for signum in handlers:
        signal.signal(signum, cancel)
    try:
        write_json(run.directory / "run.json", manifest.model_dump(mode="json"))
        print(run.directory, flush=True)
        for case, program in zip(cases, programs, strict=True):
            config = resolve_deployment(case) if isinstance(case, OwnedBenchCase) else None
            serving_metadata = case.load_serving_metadata() if isinstance(case, ClientBenchCase) else None
            workload = prepare_workload(case, config=config)
            case_directory = run.directory / "cases" / case.id
            case_directory.mkdir(parents=True)
            write_json(
                case_directory / "workload.json",
                {
                    **workload.model_dump(
                        mode="json", exclude={"case_id", "model_ids", "prompts", "requests", "warmup"}
                    ),
                    "inputs": {"prompts": "prompts.jsonl", "requests": "trace.jsonl", "warmup": "warmup.jsonl"},
                },
            )
            write_jsonl(
                case_directory / "prompts.jsonl", (prompt.model_dump(mode="json") for prompt in workload.prompts)
            )
            write_jsonl(
                case_directory / "trace.jsonl", (request.model_dump(mode="json") for request in workload.requests)
            )
            write_jsonl(
                case_directory / "warmup.jsonl", (request.model_dump(mode="json") for request in workload.warmup)
            )
            case_manifest = BenchCaseManifest(
                case=redact_case(case),
                prompt_sha256=workload.prompt_sha256,
                trace_sha256=workload.trace_sha256,
                warmup_sha256=workload.warmup_sha256,
                deployment=ResolvedDeployment(
                    runtime_config=case.runtime_config,
                    cwd=Path.cwd().resolve(),
                    effective=config.model_dump(mode="json"),
                    sources=TypeAdapter(tuple[ConfigSourceRecord, ...]).dump_python(config.sources, mode="json"),
                )
                if isinstance(case, OwnedBenchCase) and config is not None
                else None,
                serving_metadata=serving_metadata,
                repetitions=(),
            )
            write_json(case_directory / "case.json", case_manifest.model_dump(mode="json"))
            manifest = manifest.model_copy(
                update={
                    "case_directories": (*manifest.case_directories, str(case_directory.relative_to(run.directory)))
                }
            )
            write_json(run.directory / "run.json", manifest.model_dump(mode="json"))
            for number in range(1, case.repetitions + 1):
                directory = case_directory / f"repetition-{number:04d}"
                (directory / "logs").mkdir(parents=True)
                (directory / "launch").mkdir()
                write_json(
                    directory / "repetition.json",
                    RepetitionManifest(repetition=number).model_dump(mode="json"),
                )
                case_manifest = case_manifest.model_copy(
                    update={"repetitions": (*case_manifest.repetitions, directory.name)}
                )
                write_json(case_directory / "case.json", case_manifest.model_dump(mode="json"))
                env = dict(os.environ)
                lease: DeviceLease | None = None
                worker_code = 2
                repetition_error = None
                try:
                    if workload.requests and program.requirements.requires_config:
                        if config is None:
                            configured_path = os.environ.get("XPOOL_CONFIG")
                            if not configured_path:
                                raise MissingRequiredConfig("set XPOOL_CONFIG to an xpool TOML file")
                            config = XpoolConfig.from_file(Path(configured_path).expanduser())
                        for model_id in program.requirements.model_ids:
                            resolve_model_weights(config, model_id)
                    if workload.requests and program.requirements.device_count:
                        if pool is None:
                            pool = DevicePool.from_environment()
                        lease = pool.try_lease(program.requirements.device_count)
                        if lease is None:
                            raise RuntimeError(f"case requires {program.requirements.device_count} visible devices")
                        env["CUDA_VISIBLE_DEVICES"] = ",".join(lease.uuids)
                    if isinstance(case, ClientBenchCase):
                        env["XBENCH_ENDPOINTS"] = json.dumps(
                            {str(target.model_id): target.base_url for target in case.targets}
                        )
                    completion = SupervisedTaskScope.run(
                        f"xbench:{case.id}:{number}",
                        [
                            sys.executable,
                            "-m",
                            "xbench.harness.serving.worker",
                            str(directory),
                            f"--module={program.module}",
                            f"--source-path={program.source_path}",
                            f"--entrypoint={program.entrypoint}",
                            *(f"--import-root={path}" for path in program.import_roots),
                        ],
                        cwd=Path.cwd(),
                        env=env,
                        log_path=directory / "logs/worker.log",
                        timeout_seconds=None,
                    )
                    worker_code = completion.returncode if completion.returncode is not None else 2
                    if completion.kind != TaskCompletionKind.EXITED:
                        repetition_error = completion.diagnostics or f"benchmark worker {completion.kind}"
                except BenchmarkCancelled as error:
                    worker_code = 128 + error.signum
                    repetition_error = str(error)
                except TaskScopeFailure:
                    safe = False
                    raise
                except (OSError, RuntimeError, ValueError) as error:
                    repetition_error = f"benchmark repetition infrastructure failed: {error}"
                finally:
                    # A scope failure retains the lease: absence was not proven.
                    if lease is not None and safe:
                        if pool is None:
                            raise RuntimeError("device lease lost its pool")
                        pool.release(lease)
                code = finalize_repetition(
                    directory,
                    workload,
                    cleanup_verified=safe,
                    worker_code=worker_code,
                    infrastructure_error=repetition_error,
                )
                result = max(result, code)
                if code not in (0, 1):
                    break
            if result not in (0, 1):
                break
    except BenchmarkCancelled as error:
        result = 128 + error.signum
        infrastructure_error = str(error)
    except (OSError, RuntimeError, ValueError) as error:
        result = 2
        infrastructure_error = str(error)
        print(f"xbench infrastructure failure: {error}", file=sys.stderr)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if pool is not None and safe:
            pool.close()
        manifest = manifest.model_copy(
            update={"finished": safe, "result_code": result, "infrastructure_error": infrastructure_error}
        )
        try:
            write_json(run.directory / "run.json", manifest.model_dump(mode="json"))
        finally:
            if safe:
                run.complete()
            else:
                run.lock_file.close()
    return result
