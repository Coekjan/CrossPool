"""Concurrent case admission with serial repetitions and retained replay authority."""

from __future__ import annotations

import os
import shutil
import signal
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from types import FrameType

from pydantic import TypeAdapter

from xbench.harness.serving.case import (
    BenchCase,
    BenchValue,
    HostedBenchCase,
    HostedLaunchDeclaration,
    OwnedBenchCase,
    ResolvedDeployment,
    resolve_deployment,
)
from xbench.harness.serving.hosted import collect_hosted_launches
from xbench.harness.serving.measure import (
    REPETITION_DIRECTORY_PATTERN,
    BenchCaseManifest,
    BenchRunManifest,
    MeasurementOrigin,
    RepetitionManifest,
    RequestRecord,
    StreamEvent,
    load_measurement,
    retained_file,
    validate_measurement,
)
from xbench.harness.serving.workload import PreparedWorkload, ScheduledRequest, content_digest, prepare_workload
from xkit.case import CaseId
from xkit.config import XpoolDevConfig, get_global_config, init_global_config, resolve_model_weights
from xkit.device import DevicePool
from xkit.results import RunEntry, RunStore, file_sha256, iter_jsonl, write_json, write_jsonl
from xkit.scheduler import ActiveTask, TaskScheduler
from xkit.supervisor import TaskCompletion, TaskCompletionKind, TaskScopeFailure, TaskStartFailure
from xpool.config import ConfigSourceRecord, XpoolConfig


def recover_records[V: BenchValue](path: Path, value_type: type[V]) -> tuple[tuple[V, ...], bool]:
    """Preserve damaged raw files and recover only their validated ordered prefix."""

    values = []
    complete = path.is_file()
    if complete:
        try:
            values.extend(iter_jsonl(path, value_type))
        except ValueError:
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
        if window is None or window_kind == "interrupted":
            last = max(
                chain(
                    (event.observed_at_seconds for event in events),
                    (request.ended_at_seconds for request in requests if request.ended_at_seconds is not None),
                ),
                default=None,
            )
            if last is not None:
                if window is None:
                    window, window_kind = last, "observed_prefix"
                else:
                    window = max(window, last)
    evidence = validate_measurement(workload, requests, events, window_end_seconds=window, window_kind=window_kind)
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
            "raw_evidence_complete": requests_valid and events_valid and evidence.evidence_complete,
            "window_end_seconds": window,
            "window_kind": window_kind,
            "infrastructure_error": infrastructure_error,
            "artifact_sha256": {
                name: file_sha256(directory / name)
                for name in ("requests.jsonl", "events.jsonl", "measurement.json")
                if name != "measurement.json" or measured
            },
        }
    )
    write_json(directory / "repetition.json", manifest.model_dump(mode="json"))
    return code


@dataclass(frozen=True, slots=True)
class BenchmarkRepetition:
    """One scheduled repetition referencing its case's prepared replay authority."""

    case: BenchCase
    workload: PreparedWorkload
    case_directory: Path
    device_count: int
    config: XpoolConfig | None
    number: int = 1
    attempt: int = 1
    following: tuple[int, ...] = ()

    @property
    def directory(self) -> Path:
        return self.case_directory / f"repetition-{self.number:04d}.attempt-{self.attempt:04d}"


class BenchmarkRunner:
    """Own case preparation, repetition progression, manifests and measurement verdicts."""

    def __init__(
        self,
        entry: RunEntry,
        manifest: BenchRunManifest,
        *,
        fast_fail: bool = False,
    ) -> None:
        self.run_entry = entry
        self.manifest = manifest
        self.scheduler = TaskScheduler[BenchmarkRepetition](device_pool=None, on_complete=self.complete_repetition)
        self.result = 0
        self.fast_fail = fast_fail
        self.infrastructure_error: str | None = None
        self.cancelled_signal: int | None = None
        self.preflighted: set[CaseId] = set()

    def request_cancel(self, signum: int, frame: FrameType | None) -> None:
        if self.cancelled_signal is None:
            self.cancelled_signal = signum
            self.scheduler.request_cancel()

    def prepare_case(self, case: BenchCase, hosted_launch: HostedLaunchDeclaration | None) -> BenchmarkRepetition:
        config = resolve_deployment(case) if isinstance(case, OwnedBenchCase) else None
        workload = prepare_workload(case, config=config)
        device_count = 0
        if config is not None:
            device_count = len(config.atn.devices) + len(config.ffn.devices)
        elif hosted_launch is not None:
            device_count = hosted_launch.requirements.device_count
        directory = self.run_entry.directory / "cases" / str(case.id)
        directory.mkdir(parents=True)
        write_json(
            directory / "workload.json",
            {
                **workload.model_dump(mode="json", exclude={"case_id", "model_ids", "prompts", "requests", "warmup"}),
                "inputs": {"prompts": "prompts.jsonl", "requests": "trace.jsonl", "warmup": "warmup.jsonl"},
            },
        )
        write_jsonl(directory / "prompts.jsonl", (prompt.model_dump(mode="json") for prompt in workload.prompts))
        write_jsonl(directory / "trace.jsonl", (request.model_dump(mode="json") for request in workload.requests))
        write_jsonl(directory / "warmup.jsonl", (request.model_dump(mode="json") for request in workload.warmup))
        manifest = BenchCaseManifest(
            case=case,
            prompt_sha256=content_digest(workload.prompts),
            trace_sha256=content_digest(workload.requests),
            warmup_sha256=content_digest(workload.warmup),
            deployment=ResolvedDeployment(
                runtime_config=case.runtime_config,
                cwd=Path.cwd().resolve(),
                effective=config.model_dump(mode="json"),
                sources=TypeAdapter(tuple[ConfigSourceRecord, ...]).dump_python(config.sources, mode="json"),
            )
            if isinstance(case, OwnedBenchCase) and config is not None
            else None,
            hosted_launch=hosted_launch,
            repetitions=(),
        )
        write_json(directory / "case.json", manifest.model_dump(mode="json"))
        self.manifest = self.manifest.model_copy(
            update={
                "case_directories": (
                    *self.manifest.case_directories,
                    str(directory.relative_to(self.run_entry.directory)),
                )
            }
        )
        write_json(self.run_entry.directory / "run.json", self.manifest.model_dump(mode="json"))
        return BenchmarkRepetition(case, workload, directory, device_count if workload.requests else 0, config)

    def queue_repetition(self, task: BenchmarkRepetition, numbers: tuple[int, ...]) -> None:
        """Queue the next needed logical repetition at a fresh physical attempt path."""
        if not numbers:
            return
        number = numbers[0]
        attempts = []
        for path in task.case_directory.iterdir():
            match = REPETITION_DIRECTORY_PATTERN.fullmatch(path.name)
            if match is not None and match[2] is not None and int(match[1]) == number:
                attempts.append(int(match[2]))
        task = replace(task, number=number, attempt=max(attempts, default=0) + 1, following=numbers[1:])
        if task.device_count and self.scheduler.device_pool is None:
            self.scheduler.device_pool = DevicePool.from_environment()
        self.scheduler.submit(task, device_count=task.device_count)

    def start_repetition(self, task: BenchmarkRepetition) -> None:
        directory = task.directory
        (directory / "logs").mkdir(parents=True)
        (directory / "launch").mkdir()
        write_json(directory / "repetition.json", RepetitionManifest(repetition=task.number).model_dump(mode="json"))
        case_path = task.case_directory / "case.json"
        manifest = BenchCaseManifest.model_validate_json(case_path.read_bytes())
        attempts = manifest.effective_attempts(task.case_directory)
        attempts[task.number] = directory
        write_json(
            case_path,
            manifest.model_copy(
                update={"repetitions": tuple(attempts[number].name for number in sorted(attempts))}
            ).model_dump(mode="json"),
        )
        link = task.case_directory / f"repetition-{task.number:04d}"
        if link.is_symlink() or not link.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(directory.name, target_is_directory=True)
        env = dict(os.environ)
        startup_error: str | None = None
        config = task.config
        if config is not None and task.workload.requests and task.case.id not in self.preflighted:
            try:
                for model in config.models:
                    resolve_model_weights(config, model.id)
                self.preflighted.add(task.case.id)
            except (OSError, RuntimeError, ValueError) as error:
                startup_error = str(error)
        if startup_error is None:
            try:
                self.scheduler.start(
                    task,
                    name=f"xbench:{task.case.id}:{task.number}",
                    device_count=task.device_count,
                    command=[sys.executable, "-m", "xbench.harness.serving.worker", str(directory)],
                    cwd=Path.cwd(),
                    env=env,
                    log_path=directory / "logs/worker.log",
                    timeout_seconds=None,
                )
            except TaskStartFailure as error:
                startup_error = f"benchmark repetition startup failed: {error}"
        if startup_error is not None:
            code = finalize_repetition(
                directory,
                task.workload,
                cleanup_verified=True,
                worker_code=2,
                infrastructure_error=startup_error,
            )
            self.result = max(self.result, code)
            self.infrastructure_error = self.infrastructure_error or startup_error
            print(
                f"xbench case={task.case.id} repetition={task.number} code={code} error={startup_error}",
                file=sys.stderr,
            )
            if self.fast_fail:
                self.scheduler.stop_admission()
            elif task.following:
                self.queue_repetition(task, task.following)

    def complete_repetition(self, running: ActiveTask[BenchmarkRepetition], completion: TaskCompletion) -> None:
        task = running.task
        worker_code = completion.returncode
        error = None
        if completion.kind is not TaskCompletionKind.EXITED:
            error = completion.diagnostics or f"benchmark worker {completion.kind}"
        if worker_code is None:
            worker_code = 128 + self.cancelled_signal if self.cancelled_signal is not None else 2
        code = finalize_repetition(
            task.directory,
            task.workload,
            cleanup_verified=True,
            worker_code=worker_code,
            infrastructure_error=error,
        )
        self.result = max(self.result, code)
        print(
            f"xbench case={task.case.id} repetition={task.number} code={code} "
            f"devices={running.device_assignments} elapsed={time.monotonic() - running.started_at:.3f}s",
            file=sys.stderr,
        )
        if code not in (0, 1):
            checkpoint = RepetitionManifest.model_validate_json((task.directory / "repetition.json").read_bytes())
            self.infrastructure_error = self.infrastructure_error or checkpoint.infrastructure_error
        if self.fast_fail and code != 0:
            self.scheduler.stop_admission()
        elif task.following:
            self.queue_repetition(task, task.following)

    def run(
        self,
        pending: Iterator[tuple[BenchmarkRepetition, tuple[int, ...]]],
        *,
        retained_result: int = 0,
        retained_error: str | None = None,
    ) -> int:
        self.result = retained_result
        self.infrastructure_error = retained_error
        handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        for signum in handlers:
            signal.signal(signum, self.request_cancel)
        try:
            write_json(self.run_entry.directory / "run.json", self.manifest.model_dump(mode="json"))
            print(self.run_entry.directory, flush=True)
            try:
                while self.scheduler.admission_open:
                    work = next(pending, None)
                    if work is None:
                        break
                    self.queue_repetition(*work)
            except TaskScopeFailure:
                raise
            except (OSError, RuntimeError, ValueError) as error:
                self.result = 2
                self.infrastructure_error = self.infrastructure_error or str(error)
                self.scheduler.stop_admission()
                print(f"xbench preparation failure: {error}", file=sys.stderr)
            self.scheduler.run(self.start_repetition)
        except (OSError, RuntimeError, ValueError) as error:
            self.result = 2
            self.infrastructure_error = self.infrastructure_error or str(error)
            print(f"xbench infrastructure failure: {error}", file=sys.stderr)
            try:
                self.scheduler.cancel_active()
            except (OSError, RuntimeError) as cleanup_error:
                print(f"xbench cleanup failure: {cleanup_error}", file=sys.stderr)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            if self.cancelled_signal is not None:
                self.result = 128 + self.cancelled_signal
            safe = self.scheduler.resources_releasable
            if self.scheduler.device_pool is not None and safe:
                self.scheduler.device_pool.close()
            self.manifest = self.manifest.model_copy(
                update={"finished": safe, "result_code": self.result, "infrastructure_error": self.infrastructure_error}
            )
            try:
                write_json(self.run_entry.directory / "run.json", self.manifest.model_dump(mode="json"))
            finally:
                if safe:
                    self.run_entry.complete()
                else:
                    self.run_entry.lock_file.close()
        return self.result


def continue_benchmarks(directory: Path, *, rerun_failure: bool = False, fast_fail: bool = False) -> int:
    """Run unattempted repetitions; optionally retry unsuccessful attempts once.

    Retained successes are reused and skipped failures keep the run unsuccessful.
    Fast-fail applies only to new execution outcomes, not retained failures.
    Prepared replay is restored under one exclusive retained-run lease.
    """
    directory = directory.expanduser().absolute()
    with RunStore(directory.parent).resume(directory.name) as entry:
        manifest = BenchRunManifest.model_validate_json(retained_file(entry.directory, "run.json").read_bytes())
        if (
            manifest.run_id != entry.directory.name
            or not manifest.selected_cases
            or len(set(manifest.selected_cases)) != len(manifest.selected_cases)
        ):
            raise ValueError("benchmark retained run identity or selection is invalid")
        config = init_global_config(resolved=XpoolDevConfig.from_record(manifest.tool_config))
        pending: list[tuple[BenchmarkRepetition, tuple[int, ...]]] = []
        pending_hosted: list[tuple[HostedBenchCase, HostedLaunchDeclaration]] = []
        repetitions = range(1, config.xbench.repetitions + 1)
        reused = 0
        retained_unsuccessful = 0
        result = 0
        infrastructure_error: str | None = None
        for case_id in manifest.selected_cases:
            case_directory = entry.directory / "cases" / str(case_id)
            if case_directory.resolve().parent != entry.directory / "cases":
                raise ValueError("benchmark case directory escapes its invocation")
            case_manifest = BenchCaseManifest.model_validate_json(
                retained_file(case_directory, "case.json").read_bytes()
            )
            case = case_manifest.case
            if case.id != case_id:
                raise ValueError("retained case identity disagrees with its directory")
            device_count = 0
            case_config = None
            if isinstance(case, OwnedBenchCase):
                if case_manifest.deployment is None:
                    raise ValueError(f"{case_id}: owned case lacks its original deployment")
                case_config = XpoolConfig.model_validate(case_manifest.deployment.effective)
                device_count = len(case_config.atn.devices) + len(case_config.ffn.devices)
            workload = case_manifest.load_workload(case_directory)
            attempts = case_manifest.effective_attempts(case_directory)
            if any(number > config.xbench.repetitions for number in attempts):
                raise ValueError("retained repetition exceeds the original configured count")
            skipped: set[int] = set()
            for number, attempt in attempts.items():
                try:
                    checkpoint = RepetitionManifest.from_directory(attempt)
                    measurement = load_measurement(attempt, workload, checkpoint=checkpoint)
                except (OSError, ValueError) as error:
                    code = 2
                    retained_error = f"retained {case_id} repetition {number}: {error}"
                else:
                    if checkpoint.result_code == 0 and measurement.evidence_complete:
                        reused += 1
                        skipped.add(number)
                        continue
                    code = checkpoint.result_code or 2
                    if not measurement.evidence_complete:
                        code = max(code, 2)
                    retained_error = checkpoint.infrastructure_error
                if not rerun_failure:
                    skipped.add(number)
                    retained_unsuccessful += 1
                    result = max(result, code)
                    infrastructure_error = infrastructure_error or retained_error
            numbers = tuple(number for number in repetitions if number not in skipped)
            if isinstance(case, HostedBenchCase):
                launch = case_manifest.hosted_launch
                if launch is None:
                    raise ValueError(f"{case_id}: hosted case lacks its original launch declaration")
                device_count = launch.requirements.device_count
                if numbers:
                    pending_hosted.append((case, launch))
            if numbers:
                pending.append(
                    (
                        BenchmarkRepetition(
                            case, workload, case_directory, device_count if workload.requests else 0, case_config
                        ),
                        numbers,
                    )
                )
        if (
            reused == len(manifest.selected_cases) * config.xbench.repetitions
            and manifest.finished
            and manifest.result_code == 0
            and (entry.directory / ".completed").is_file()
        ):
            print(entry.directory, flush=True)
            print(f"benchmark already complete; uv run xbench report {manifest.run_id}", file=sys.stderr)
            return 0
        hosted_launches = collect_hosted_launches(case for case, _ in pending_hosted)
        for case, saved_launch in pending_hosted:
            if hosted_launches[case.id] != saved_launch:
                raise ValueError(f"{case.id}: hosted script content or requirements changed")
        reopened_manifest = manifest.model_copy(
            update={
                "finished": False,
                "result_code": None,
                "infrastructure_error": None,
                "case_directories": tuple(f"cases/{case_id}" for case_id in manifest.selected_cases),
            }
        )
        print(
            f"xbench continuation reused={reused} retained_unsuccessful={retained_unsuccessful} "
            f"pending={sum(len(numbers) for _, numbers in pending)}",
            file=sys.stderr,
        )
        index = 1
        while (entry.directory / f"run.previous-{index:04d}.json").exists():
            index += 1
        shutil.copyfile(entry.directory / "run.json", entry.directory / f"run.previous-{index:04d}.json")
        entry.reopen()
        runner = BenchmarkRunner(entry, reopened_manifest, fast_fail=fast_fail)
        result = runner.run(iter(pending), retained_result=result, retained_error=infrastructure_error)
        print(f"uv run xbench report {manifest.run_id}", file=sys.stderr)
        return result


def run_benchmarks(cases: tuple[BenchCase, ...], *, root: Path, fast_fail: bool = False) -> int:
    hosted_launches = collect_hosted_launches(case for case in cases if isinstance(case, HostedBenchCase))
    tool_config = get_global_config().record()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    entry = RunStore(root.expanduser()).start(run_id)
    manifest = BenchRunManifest(
        run_id=entry.directory.name, tool_config=tool_config, selected_cases=tuple(case.id for case in cases)
    )
    runner = BenchmarkRunner(entry, manifest, fast_fail=fast_fail)
    numbers = tuple(range(1, tool_config.settings.xbench.repetitions + 1))
    pending = (
        (runner.prepare_case(case, hosted_launches[case.id] if isinstance(case, HostedBenchCase) else None), numbers)
        for case in cases
    )
    return runner.run(pending)
