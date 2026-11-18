"""Supervised orchestration of compiled Python test-suite tasks."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xkit.device import DevicePool
from xkit.scheduler import ActiveTask, TaskScheduler
from xkit.supervisor import TaskCompletion, TaskCompletionKind
from xtest.harness.report import TaskReportRecord, TestResultWriter
from xtest.harness.runner.plan import TestPlan, TestStage
from xtest.harness.runner.pytest_report import PytestTaskReport
from xtest.harness.runner.task import ExecutionTask, compile_execution_tasks

logger = logging.getLogger("xtest.runner")


class SuiteInfrastructureFailure(RuntimeError):
    """Raised when suite infrastructure cannot safely continue scheduling."""


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """One fully reaped task result and its durable artifact location."""

    task: ExecutionTask
    completion: TaskCompletion
    report: PytestTaskReport | None
    directory: Path

    @property
    def result_code(self) -> Literal[0, 1, 2]:
        """Compose process lifecycle and pytest facts into the accepted code."""

        if self.completion.kind in {
            TaskCompletionKind.TIMED_OUT,
            TaskCompletionKind.LEAKED,
            TaskCompletionKind.INFRASTRUCTURE_FAILED,
        }:
            return 2
        assert self.completion.returncode is not None
        if self.completion.returncode == 0:
            return 0 if self.report is not None and not self.report.failed else 2
        if self.completion.returncode == 1:
            return 1 if self.report is not None and self.report.failed else 2
        return 2


class SuiteRunner:
    """Own test-stage ordering, pytest verdicts and artifacts on shared scheduling."""

    def __init__(
        self,
        plan: TestPlan,
        *,
        repository_root: Path,
        run_directory: Path,
        strict_requirements: bool,
        catalogue_path: Path,
        tool_config_path: Path,
        device_pool: DevicePool | None = None,
        result_writer: TestResultWriter | None = None,
    ) -> None:
        self.plan = plan
        self.result_writer = result_writer
        self.tasks = compile_execution_tasks(plan)
        self.repository_root = repository_root
        self.run_directory = run_directory
        self.strict_requirements = strict_requirements
        self.catalogue_path = catalogue_path
        self.tool_config_path = tool_config_path
        self.scheduler = TaskScheduler[ExecutionTask](device_pool=device_pool, on_complete=self.complete_task)
        self.outcomes: dict[str, TaskOutcome] = {}
        self.stop_signal: int | None = None
        device_tasks = tuple(task for task in self.tasks if task.requirements.device_count)
        if device_tasks and device_pool is None:
            raise ValueError("device execution tasks require a borrowed device pool")
        if device_pool is not None:
            oversized = tuple(
                task.key for task in device_tasks if task.requirements.device_count > len(device_pool.uuids)
            )
            if oversized:
                raise ValueError(
                    f"execution tasks exceed the eligible device pool of {len(device_pool.uuids)}: {oversized}"
                )

    def request_stop(self, signal_number: int, frame: object) -> None:
        """Record only the first terminal stop request for cooperative cleanup."""

        del frame
        if self.stop_signal is None:
            self.stop_signal = signal_number
            self.scheduler.request_cancel()

    def run(self) -> int:
        """Execute every admitted stage and return the canonical suite exit code."""

        try:
            unit_code = self.run_stage(TestStage.UNIT)
            self.report_stage(TestStage.UNIT, unit_code)
            if unit_code:
                return self.result_code(unit_code)
            integration_code = self.run_stage(TestStage.INTEGRATION)
            self.report_stage(TestStage.INTEGRATION, integration_code)
            if integration_code:
                return self.result_code(integration_code)
            e2e_code = self.run_stage(TestStage.E2E)
            self.report_stage(TestStage.E2E, e2e_code)
            if e2e_code == 2 or self.stop_signal is not None:
                return self.result_code(e2e_code)
            models_code = self.run_stage(TestStage.MODELS)
            self.report_stage(TestStage.MODELS, models_code)
            return self.result_code(max(e2e_code, models_code))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"xpool test infrastructure failure: {error}", file=sys.stderr)
            try:
                self.scheduler.cancel_active()
            except (OSError, RuntimeError) as cleanup_error:
                print(f"xpool test cleanup failure: {cleanup_error}", file=sys.stderr)
            if self.result_writer is not None:
                self.result_writer.fail(str(error))
            return self.result_code(2)

    def result_code(self, ordinary_code: int) -> int:
        """Return a signal-derived code only after cooperative cleanup completes."""

        if self.stop_signal is not None:
            return 128 + self.stop_signal
        return ordinary_code

    def report_stage(self, stage: TestStage, result_code: int) -> None:
        """Print one durable summary for an admitted suite stage."""

        tasks = tuple(task for task in self.tasks if task.stage is stage)
        completed = sum(task.key in self.outcomes for task in tasks)
        print(f"STAGE {stage.value}: code={result_code} completed={completed}/{len(tasks)}")
        if self.result_writer is not None:
            self.result_writer.stage(stage.value, result_code)

    def run_stage(self, stage: TestStage) -> int:
        """Run one admitted stage, allowing deterministic device backfill."""

        pending = sorted(
            (task for task in self.tasks if task.stage is stage),
            key=lambda task: (-task.requirements.device_count, -task.estimated_duration_seconds, task.key),
        )
        if not pending:
            return 0
        try:
            for task in pending:
                self.scheduler.submit(task, device_count=task.requirements.device_count)
            self.scheduler.run(self.start_task)
        except BaseException as error:
            try:
                self.scheduler.cancel_active()
            except BaseException as cleanup_error:
                raise SuiteInfrastructureFailure(
                    f"stage {stage.value} failed ({error}) and cleanup failed: {cleanup_error}"
                ) from error
            if isinstance(error, SuiteInfrastructureFailure):
                raise
            raise SuiteInfrastructureFailure(f"stage {stage.value} scheduling failed: {error}") from error
        return max(
            (outcome.result_code for outcome in self.outcomes.values() if outcome.task.stage is stage), default=0
        )

    def start_task(self, task: ExecutionTask) -> None:
        """Acquire resources and atomically start one supervised pytest root."""

        directory = self.run_directory / task.key
        artifact_directory = directory / "artifacts"
        temporary_directory = directory / "pytest-tmp"
        artifact_directory.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-m",
            "xtest.harness.runner.worker",
            *(case.nodeid for case in task.cases),
            f"--xpool-test-catalog={self.catalogue_path}",
            f"--xpool-tool-config={self.tool_config_path}",
            f"--basetemp={temporary_directory}",
            f"--junitxml={directory / 'pytest.xml'}",
        ]
        if task.requirements.device_count:
            command.append("-v")
        if self.strict_requirements:
            command.append("--strict-requirements")
        command.append(f"--xpool-task-artifact-dir={artifact_directory}")
        running = self.scheduler.start(
            task,
            name=task.key,
            device_count=task.requirements.device_count,
            command=command,
            cwd=self.repository_root,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            log_path=directory / "pytest.log",
            timeout_seconds=task.timeout_seconds,
        )
        logger.info("%s devices=%s", task.key, running.device_assignments, extra={"status": "RUNNING"})

    def complete_task(self, running: ActiveTask[ExecutionTask], completion: TaskCompletion) -> None:
        """Interpret pytest/JUnit facts after the shared owner retires the domain."""
        key = running.task.key
        directory = self.run_directory / key
        report: PytestTaskReport | None = None
        if completion.kind is TaskCompletionKind.EXITED and completion.returncode in {0, 1}:
            try:
                report = PytestTaskReport.read(directory / "pytest.xml", running.task.cases)
            except ValueError as error:
                print(f"INVALID {key}: {error}; see {directory}", file=sys.stderr)
        outcome = TaskOutcome(running.task, completion, report, directory)
        self.outcomes[key] = outcome
        if outcome.result_code == 2:
            self.scheduler.stop_admission()
        if self.result_writer is not None:
            self.result_writer.task(
                TaskReportRecord(
                    key=key,
                    stage=running.task.stage.value,
                    completion=completion,
                    result_code=outcome.result_code,
                    elapsed_seconds=time.monotonic() - running.started_at,
                    cases=report.cases if report is not None else (),
                    artifact_directory=str(directory.relative_to(self.run_directory)),
                )
            )
        status = "PASSED" if outcome.result_code == 0 else "FAILED"
        pytest_summary = outcome.report.summary() if outcome.report is not None else "pytest-report=unavailable"
        logger.info(
            "%s devices=%s elapsed=%.3fs (%s, returncode=%s); %s; log=%s junit=%s",
            key,
            running.device_assignments,
            time.monotonic() - running.started_at,
            completion.kind.value,
            completion.returncode,
            pytest_summary,
            directory / "pytest.log",
            directory / "pytest.xml",
            extra={"status": status},
        )
        if outcome.report is not None:
            for case in outcome.report.cases:
                duration = "unavailable" if case.elapsed_seconds is None else f"{case.elapsed_seconds:.3f}s"
                logger.info("  %s elapsed=%s", case.nodeid, duration)

    @property
    def resources_releasable(self) -> bool:
        """Return whether every borrowed task lease was safely returned."""

        return self.scheduler.resources_releasable
