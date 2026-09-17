"""Supervised orchestration of compiled Python test-suite tasks."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.harness.runner.artifact import ArtifactGroupAdapter, ArtifactGroupRef, ArtifactGroupResult
from tests.harness.runner.gpu import GpuLease, GpuPool
from tests.harness.runner.plan import CollectedTestCase, TestPlan, TestStage
from tests.harness.runner.pytest_report import PytestTaskReport
from tests.harness.runner.supervisor import (
    SupervisedTaskScope,
    TaskCompletion,
    TaskCompletionKind,
    TaskScopeFailure,
    TaskScopeState,
    TaskStartFailure,
    TaskSupervisionFailure,
    drain_unprotected_subreaper_descendants,
)
from tests.harness.runner.task import ExecutionTask, compile_execution_tasks
from xpool.mps import probe_mps_controller

SCHEDULER_POLL_INTERVAL_SECONDS = 0.05


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
    def result_code(self) -> int:
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

    @property
    def succeeded(self) -> bool:
        """Return whether lifecycle, JUnit, and pytest exit facts all succeed."""

        return self.result_code == 0


@dataclass(slots=True)
class RunningTask:
    """One live supervised task and its optional GPU lease."""

    task: ExecutionTask
    directory: Path
    scope: SupervisedTaskScope
    lease: GpuLease | None


class SuiteRunner:
    """Own staged scheduling, supervised scopes, GPU leases, and suite artifacts."""

    def __init__(
        self,
        plan: TestPlan,
        *,
        repository_root: Path,
        run_directory: Path,
        strict_requirements: bool,
        artifact_group_adapters: Sequence[ArtifactGroupAdapter] = (),
        gpu_pool: GpuPool | None = None,
    ) -> None:
        self.plan = plan
        self.tasks = compile_execution_tasks(plan)
        self.repository_root = repository_root
        self.run_directory = run_directory
        self.strict_requirements = strict_requirements
        self.artifact_group_adapters = {adapter.kind: adapter for adapter in artifact_group_adapters}
        if len(self.artifact_group_adapters) != len(artifact_group_adapters):
            raise ValueError("artifact group adapter kinds must be unique")
        declared_kinds = {case.artifact_group.kind for case in plan.cases if case.artifact_group is not None}
        missing_kinds = declared_kinds - self.artifact_group_adapters.keys()
        if missing_kinds:
            raise ValueError(f"artifact groups have no injected adapter: {sorted(missing_kinds)}")
        self.gpu_pool = gpu_pool
        self.active: dict[str, RunningTask] = {}
        self.outcomes: dict[str, TaskOutcome] = {}
        self.retained_leases: list[GpuLease] = []
        self.stop_signal: int | None = None
        gpu_tasks = tuple(task for task in self.tasks if task.requirements.cuda_count)
        if gpu_tasks and self.gpu_pool is None:
            raise ValueError("GPU execution tasks require a borrowed GPU pool")
        if self.gpu_pool is not None:
            oversized = tuple(task.key for task in gpu_tasks if task.requirements.cuda_count > len(self.gpu_pool.uuids))
            if oversized:
                raise ValueError(
                    f"execution tasks exceed the eligible GPU pool of {len(self.gpu_pool.uuids)}: {oversized}"
                )

    def request_stop(self, signal_number: int, frame: object) -> None:
        """Record only the first terminal stop request for cooperative cleanup."""

        del frame
        if self.stop_signal is None:
            self.stop_signal = signal_number

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
            artifact_results = self.artifact_group_results()
            self.report_artifact_groups(artifact_results)
            artifact_code = max((result.result_code for result in artifact_results), default=0)
            return self.result_code(max(e2e_code, artifact_code))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"xpool test infrastructure failure: {error}", file=sys.stderr)
            try:
                self.cancel_active_tasks()
            except (OSError, RuntimeError) as cleanup_error:
                print(f"xpool test cleanup failure: {cleanup_error}", file=sys.stderr)
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

    def run_stage(self, stage: TestStage) -> int:
        """Run one admitted stage, allowing deterministic GPU backfill."""

        pending = sorted(
            (task for task in self.tasks if task.stage is stage),
            key=lambda task: (-task.requirements.cuda_count, -task.estimated_duration_seconds, task.key),
        )
        if not pending:
            return 0
        stage_code = 0
        try:
            while pending or self.active:
                if self.stop_signal is not None:
                    self.cancel_active_tasks()
                    return 0
                launched = self.launch_fitting_tasks(pending)
                completed = self.collect_completed_tasks()
                stage_code = max((stage_code, *(outcome.result_code for outcome in completed)))
                if stage_code == 2:
                    self.cancel_active_tasks()
                    return 2
                if not launched and not completed:
                    if not self.active:
                        raise SuiteInfrastructureFailure(
                            f"stage {stage.value} has pending tasks that cannot fit the idle GPU pool"
                        )
                    time.sleep(SCHEDULER_POLL_INTERVAL_SECONDS)
        except BaseException as error:
            try:
                self.cancel_active_tasks()
            except BaseException as cleanup_error:
                raise SuiteInfrastructureFailure(
                    f"stage {stage.value} failed ({error}) and cleanup failed: {cleanup_error}"
                ) from error
            if isinstance(error, SuiteInfrastructureFailure):
                raise
            raise SuiteInfrastructureFailure(f"stage {stage.value} scheduling failed: {error}") from error
        return stage_code

    def launch_fitting_tasks(self, pending: list[ExecutionTask]) -> bool:
        """Launch fixed-priority tasks while their complete requirements fit."""

        launched = False
        while True:
            selected_index = next(
                (
                    index
                    for index, task in enumerate(pending)
                    if task.requirements.cuda_count == 0
                    or (self.gpu_pool is not None and task.requirements.cuda_count <= self.gpu_pool.available_count)
                ),
                None,
            )
            if selected_index is None:
                return launched
            task = pending.pop(selected_index)
            self.start_task(task)
            launched = True

    def start_task(self, task: ExecutionTask) -> None:
        """Acquire resources and atomically start one supervised pytest root."""

        if task.requirements.cuda_count:
            self.require_mps_controller(f"before task {task.key}")
            if self.gpu_pool is None:
                raise SuiteInfrastructureFailure(f"GPU task {task.key} has no GPU pool")
        directory = self.run_directory / task.key
        artifact_directory = directory / "artifacts"
        temporary_directory = directory / "pytest-tmp"
        artifact_directory.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-m",
            "pytest",
            *(case.nodeid for case in task.cases),
            f"--basetemp={temporary_directory}",
            f"--junitxml={directory / 'pytest.xml'}",
        ]
        if task.requirements.cuda_count:
            command.append("-v")
        if self.strict_requirements:
            command.append("--strict-requirements")
        command.append(f"--xpool-task-artifact-dir={artifact_directory}")
        lease: GpuLease | None = None
        if task.requirements.cuda_count:
            assert self.gpu_pool is not None
            lease = self.gpu_pool.try_lease(task.requirements.cuda_count)
            if lease is None:
                raise SuiteInfrastructureFailure(f"scheduler selected GPU task {task.key} without capacity")
        gpu_assignments = (
            ",".join(f"{self.gpu_pool.physical_index_by_uuid[uuid]}:{uuid}" for uuid in lease.uuids)
            if lease is not None and self.gpu_pool is not None
            else "none"
        )
        print(f"START {task.key} gpus={gpu_assignments}", flush=True)
        if lease is not None:
            for case in task.cases:
                print(f"ASSIGN {case.nodeid} gpus={gpu_assignments}", flush=True)
        try:
            scope = SupervisedTaskScope.start(
                task.key,
                command,
                cwd=self.repository_root,
                env=self.task_environment(lease),
                log_path=directory / "pytest.log",
                timeout_seconds=task.timeout_seconds,
            )
        except TaskStartFailure:
            if lease is not None:
                assert self.gpu_pool is not None
                self.gpu_pool.release(lease)
            raise
        except BaseException:
            if lease is not None:
                self.retained_leases.append(lease)
            raise
        self.active[task.key] = RunningTask(task, directory, scope, lease)

    def collect_completed_tasks(self) -> tuple[TaskOutcome, ...]:
        """Poll every active scope and release only proven-empty task leases."""

        completed: list[TaskOutcome] = []
        for key, running in tuple(self.active.items()):
            completion = running.scope.poll()
            if completion is None:
                continue
            running.scope.close()
            if running.lease is not None:
                assert self.gpu_pool is not None
                self.gpu_pool.release(running.lease)
            report: PytestTaskReport | None = None
            if completion.kind is TaskCompletionKind.EXITED and completion.returncode in {0, 1}:
                try:
                    report = PytestTaskReport.read(running.directory / "pytest.xml", running.task.cases)
                except ValueError as error:
                    print(f"INVALID {key}: {error}; see {running.directory}", file=sys.stderr)
            outcome = TaskOutcome(running.task, completion, report, running.directory)
            self.outcomes[key] = outcome
            del self.active[key]
            completed.append(outcome)
            status = "PASS" if outcome.result_code == 0 else "FAIL" if outcome.result_code == 1 else "ERROR"
            pytest_summary = outcome.report.summary() if outcome.report is not None else "pytest-report=unavailable"
            print(
                f"{status} {key} ({completion.kind.value}, returncode={completion.returncode}); "
                f"{pytest_summary}; "
                f"log={running.directory / 'pytest.log'} junit={running.directory / 'pytest.xml'}"
            )
        return tuple(completed)

    def cancel_active_tasks(self) -> None:
        """Fan out cancellation and release only scopes proven empty."""

        running_tasks = tuple(self.active.values())
        terminal_failures: list[str] = []
        recovered_failures: list[str] = []
        if running_tasks:
            try:
                SupervisedTaskScope.terminate_all(tuple(running.scope for running in running_tasks))
            except TaskScopeFailure as error:
                terminal_failures.append(str(error))
        for running in running_tasks:
            if running.scope.state in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
                try:
                    running.scope.close()
                except TaskSupervisionFailure as error:
                    recovered_failures.append(str(error))
                    continue
                if running.lease is not None:
                    assert self.gpu_pool is not None
                    self.gpu_pool.release(running.lease)
                self.active.pop(running.task.key, None)
        failed_scopes = tuple(
            running.scope for running in running_tasks if running.scope.state is TaskScopeState.FAILED
        )
        if failed_scopes and not terminal_failures:
            try:
                SupervisedTaskScope.terminate_all(failed_scopes)
            except TaskScopeFailure as error:
                terminal_failures.append(str(error))
            else:
                for running in running_tasks:
                    if running.scope.state is not TaskScopeState.DRAINED:
                        continue
                    try:
                        running.scope.close()
                    except TaskSupervisionFailure as error:
                        terminal_failures.append(str(error))
                        continue
                    if running.lease is not None:
                        assert self.gpu_pool is not None
                        self.gpu_pool.release(running.lease)
                    self.active.pop(running.task.key, None)
        try:
            drain_unprotected_subreaper_descendants()
        except TaskScopeFailure as error:
            terminal_failures.append(str(error))
        if terminal_failures:
            raise SuiteInfrastructureFailure("; ".join((*recovered_failures, *terminal_failures)))
        if recovered_failures:
            raise SuiteInfrastructureFailure("; ".join(recovered_failures))

    def task_environment(self, lease: GpuLease | None) -> dict[str, str]:
        """Build one task-local process environment without ownership tokens."""

        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(self.repository_root / ".xpool-cache" / "pycache")
        environment["CUDA_VISIBLE_DEVICES"] = "" if lease is None else ",".join(lease.uuids)
        return environment

    def require_mps_controller(self, context: str) -> None:
        """Fail infrastructure-wide when the externally owned controller is unhealthy."""

        result = probe_mps_controller()
        if not result.online:
            raise SuiteInfrastructureFailure(f"MPS is unhealthy {context}: {result.diagnostic}")

    def artifact_group_results(self) -> tuple[ArtifactGroupResult, ...]:
        """Classify every complete cross-task artifact group."""

        groups: dict[tuple[str, str], list[CollectedTestCase]] = defaultdict(list)
        task_by_nodeid = {case.nodeid: task for task in self.tasks for case in task.cases}
        for case in self.plan.cases:
            if case.artifact_group is not None:
                groups[(case.artifact_group.kind, case.artifact_group.name)].append(case)
        results: list[ArtifactGroupResult] = []
        for (kind, name), cases in sorted(groups.items()):
            tasks = tuple(task_by_nodeid[case.nodeid] for case in cases)
            outcomes = tuple(self.outcomes.get(task.key) for task in tasks)
            if any(
                outcome is None
                or outcome.completion.kind is not TaskCompletionKind.EXITED
                or outcome.result_code == 2
                or outcome.report is None
                for outcome in outcomes
            ):
                continue
            reports = tuple(
                outcome.report.case(case.nodeid)
                for case, outcome in zip(cases, outcomes, strict=True)
                if outcome is not None and outcome.report is not None
            )
            declared_group = cases[0].artifact_group
            assert declared_group is not None
            group = ArtifactGroupRef(kind, name, declared_group.expected_case_count)
            adapter = self.artifact_group_adapters[kind]
            result = adapter.evaluate(
                group,
                reports,
                tuple(outcome.directory / "artifacts" for outcome in outcomes if outcome is not None),
            )
            if result.name != name or result.result_code not in {0, 1, 2}:
                raise SuiteInfrastructureFailure(f"artifact adapter {kind!r} returned an invalid result: {result!r}")
            results.append(result)
        return tuple(results)

    @staticmethod
    def report_artifact_groups(results: tuple[ArtifactGroupResult, ...]) -> None:
        """Print one concise diagnostic for every classified artifact group."""

        for result in results:
            message = f"ARTIFACT GROUP {result.name}: code={result.result_code}"
            if result.detail is not None:
                message += f": {result.detail}"
            print(message, file=sys.stderr if result.result_code else sys.stdout)

    @property
    def resources_releasable(self) -> bool:
        """Return whether every borrowed task lease was safely returned."""

        return (
            not self.retained_leases and not self.active and (self.gpu_pool is None or not self.gpu_pool.active_leases)
        )
