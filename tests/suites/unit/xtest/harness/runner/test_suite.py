from __future__ import annotations

import logging
import signal
import xml.etree.ElementTree
from dataclasses import dataclass
from pathlib import Path

import pytest

import xtest.harness.runner.artifact
import xtest.harness.runner.console
import xtest.harness.runner.plan
import xtest.harness.runner.pytest_report
import xtest.harness.runner.suite
import xtest.harness.runner.task
import xtest.harness.sglang.serving.alignment
from xkit import ResourceRequirements
from xkit.gpu import GpuPool
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.supervisor import (
    TaskCompletion,
    TaskCompletionKind,
    TaskScopeFailure,
    TaskScopeState,
    TaskStartFailure,
)
from xpool.model import ModelId
from xpool.mps import MpsProbeResult
from xtest.harness.sglang.serving.alignment import (
    SERVING_GRAPH_ARTIFACT_FILENAME,
    ServingGraphArtifact,
    TokenOutput,
)
from xtest.harness.support.config import TEST_MODEL_ID


def test_task_formatter_colors_only_status_on_tty() -> None:
    record = logging.LogRecord("xtest.runner", logging.INFO, __file__, 1, "case elapsed=1.0s", (), None)
    record.status = "PASSED"

    assert xtest.harness.runner.console.TaskFormatter(color=False).format(record) == "PASSED case elapsed=1.0s"
    assert xtest.harness.runner.console.TaskFormatter(color=True).format(record) == (
        "\x1b[32mPASSED\x1b[0m case elapsed=1.0s"
    )


def test_compiler_builds_stage_tasks_with_exact_gpu_batching() -> None:
    cpu_requirements = requirements()
    gpu_one = requirements(cuda_count=1, requires_mps=True)
    gpu_two = requirements(cuda_count=2, requires_mps=True)
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case("tests/suites/unit/test_alpha.py", "test_alpha", requirements=cpu_requirements),
            case("tests/suites/unit/test_beta.py", "test_beta", requirements=cpu_requirements, timeout=20),
            case("tests/suites/integration/test_cpu.py", "test_cpu", requirements=cpu_requirements),
            case("tests/suites/integration/test_gpu.py", "test_one", requirements=gpu_one, estimate=4),
            case("tests/suites/integration/test_gpu.py", "test_two", requirements=gpu_one, estimate=6),
            case("tests/suites/integration/test_gpu.py", "test_wide", requirements=gpu_two, estimate=8),
            case("tests/suites/integration/test_other.py", "test_other", requirements=gpu_one, estimate=2),
            case("tests/suites/e2e/test_e2e_model.py", "test_model[eager]", requirements=gpu_two, estimate=30),
            case("tests/suites/e2e/test_e2e_model.py", "test_model[full]", requirements=gpu_two, estimate=40),
            case(
                "tests/suites/models/Qwen/Qwen3-0.6B/test_model.py",
                "test_model[full]",
                requirements=gpu_two,
                estimate=50,
            ),
        )
    )

    tasks = xtest.harness.runner.task.compile_execution_tasks(plan)

    assert tuple(task.stage for task in tasks) == (
        xtest.harness.runner.plan.TestStage.UNIT,
        xtest.harness.runner.plan.TestStage.INTEGRATION,
        xtest.harness.runner.plan.TestStage.INTEGRATION,
        xtest.harness.runner.plan.TestStage.INTEGRATION,
        xtest.harness.runner.plan.TestStage.INTEGRATION,
        xtest.harness.runner.plan.TestStage.E2E,
        xtest.harness.runner.plan.TestStage.E2E,
        xtest.harness.runner.plan.TestStage.MODELS,
    )
    assert tuple(len(task.cases) for task in tasks) == (2, 1, 2, 1, 1, 1, 1, 1)
    assert tasks[0].key == "unit"
    assert tasks[0].estimated_duration_seconds == 30
    assert tasks[0].timeout_seconds == 30
    assert tasks[2].estimated_duration_seconds == 10
    assert tasks[2].timeout_seconds == 20
    assert len({task.key for task in tasks}) == len(tasks)
    assert all(not xtest.harness.runner.task.TASK_KEY_CHARACTER_PATTERN.search(task.key) for task in tasks)


def test_compiler_uses_timeout_when_estimate_is_absent_and_merges_cpu_requirements() -> None:
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case(
                "tests/suites/integration/test_alpha.py",
                "test_alpha",
                requirements=requirements(requires_config=True, model_ids=(ModelId("org/alpha"),)),
                timeout=12,
            ),
            case(
                "tests/suites/integration/test_beta.py",
                "test_beta",
                requirements=requirements(requires_config=True, model_ids=(ModelId("org/beta"), ModelId("org/alpha"))),
                timeout=18,
                estimate=3,
            ),
        )
    )

    (task,) = xtest.harness.runner.task.compile_execution_tasks(plan)

    assert task.key == "integration-cpu"
    assert task.estimated_duration_seconds == 15
    assert task.timeout_seconds == 30
    assert task.requirements.model_ids == (ModelId("org/alpha"), ModelId("org/beta"))


def test_execution_task_rejects_requirement_drift() -> None:
    collected = case("tests/suites/integration/test_gpu.py", "test_gpu", requirements=requirements(cuda_count=1))

    with pytest.raises(ValueError, match="requirements"):
        xtest.harness.runner.task.ExecutionTask(
            key="invalid",
            stage=xtest.harness.runner.plan.TestStage.INTEGRATION,
            cases=(collected,),
            requirements=requirements(cuda_count=2),
            estimated_duration_seconds=10,
            timeout_seconds=10,
        )


def test_pytest_task_report_matches_exact_parametrized_nodeids(tmp_path: Path) -> None:
    expected = (
        case("tests/suites/e2e/test_e2e_model.py", "test_model[eager]", requirements=requirements()),
        case("tests/suites/e2e/test_e2e_model.py", "test_model[full]", requirements=requirements()),
    )
    path = tmp_path / "pytest.xml"
    path.write_text(
        '<testsuites><testsuite name="pytest" tests="2" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.suites.e2e.test_e2e_model" name="test_model[eager]" time="0.25"/>'
        '<testcase classname="tests.suites.e2e.test_e2e_model" name="test_model[full]" time="0.25"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    report = xtest.harness.runner.pytest_report.PytestTaskReport.read(path, expected)

    assert tuple(case.nodeid for case in report.cases) == tuple(case.nodeid for case in expected)
    assert all(case.status is xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED for case in report.cases)


def test_pytest_task_report_rejects_summary_drift_and_xfail(tmp_path: Path) -> None:
    expected = (case("tests/suites/e2e/test_e2e_model.py", "test_model[eager]", requirements=requirements()),)
    path = tmp_path / "pytest.xml"
    junit = (
        '<testsuites><testsuite name="pytest" tests="1" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.suites.e2e.test_e2e_model" name="test_model[eager]" time="0.25"/>'
        "</testsuite></testsuites>"
    )
    path.write_text(junit, encoding="utf-8")
    tree = xml.etree.ElementTree.parse(path)
    suite = tree.getroot().find("testsuite")
    assert suite is not None
    suite.set("tests", "2")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="summary counts"):
        xtest.harness.runner.pytest_report.PytestTaskReport.read(path, expected)

    path.write_text(junit, encoding="utf-8")
    tree = xml.etree.ElementTree.parse(path)
    suite = tree.getroot().find("testsuite")
    assert suite is not None
    testcase = suite.find("testcase")
    assert testcase is not None
    xml.etree.ElementTree.SubElement(
        testcase,
        "skipped",
        {"type": "pytest.xfail", "message": "reason"},
    )
    suite.set("skipped", "1")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match=r"pytest\.xfail"):
        xtest.harness.runner.pytest_report.PytestTaskReport.read(path, expected)


def test_task_outcome_composes_lifecycle_and_pytest_facts(tmp_path: Path) -> None:
    task = xtest.harness.runner.task.build_task(
        "unit",
        (case("tests/suites/unit/test_alpha.py", "test_alpha", requirements=requirements()),),
    )
    passed = xtest.harness.runner.pytest_report.PytestTaskReport(
        (
            xtest.harness.runner.pytest_report.PytestCaseReport(
                task.cases[0].nodeid,
                xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
                None,
            ),
        )
    )
    failed = xtest.harness.runner.pytest_report.PytestTaskReport(
        (
            xtest.harness.runner.pytest_report.PytestCaseReport(
                task.cases[0].nodeid,
                xtest.harness.runner.pytest_report.PytestCaseStatus.FAILED,
                "assertion failed",
            ),
        )
    )

    assert (
        xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.EXITED, 0, None),
            passed,
            tmp_path,
        ).result_code
        == 0
    )
    assert (
        xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.EXITED, 1, None),
            failed,
            tmp_path,
        ).result_code
        == 1
    )
    assert (
        xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.INFRASTRUCTURE_FAILED, None, "internal failure"),
            None,
            tmp_path,
        ).result_code
        == 2
    )
    assert (
        xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.EXITED, 0, None),
            failed,
            tmp_path,
        ).result_code
        == 2
    )
    assert (
        xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "deadline"),
            None,
            tmp_path,
        ).result_code
        == 2
    )


def test_suite_runner_stops_after_failed_unit_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[str] = []

    class ScopeFactory:
        @staticmethod
        def start(
            name: str,
            command: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            log_path: Path,
            timeout_seconds: float,
        ) -> FakeScope:
            del cwd, env, timeout_seconds
            starts.append(name)
            returncode = 1 if name == "unit" else 0
            write_pytest_junit(command, log_path.parent / "pytest.xml", failed=returncode == 1)
            return FakeScope(TaskCompletion(TaskCompletionKind.EXITED, returncode, None))

    monkeypatch.setattr(xtest.harness.runner.suite, "SupervisedTaskScope", ScopeFactory)
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case("tests/suites/unit/test_alpha.py", "test_alpha", requirements=requirements()),
            case("tests/suites/integration/test_beta.py", "test_beta", requirements=requirements()),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=tmp_path,
        catalogue_path=tmp_path / "tests/tests.toml",
        run_directory=tmp_path / "run",
        strict_requirements=False,
    )

    assert runner.run() == 1
    assert starts == ["unit"]


@pytest.mark.parametrize("stop", (False, True))
def test_suite_runner_preserves_e2e_failure_during_completion_or_stop(
    stop: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    starts: list[str] = []

    class ScopeFactory:
        @staticmethod
        def start(
            name: str,
            command: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            log_path: Path,
            timeout_seconds: float,
        ) -> FakeScope:
            del cwd, env, timeout_seconds
            starts.append(name)
            failed = any("test_alpha" in argument for argument in command)
            write_pytest_junit(command, log_path.parent / "pytest.xml", failed=failed)
            if stop and not failed:
                runner.request_stop(signal.SIGTERM, None)
            return FakeScope(
                TaskCompletion(TaskCompletionKind.EXITED, int(failed), None),
                polls_before_completion=int(not failed),
            )

        @staticmethod
        def terminate_all(scopes: tuple[FakeScope, ...]) -> None:
            for scope in scopes:
                scope.state = TaskScopeState.DRAINED

    monkeypatch.setattr(xtest.harness.runner.suite, "SupervisedTaskScope", ScopeFactory)
    monkeypatch.setattr(xtest.harness.runner.suite, "drain_unprotected_subreaper_descendants", lambda: None)
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case("tests/suites/e2e/test_e2e_alpha.py", "test_alpha", requirements=requirements()),
            case("tests/suites/e2e/test_e2e_beta.py", "test_beta", requirements=requirements()),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=tmp_path,
        catalogue_path=tmp_path / "tests/tests.toml",
        run_directory=tmp_path / "run",
        strict_requirements=False,
    )

    assert runner.run() == (128 + signal.SIGTERM if stop else 1)
    assert len(starts) == 2
    assert "STAGE e2e: code=1" in capsys.readouterr().out


def test_suite_runner_backfills_gpu_pool_and_builds_exact_pytest_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    starts: list[tuple[str, list[str], dict[str, str]]] = []
    poll_counts = {"test-wide-a": 2, "test-wide-b": 0, "test-small": 0}
    gpu_pool = GpuPool(("GPU-a", "GPU-b", "GPU-c"), {"GPU-a": 0, "GPU-b": 1, "GPU-c": 2})

    class ScopeFactory:
        @staticmethod
        def start(
            name: str,
            command: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            log_path: Path,
            timeout_seconds: float,
        ) -> FakeScope:
            del cwd, timeout_seconds
            starts.append((name, command, env))
            write_pytest_junit(command, log_path.parent / "pytest.xml")
            return FakeScope(
                TaskCompletion(TaskCompletionKind.EXITED, 0, None),
                polls_before_completion=poll_counts[case_name(command)],
            )

    monkeypatch.setattr(xtest.harness.runner.suite, "SupervisedTaskScope", ScopeFactory)
    monkeypatch.setattr(
        xtest.harness.runner.suite,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case(
                "tests/suites/integration/test_wide_a.py",
                "test-wide-a",
                requirements=requirements(cuda_count=2, requires_mps=True),
                estimate=100,
            ),
            case(
                "tests/suites/integration/test_wide_b.py",
                "test-wide-b",
                requirements=requirements(cuda_count=2, requires_mps=True),
                estimate=90,
            ),
            case(
                "tests/suites/integration/test_small.py",
                "test-small",
                requirements=requirements(cuda_count=1, requires_mps=True),
                estimate=80,
            ),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=tmp_path,
        catalogue_path=tmp_path / "tests/tests.toml",
        run_directory=tmp_path / "run",
        strict_requirements=True,
        gpu_pool=gpu_pool,
    )

    with caplog.at_level("INFO", logger="xtest.runner"):
        assert runner.run() == 0
    output = caplog.text
    assert "gpus=0:GPU-a,1:GPU-b" in output
    assert "gpus=2:GPU-c" in output
    assert "PASSED" in tuple(getattr(record, "status", None) for record in caplog.records)
    assert "tests/suites/integration/test_small.py::test-small elapsed=0.250s" in output
    task_starts = starts
    assert tuple(case_name(command) for _, command, _ in task_starts) == (
        "test-wide-a",
        "test-small",
        "test-wide-b",
    )
    assert tuple(environment["CUDA_VISIBLE_DEVICES"] for _, _, environment in task_starts) == (
        "GPU-a,GPU-b",
        "GPU-c",
        "GPU-a,GPU-b",
    )
    assert all("--strict-requirements" in command for _, command, _ in task_starts)
    assert all("-v" in command for _, command, _ in task_starts)
    assert all(any(argument.startswith("--basetemp=") for argument in command) for _, command, _ in task_starts)
    assert all(any(argument.startswith("--junitxml=") for argument in command) for _, command, _ in task_starts)
    assert all(f"--xpool-test-catalog={tmp_path / 'tests/tests.toml'}" in command for _, command, _ in task_starts)
    assert all(
        any(argument.startswith("--xpool-task-artifact-dir=") for argument in command) for _, command, _ in task_starts
    )
    assert runner.resources_releasable
    assert not gpu_pool.closed


@pytest.mark.parametrize(
    ("failure", "lease_retained"),
    (
        (TaskStartFailure("startup rolled back"), False),
        (TaskScopeFailure("startup scope unproven"), True),
    ),
)
def test_suite_runner_classifies_gpu_lease_after_start_failure(
    failure: RuntimeError,
    lease_retained: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_pool = GpuPool(("GPU-a",), {"GPU-a": 0})

    class ScopeFactory:
        @staticmethod
        def start(*args: object, **kwargs: object) -> FakeScope:
            del args, kwargs
            raise failure

    monkeypatch.setattr(xtest.harness.runner.suite, "SupervisedTaskScope", ScopeFactory)
    monkeypatch.setattr(
        xtest.harness.runner.suite,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case(
                "tests/suites/integration/test_gpu.py",
                "test_gpu",
                requirements=requirements(cuda_count=1, requires_mps=True),
            ),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=tmp_path,
        catalogue_path=tmp_path / "tests/tests.toml",
        run_directory=tmp_path / "run",
        strict_requirements=False,
        gpu_pool=gpu_pool,
    )

    with pytest.raises(type(failure), match=str(failure)):
        runner.start_task(runner.tasks[0])

    assert bool(runner.retained_leases) is lease_retained
    assert bool(gpu_pool.active_leases) is lease_retained
    assert runner.resources_releasable is not lease_retained


def test_suite_runner_prepares_directory_before_gpu_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_pool = GpuPool(("GPU-a",), {"GPU-a": 0})
    monkeypatch.setattr(
        xtest.harness.runner.suite,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, 100, "online"),
    )
    blocked_run_directory = tmp_path / "blocked"
    blocked_run_directory.write_text("not a directory", encoding="utf-8")
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case(
                "tests/suites/integration/test_gpu.py",
                "test_gpu",
                requirements=requirements(cuda_count=1, requires_mps=True),
            ),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=tmp_path,
        catalogue_path=tmp_path / "tests/tests.toml",
        run_directory=blocked_run_directory,
        strict_requirements=False,
        gpu_pool=gpu_pool,
    )

    with pytest.raises(NotADirectoryError):
        runner.start_task(runner.tasks[0])

    assert not gpu_pool.active_leases
    assert runner.resources_releasable


@pytest.mark.parametrize(
    ("statuses", "details", "expected_status"),
    (
        (
            (
                xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
                xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
            ),
            (None, None),
            xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus.COMPARED,
        ),
        (
            (
                xtest.harness.runner.pytest_report.PytestCaseStatus.SKIPPED,
                xtest.harness.runner.pytest_report.PytestCaseStatus.SKIPPED,
            ),
            ("missing config", "missing config"),
            xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus.SKIPPED,
        ),
        (
            (
                xtest.harness.runner.pytest_report.PytestCaseStatus.SKIPPED,
                xtest.harness.runner.pytest_report.PytestCaseStatus.SKIPPED,
            ),
            ("missing config", "missing weights"),
            xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus.INCONSISTENT,
        ),
        (
            (
                xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
                xtest.harness.runner.pytest_report.PytestCaseStatus.SKIPPED,
            ),
            (None, "missing config"),
            xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus.INCONSISTENT,
        ),
        (
            (
                xtest.harness.runner.pytest_report.PytestCaseStatus.FAILED,
                xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
            ),
            ("assertion failed", None),
            xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus.FAILED,
        ),
    ),
)
def test_serving_graph_group_results_follow_case_outcomes(
    statuses: tuple[
        xtest.harness.runner.pytest_report.PytestCaseStatus, xtest.harness.runner.pytest_report.PytestCaseStatus
    ],
    details: tuple[str | None, str | None],
    expected_status: xtest.harness.sglang.serving.alignment.ServingGraphGroupStatus,
    tmp_path: Path,
) -> None:
    runner = serving_graph_runner(tmp_path, statuses, details)

    (result,) = runner.artifact_group_results()

    assert result.name == "example"
    assert result.detail is not None and result.detail.startswith(expected_status.value)


def test_serving_graph_omits_group_without_complete_ordinary_outcomes(tmp_path: Path) -> None:
    runner = serving_graph_runner(
        tmp_path,
        (
            xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
            xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED,
        ),
        (None, None),
    )
    first = runner.tasks[0]
    runner.outcomes[first.key] = xtest.harness.runner.suite.TaskOutcome(
        first,
        TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "deadline"),
        None,
        runner.outcomes[first.key].directory,
    )

    assert runner.artifact_group_results() == ()


@dataclass(slots=True)
class FakeScope:
    completion: TaskCompletion
    polls_before_completion: int = 0
    state: TaskScopeState = TaskScopeState.RUNNING

    def poll(self) -> TaskCompletion | None:
        if self.polls_before_completion:
            self.polls_before_completion -= 1
            return None
        self.state = TaskScopeState.COMPLETED
        return self.completion

    def wait(self) -> TaskCompletion:
        self.state = TaskScopeState.COMPLETED
        return self.completion

    def close(self) -> None:
        self.state = TaskScopeState.CLOSED


def case_name(command: list[str]) -> str:
    nodeid = next(argument for argument in command if argument.startswith("tests/"))
    return nodeid.rsplit("::", maxsplit=1)[1]


def write_pytest_junit(command: list[str], path: Path, *, failed: bool = False) -> None:
    """Write the exact xunit2 subset emitted by one fake pytest task."""

    nodeids = tuple(argument for argument in command if argument.startswith("tests/"))
    suite = xml.etree.ElementTree.Element(
        "testsuite",
        {
            "name": "pytest",
            "errors": "0",
            "failures": "1" if failed else "0",
            "skipped": "0",
            "tests": str(len(nodeids)),
        },
    )
    for index, nodeid in enumerate(nodeids):
        classname, name = xtest.harness.runner.pytest_report.PytestTaskReport.junit_identity(nodeid)
        testcase = xml.etree.ElementTree.SubElement(
            suite, "testcase", {"classname": classname, "name": name, "time": "0.25"}
        )
        if failed and index == 0:
            xml.etree.ElementTree.SubElement(testcase, "failure", {"message": "assertion failed"})
    root = xml.etree.ElementTree.Element("testsuites", {"name": "pytest tests"})
    root.append(suite)
    xml.etree.ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def case(
    path: str,
    name: str,
    *,
    requirements: ResourceRequirements,
    timeout: float = 10,
    estimate: float | None = None,
    artifact_group: xtest.harness.runner.artifact.ArtifactGroupRef | None = None,
) -> xtest.harness.runner.plan.CollectedTestCase:
    return xtest.harness.runner.plan.CollectedTestCase(
        path=path,
        nodeid=f"{path}::{name}",
        stage=xtest.harness.runner.plan.TestStage.from_path(path),
        requirements=requirements,
        estimated_duration_seconds=estimate,
        timeout_seconds=timeout,
        artifact_group=artifact_group,
    )


def requirements(
    *,
    cuda_count: int = 0,
    requires_mps: bool = False,
    requires_config: bool = False,
    model_ids: tuple[ModelId, ...] = (),
) -> ResourceRequirements:
    return ResourceRequirements(
        cuda_count=cuda_count,
        requires_mps=requires_mps,
        requires_config=requires_config,
        model_ids=model_ids,
    )


def serving_graph_runner(
    root: Path,
    statuses: tuple[
        xtest.harness.runner.pytest_report.PytestCaseStatus, xtest.harness.runner.pytest_report.PytestCaseStatus
    ],
    details: tuple[str | None, str | None],
) -> xtest.harness.runner.suite.SuiteRunner:
    """Build one fully classified two-mode serving graph group without subprocesses."""

    group = xtest.harness.runner.artifact.ArtifactGroupRef("serving_graph", "example", 2)
    plan = xtest.harness.runner.plan.TestPlan(
        (
            case(
                "tests/suites/e2e/test_e2e_model.py",
                "test_model[eager]",
                requirements=requirements(),
                artifact_group=group,
            ),
            case(
                "tests/suites/e2e/test_e2e_model.py",
                "test_model[full]",
                requirements=requirements(),
                artifact_group=group,
            ),
        )
    )
    runner = xtest.harness.runner.suite.SuiteRunner(
        plan,
        repository_root=root,
        catalogue_path=root / "tests/tests.toml",
        run_directory=root / "run",
        strict_requirements=False,
        artifact_group_adapters=(xtest.harness.sglang.serving.alignment.ServingGraphAdapter(),),
    )
    graph_modes = (SglangGraphMode.EAGER, SglangGraphMode.DECODE_FULL)
    for task, status, detail, graph_mode in zip(runner.tasks, statuses, details, graph_modes, strict=True):
        directory = root / task.key
        artifact_directory = directory / "artifacts"
        artifact_directory.mkdir(parents=True)
        report = xtest.harness.runner.pytest_report.PytestTaskReport(
            (xtest.harness.runner.pytest_report.PytestCaseReport(task.cases[0].nodeid, status, detail),)
        )
        returncode = 1 if status is xtest.harness.runner.pytest_report.PytestCaseStatus.FAILED else 0
        runner.outcomes[task.key] = xtest.harness.runner.suite.TaskOutcome(
            task,
            TaskCompletion(TaskCompletionKind.EXITED, returncode, None),
            report,
            directory,
        )
        if status is xtest.harness.runner.pytest_report.PytestCaseStatus.PASSED:
            ServingGraphArtifact(
                graph_settings=graph_mode.settings(),
                outputs=(TokenOutput(TEST_MODEL_ID, (1, 2, 3)),),
            ).write(artifact_directory / SERVING_GRAPH_ARTIFACT_FILENAME)
    return runner
