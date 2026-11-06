"""Canonical test-suite command-line behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import xkit.config
import xtest.harness.runner.execution
import xtest.harness.runner.plan
import xtest.harness.runner.selection
from xkit import ResourceRequirements
from xkit.config import ToolConfigRecord
from xkit.device import DevicePool
from xkit.supervisor import TaskCompletion, TaskCompletionKind
from xtest.harness.report import TestResultWriter, TestRunManifest
from xtest.harness.runner.ctest import CtestRunResult, CtestSuite
from xtest.harness.runner.pytest_report import PytestCaseStatus
from xtest.harness.support.config import reset_development_config


@pytest.mark.usefixtures(reset_development_config.__name__)
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("record_results", (False, True))
def test_native_skip_fails_strict_stage_without_changing_original_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool, record_results: bool
) -> None:
    config = xkit.config.init_global_config()
    pool = DevicePool((), {})
    junit = tmp_path / "ctest.xml"
    junit.write_text(
        '<testsuite><testcase name="cext.skipped"><skipped message="device unavailable">'
        "device unavailable</skipped></testcase></testsuite>",
        encoding="utf-8",
    )
    completion = TaskCompletion(TaskCompletionKind.EXITED, 0, None)
    native = CtestRunResult(0, tmp_path / "ctest.log", junit, completion, 0.1)
    monkeypatch.setattr(xtest.harness.runner.execution, "collect_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(DevicePool, "from_environment", classmethod(lambda cls: pool))
    monkeypatch.setattr(CtestSuite, "inventory", lambda self: ("cext.skipped",))
    monkeypatch.setattr(CtestSuite, "run", lambda self, **kwargs: native)
    writer = (
        TestResultWriter(
            tmp_path,
            TestRunManifest(
                run_id="native-skip",
                tool_config=config.record(),
                selected_suites=("cext",),
                strict_requirements=strict,
            ),
        )
        if record_results
        else None
    )

    result = xtest.harness.runner.execution.execute_test_run(
        ("cext",), (), strict_requirements=strict, run_directory=tmp_path, result_writer=writer
    )

    assert result == int(strict)
    assert pool.closed
    if writer is not None:
        task = writer.results.tasks[0]
        assert task.completion == completion
        assert task.result_code == int(strict)
        assert task.cases[0].status is PytestCaseStatus.SKIPPED
        assert task.cases[0].detail == "device unavailable"
        assert writer.results.stages == {"cext": int(strict)}
        assert writer.results.cleanup_verified


@pytest.mark.usefixtures(reset_development_config.__name__)
def test_execution_uses_collection_scope_and_parent_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = "tests/suites/models/Qwen/Qwen3-0.6B"
    config = xkit.config.init_global_config(cli={"xtest_suites": ["Qwen/Qwen3-0.6B"]})
    collected_selectors: list[tuple[str, ...]] = []
    collected = xtest.harness.runner.plan.CollectedTestCase(
        path=f"{model_directory}/test_sglang_model_qualification.py",
        nodeid=f"{model_directory}/test_sglang_model_qualification.py::test_ffn_numerical[example]",
        stage=xtest.harness.runner.plan.TestStage.MODELS,
        requirements=ResourceRequirements(0, False, ()),
        estimated_duration_seconds=1,
        timeout_seconds=10,
        artifact_group=None,
    )

    class FakeCollectionWorker:
        def __init__(
            self,
            *,
            repository_root: Path,
            run_directory: Path,
            selectors: tuple[str, ...],
            strict_requirements: bool,
            catalogue_path: Path,
            tool_config: ToolConfigRecord,
        ) -> None:
            del repository_root, run_directory, strict_requirements
            assert catalogue_path == Path.cwd().resolve() / "tests/tests.toml"
            collected_selectors.append(selectors)
            assert tool_config.settings.xtest.suites == config.xtest.suites

        def collect(self) -> xtest.harness.runner.plan.TestPlan:
            return xtest.harness.runner.plan.TestPlan((collected,), config.xtest.suites)

    class FakeSuiteRunner:
        resources_releasable = True

        def __init__(self, plan: xtest.harness.runner.plan.TestPlan, **kwargs: object) -> None:
            assert plan.cases == (collected,)

        def request_stop(self, signal_number: int, frame: object) -> None:
            del signal_number, frame

        def run(self) -> int:
            return 0

    monkeypatch.setattr(xtest.harness.runner.selection, "CollectionWorker", FakeCollectionWorker)
    monkeypatch.setattr(xtest.harness.runner.execution, "SuiteRunner", FakeSuiteRunner)

    assert (
        xtest.harness.runner.execution.execute_test_run(
            ("Qwen/Qwen3-0.6B",), (), strict_requirements=False, run_directory=tmp_path
        )
        == 0
    )
    assert collected_selectors == [()]
