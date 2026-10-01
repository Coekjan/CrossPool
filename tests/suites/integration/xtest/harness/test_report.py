from pathlib import Path

import pytest

import xtest.cli
from xkit.results import RunStore
from xkit.supervisor import TaskCompletion, TaskCompletionKind
from xtest.harness.report import TaskReportRecord, TestResultWriter, TestRunManifest, TestRunReport, report_test_runs
from xtest.harness.runner.artifact import ArtifactGroupResult
from xtest.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus


def test_report_preserves_original_failure_and_group_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = RunStore(tmp_path / "runs").start("20260927-100000-1-1")
    writer = TestResultWriter(
        run.directory,
        TestRunManifest(
            run_id=run.directory.name,
            selected_suites=("unit",),
            strict_requirements=True,
            python_cases=("example",),
            tasks={"unit": ("example",)},
        ),
    )
    (run.directory / "unit").mkdir()
    (run.directory / "unit/pytest.xml").write_text("<testsuites/>", encoding="utf-8")
    writer.task(
        TaskReportRecord(
            key="unit",
            stage="unit",
            completion=TaskCompletion(TaskCompletionKind.EXITED, 1, None),
            result_code=1,
            elapsed_seconds=0.5,
            artifact_directory="unit",
            cases=(PytestCaseReport("example", PytestCaseStatus.FAILED, "assertion failed", 0.25),),
        )
    )
    writer.groups((ArtifactGroupResult("parity", 1, "different token output"),))
    writer.finish(1, cleanup_verified=True)
    run.complete()
    before = (run.directory / "results.json").read_bytes()
    monkeypatch.chdir(tmp_path)

    assert xtest.cli.main(["report", str(run.directory), "--output", str(tmp_path / "report")]) == 0
    report = TestRunReport.load(run.directory)
    assert report.summary()["failed"] == 1
    assert report.summary()["original_result_code"] == 1
    assert report.manifest.strict_requirements
    assert report.results.groups[0].result_code == 1
    assert (run.directory / "results.json").read_bytes() == before
    assert "assertion failed" in (tmp_path / "report/report.md").read_text()


def test_report_marks_interrupted_checkpoint_unavailable(tmp_path: Path) -> None:
    run = RunStore(tmp_path / "runs").start("20260927-100000-1-1")
    TestResultWriter(
        run.directory,
        TestRunManifest(
            run_id=run.directory.name,
            selected_suites=("unit",),
            strict_requirements=False,
            python_cases=("never-started",),
        ),
    )
    run.complete()

    report = TestRunReport.load(run.directory)
    assert report.summary()["unexecuted_or_unavailable"] == 1
    assert report.summary()["evidence_complete"] is False
    assert report.summary()["cleanup_verified"] is None
    assert report.summary()["original_result_code"] is None


def test_report_rejects_active_run_and_label_mismatch(tmp_path: Path) -> None:
    run = RunStore(tmp_path / "runs").start("20260927-100000-1-1")
    try:
        with pytest.raises(BlockingIOError):
            report_test_runs((run.directory,), output=tmp_path / "report")
        with pytest.raises(ValueError, match="one-to-one"):
            report_test_runs((run.directory,), output=tmp_path / "report", labels=())
        assert not (tmp_path / "report").exists()
    finally:
        run.complete()
