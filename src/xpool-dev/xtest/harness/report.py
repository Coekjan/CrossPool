"""Retained test verdicts and checkout-independent offline reports."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, model_validator

from xkit.config import ToolConfigRecord
from xkit.results import RunStore, write_json
from xkit.supervisor import TaskCompletion
from xtest.harness.runner.artifact import ArtifactGroupResult
from xtest.harness.runner.pytest_report import PytestCaseReport


class TestRecord(BaseModel):
    """Validated, immutable test-evidence record."""

    __test__: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TestRunManifest(TestRecord):
    """Original selection and expected cases, written before task admission."""

    run_id: str
    tool_config: ToolConfigRecord
    selected_suites: tuple[str, ...]
    selectors: tuple[str, ...] = ()
    strict_requirements: bool
    tool_software: dict[str, JsonValue] = Field(default_factory=dict)
    python_cases: tuple[str, ...] = ()
    native_cases: tuple[str, ...] = ()
    tasks: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    collection_plan: str | None = None


class TaskReportRecord(TestRecord):
    """One original process-domain classification and available JUnit projection."""

    key: str
    stage: str
    completion: TaskCompletion
    result_code: Literal[0, 1, 2]
    elapsed_seconds: FiniteFloat = Field(ge=0)
    cases: tuple[PytestCaseReport, ...] = ()
    artifact_directory: str

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        for case in self.cases:
            if case.elapsed_seconds is not None and (
                not math.isfinite(case.elapsed_seconds) or case.elapsed_seconds < 0
            ):
                raise ValueError(f"invalid case duration: {case.nodeid}")
        return self


class TestRunResults(TestRecord):
    """Checkpointed outcomes; completion and cleanup are separately established."""

    tasks: tuple[TaskReportRecord, ...] = ()
    stages: dict[str, int] = Field(default_factory=dict)
    groups: tuple[ArtifactGroupResult, ...] = ()
    overall_result_code: int | None = None
    cleanup_verified: bool | None = None
    finished: bool = False
    infrastructure_error: str | None = None


class TestResultWriter:
    """Own atomic test-result checkpoints at the outer execution boundary."""

    __test__ = False

    def __init__(self, directory: Path, manifest: TestRunManifest) -> None:
        self.directory = directory
        self.manifest = manifest
        self.results = TestRunResults()
        write_json(directory / "run.json", manifest.model_dump(mode="json"))
        self.checkpoint()

    def inventory(
        self, *, python_cases: tuple[str, ...], native_cases: tuple[str, ...], tasks: dict[str, tuple[str, ...]]
    ) -> None:
        self.manifest = self.manifest.model_copy(
            update={
                "python_cases": python_cases,
                "native_cases": native_cases,
                "tasks": tasks,
                "collection_plan": "collection/test-plan.json" if python_cases else None,
            }
        )
        write_json(self.directory / "run.json", self.manifest.model_dump(mode="json"))

    def selection(self, selected_suites: tuple[str, ...]) -> None:
        """Retain collection's effective scope before task admission."""
        self.manifest = self.manifest.model_copy(update={"selected_suites": selected_suites})
        write_json(self.directory / "run.json", self.manifest.model_dump(mode="json"))

    def checkpoint(self) -> None:
        # ponytail: whole-run checkpoints; use append-only records if task counts make snapshots expensive.
        write_json(self.directory / "results.json", self.results.model_dump(mode="json"))

    def task(self, record: TaskReportRecord) -> None:
        if any(task.key == record.key for task in self.results.tasks):
            raise ValueError(f"duplicate terminal task record: {record.key}")
        self.results = self.results.model_copy(update={"tasks": (*self.results.tasks, record)})
        self.checkpoint()

    def stage(self, stage: str, result_code: int) -> None:
        self.results = self.results.model_copy(update={"stages": {**self.results.stages, stage: result_code}})
        self.checkpoint()

    def groups(self, results: Sequence[ArtifactGroupResult]) -> None:
        self.results = self.results.model_copy(update={"groups": tuple(results)})
        self.checkpoint()

    def cleanup(self, verified: bool) -> None:
        self.results = self.results.model_copy(update={"cleanup_verified": verified})
        self.checkpoint()

    def fail(self, error: str) -> None:
        self.results = self.results.model_copy(update={"infrastructure_error": error})
        self.checkpoint()

    def finish(self, result_code: int, *, cleanup_verified: bool | None) -> None:
        self.results = self.results.model_copy(
            update={
                "overall_result_code": result_code,
                "cleanup_verified": cleanup_verified,
                "finished": True,
            }
        )
        self.checkpoint()


class TestRunReport(TestRecord):
    """Original verdicts with current retained-evidence availability, not a rerun."""

    directory: Path
    manifest: TestRunManifest
    results: TestRunResults
    missing_artifacts: tuple[str, ...] = ()

    @classmethod
    def load(cls, run_directory: Path) -> TestRunReport:
        """Read an inactive run under caller-held run-store protection."""

        manifest_path = run_directory / "run.json"
        if not manifest_path.is_file():
            raise ValueError(f"unsupported test report format: no run.json in {run_directory}")
        manifest = TestRunManifest.model_validate_json(manifest_path.read_bytes())
        results_path = run_directory / "results.json"
        results = TestRunResults.model_validate_json(results_path.read_bytes())
        if not (run_directory / ".completed").is_file() or not results.finished or results.overall_result_code is None:
            raise ValueError(f"test run is not sealed: {run_directory.name}")
        missing: list[str] = []
        artifacts = ([manifest.collection_plan] if manifest.collection_plan is not None else []) + [
            f"{task.artifact_directory}/{'ctest.xml' if task.stage == 'cext' else 'pytest.xml'}"
            for task in results.tasks
        ]
        for reference in artifacts:
            path = (run_directory / reference).resolve()
            if not path.is_relative_to(run_directory.resolve()):
                raise ValueError(f"test artifact escapes run directory: {reference}")
            if not path.is_file():
                missing.append(reference)
        known_tasks = set(manifest.tasks)
        known_cases = set((*manifest.python_cases, *manifest.native_cases))
        task_keys = [task.key for task in results.tasks]
        cases = [case.nodeid for task in results.tasks for case in task.cases]
        if len(task_keys) != len(set(task_keys)) or set(task_keys) - known_tasks:
            raise ValueError("test results contain duplicate or undeclared tasks")
        if len(cases) != len(set(cases)) or set(cases) - known_cases:
            raise ValueError("test results contain duplicate or undeclared cases")
        for task in results.tasks:
            if set(case.nodeid for case in task.cases) - set(manifest.tasks[task.key]):
                raise ValueError(f"test results contain cases outside task {task.key}")
        return cls(directory=run_directory, manifest=manifest, results=results, missing_artifacts=tuple(missing))

    def summary(self) -> dict[str, JsonValue]:
        """Project original outcomes and incomplete coverage with explicit counts."""

        counts = Counter(case.status.value for task in self.results.tasks for case in task.cases)
        expected_count = len(self.manifest.python_cases) + len(self.manifest.native_cases)
        return {
            "run_id": self.manifest.run_id,
            "tool_software": self.manifest.tool_software,
            "strict_requirements": self.manifest.strict_requirements,
            "selected_suites": list(self.manifest.selected_suites),
            "original_result_code": self.results.overall_result_code,
            "execution_finished": self.results.finished,
            "evidence_complete": (
                self.results.finished
                and self.results.overall_result_code is not None
                and self.results.cleanup_verified is not None
                and not self.missing_artifacts
                and sum(counts.values()) == expected_count
            ),
            "cleanup_verified": self.results.cleanup_verified,
            "expected_case_count": expected_count,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "unexecuted_or_unavailable": expected_count - sum(counts.values()),
            "missing_artifacts": list(self.missing_artifacts),
            "original_results": self.results.model_dump(mode="json"),
        }


def list_test_artifacts(root: Path) -> tuple[str, ...]:
    """Discover inactive, sealed target-format runs using metadata only."""
    artifacts = []
    store = RunStore(root)
    for entry in store.inactive_runs():
        try:
            with store.read(entry.name) as directory:
                manifest = TestRunManifest.model_validate_json((directory / "run.json").read_bytes())
                results = TestRunResults.model_validate_json((directory / "results.json").read_bytes())
                if (
                    manifest.run_id == directory.name
                    and (directory / ".completed").is_file()
                    and results.finished
                    and results.overall_result_code is not None
                ):
                    artifacts.append(entry.name)
        except (ValueError, FileNotFoundError, BlockingIOError):
            continue
    return tuple(artifacts)


def report_test_runs(inputs: Sequence[Path], *, output: Path | None = None) -> tuple[Path, ...]:
    """Report each exact retained run ID under exclusive run protection.

    Default output belongs to that run. An export root receives the same
    tool/run hierarchy. Regeneration preserves raw evidence and unrelated files.
    """
    if not inputs:
        raise ValueError("supply at least one test artifact ID")
    outputs = []
    for path in dict.fromkeys(path.expanduser().resolve() for path in inputs):
        identity = path.name
        with RunStore(path.parent).read(identity, exclusive=True) as directory:
            report = TestRunReport.load(directory)
            destination = (
                directory / "report"
                if output is None
                else output.expanduser().resolve() / "xtest" / identity / "report"
            )
            if destination.resolve().is_relative_to(directory) and destination.resolve() != directory / "report":
                raise ValueError("test report export overlaps source evidence")
            destination.mkdir(parents=True, exist_ok=True)
            summary = report.summary()
            paragraphs = [
                "# CrossPool Test Report\n",
                f"## {identity}\n",
                f"Original result: {report.results.overall_result_code}; "
                f"strict requirements: {report.manifest.strict_requirements}; "
                f"cleanup verified: {report.results.cleanup_verified}.\n",
                f"Passed: {summary['passed']}; failed: {summary['failed']}; skipped: {summary['skipped']}; "
                f"unexecuted or unavailable: {summary['unexecuted_or_unavailable']}.\n",
            ]
            if not summary["evidence_complete"]:
                paragraphs.append("Evidence is incomplete; this report does not establish acceptance.\n")
            if report.results.infrastructure_error is not None:
                paragraphs.append(f"Infrastructure failure: {report.results.infrastructure_error}\n")
            for stage, result_code in report.results.stages.items():
                paragraphs.append(f"Stage `{stage}`: result {result_code}.\n")
            for task in report.results.tasks:
                paragraphs.append(f"### {task.key}\n")
                paragraphs.append(f"Result: {task.result_code}; elapsed: {task.elapsed_seconds:.3f} s.\n")
                for case in task.cases:
                    paragraphs.append(
                        f"- `{case.nodeid}`: {case.status.value}; duration: {case.elapsed_seconds}; {case.detail or ''}"
                    )
                paragraphs.append(f"\nArtifacts: [{task.artifact_directory}]({directory / task.artifact_directory}).\n")
            for group in report.results.groups:
                paragraphs.append(f"\nGroup `{group.name}`: result {group.result_code}; {group.detail or ''}.\n")
            write_json(destination / "summary.json", summary)
            (destination / "report.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
            outputs.append(destination)
    return tuple(outputs)
