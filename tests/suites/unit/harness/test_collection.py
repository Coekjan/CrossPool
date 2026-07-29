from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import tests.harness.runner.collection
import tests.harness.runner.plan


def test_collection_worker_runs_isolated_pytest_and_reads_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    run_directory = tmp_path / "run"
    expected = tests.harness.runner.plan.TestPlan((unit_case(),))
    observed_command: list[str] = []
    observed_cwd: list[Path] = []
    observed_environment: dict[str, str] = {}

    def run_worker(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        check: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
        observed_cwd.append(cwd)
        observed_environment.update(env)
        output = next(
            argument.removeprefix("--xpool-test-plan=")
            for argument in command
            if argument.startswith("--xpool-test-plan=")
        )
        expected.write(Path(output))
        assert capture_output and not check and text
        return subprocess.CompletedProcess(command, 0, "collected\n", "")

    monkeypatch.setattr(tests.harness.runner.collection.subprocess, "run", run_worker)

    actual = tests.harness.runner.collection.CollectionWorker(
        repository_root=repository_root,
        run_directory=run_directory,
        selectors=("tests/suites/unit", "-k", "alpha"),
        strict_requirements=True,
    ).collect()

    assert actual == expected
    assert observed_cwd == [repository_root]
    assert observed_command == [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "tests/suites/unit",
        "-k",
        "alpha",
        f"--xpool-test-plan={run_directory / 'test-plan.json'}",
        "--strict-requirements",
    ]
    assert observed_environment["PYTHONPYCACHEPREFIX"] == str(repository_root / ".xpool-cache" / "pycache")
    assert (run_directory / "collection.log").read_text(encoding="utf-8") == "collected\n"


def test_collection_worker_preserves_failure_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    run_directory = tmp_path / "run"
    monkeypatch.setattr(
        tests.harness.runner.collection.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 4, "partial\n", "collection failed\n"),
    )

    with pytest.raises(tests.harness.runner.collection.CollectionFailure, match="exited with code 4"):
        tests.harness.runner.collection.CollectionWorker(repository_root, run_directory, (), False).collect()

    assert (run_directory / "collection.log").read_text(encoding="utf-8") == "partial\ncollection failed\n"


def unit_case() -> tests.harness.runner.plan.CollectedTestCase:
    return tests.harness.runner.plan.CollectedTestCase(
        path="tests/suites/unit/test_example.py",
        nodeid="tests/suites/unit/test_example.py::test_example",
        stage=tests.harness.runner.plan.TestStage.UNIT,
        requirements=tests.harness.runner.plan.TestRequirements(0, False, False, ()),
        estimated_duration_seconds=None,
        timeout_seconds=10,
        artifact_group=None,
    )
