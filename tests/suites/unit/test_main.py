"""Canonical test-suite command-line behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tests.cli
import tests.harness.runner.plan


def test_main_rejects_duplicate_suites(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject duplicate stage selection instead of silently rewriting it."""

    with pytest.raises(SystemExit) as exc_info:
        tests.cli.main(["run", "--suite", "unit", "--suite", "unit"])

    assert exc_info.value.code == 2
    assert "--suite cannot select the same suite more than once" in capsys.readouterr().err


def test_main_rejects_unknown_model_suite(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        tests.cli.main(["run", "--suite", "Qwen/Unknown-Model"])

    assert exc_info.value.code == 2
    assert "unknown model suite: Qwen/Unknown-Model" in capsys.readouterr().err


def test_model_suite_collects_only_its_literal_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = "tests/suites/models/Qwen/Qwen3-0.6B"
    collected_selectors: list[tuple[str, ...]] = []
    collected = tests.harness.runner.plan.CollectedTestCase(
        path=f"{model_directory}/test_sglang_model_qualification.py",
        nodeid=f"{model_directory}/test_sglang_model_qualification.py::test_ffn_numerical[example]",
        stage=tests.harness.runner.plan.TestStage.MODELS,
        requirements=tests.harness.runner.plan.TestRequirements(0, False, False, ()),
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
        ) -> None:
            del repository_root, run_directory, strict_requirements
            collected_selectors.append(selectors)

        def collect(self) -> tests.harness.runner.plan.TestPlan:
            return tests.harness.runner.plan.TestPlan((collected,))

    class FakeSuiteRunner:
        resources_releasable = True

        def __init__(self, plan: tests.harness.runner.plan.TestPlan, **kwargs: object) -> None:
            assert plan.cases == (collected,)

        def request_stop(self, signal_number: int, frame: object) -> None:
            del signal_number, frame

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tests.cli, "CollectionWorker", FakeCollectionWorker)
    monkeypatch.setattr(tests.cli, "SuiteRunner", FakeSuiteRunner)

    assert tests.cli.execute_test_run(("Qwen/Qwen3-0.6B",), (), strict_requirements=False, run_directory=tmp_path) == 0
    assert collected_selectors == [(model_directory,)]
    assert "models" not in tests.cli.SUITE_ORDER


def test_integration_selects_engine_files_without_dropping_neutral_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter before collection while retaining engine-neutral suite coverage."""

    paths = (
        "tests/suites/integration/native/test_runtime.py",
        "tests/suites/integration/devkit/sglang/test_graph.py",
        "tests/suites/integration/vllm/test_plugin.py",
        "tests/suites/e2e/native/test_e2e_control_plane.py",
        "tests/suites/e2e/sglang/test_e2e_model_serving.py",
        "tests/suites/e2e/vllm/test_e2e_model_serving.py",
        "tests/suites/models/Qwen/Qwen3-0.6B/test_sglang_model_qualification.py",
        "tests/suites/models/Qwen/Qwen3-0.6B/test_vllm_model_qualification.py",
        "tests/suites/models/Qwen/Qwen3-14B/test_sglang_model_qualification.py",
    )
    for path in paths:
        test_file = tmp_path / path
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()
    monkeypatch.setattr(tests.cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(tests.cli, "INTEGRATIONS", ("sglang", "vllm"))

    selected = tuple(
        path
        for suite in ("integration", "e2e", "Qwen/Qwen3-0.6B")
        for path in tests.cli.suite_selectors(
            suite,
            model_suites=("Qwen/Qwen3-0.6B",),
            integration="sglang",
        )
    )
    assert tuple(sorted(selected)) == tuple(
        sorted(
            path for path in paths if "/vllm/" not in path and "test_vllm_" not in path and "/Qwen3-14B/" not in path
        )
    )


def test_clean_defaults_to_twenty_retained_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the documented retention default only for explicit cleanup."""

    monkeypatch.setattr(tests.cli, "REPOSITORY_ROOT", tmp_path)
    result_root = tmp_path / ".xpool-cache" / "test-runs"
    for index in range(21):
        entry = result_root / f"unrecognized-{index:02d}"
        entry.mkdir(parents=True)
        os.utime(entry, ns=(index + 1, index + 1))

    assert tests.cli.main(["clean"]) == 0
    assert len(tuple(path for path in result_root.iterdir() if path.name != ".cleanup.lock")) == 20


def test_clean_applies_explicit_dry_run_and_all(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests.cli, "REPOSITORY_ROOT", tmp_path)
    result_root = tmp_path / ".xpool-cache" / "test-runs"
    entries = tuple(result_root / f"unrecognized-{index}" for index in range(3))
    for entry in entries:
        entry.mkdir(parents=True)

    assert tests.cli.main(["clean", "--keep", "1", "--dry-run"]) == 0
    assert all(entry.is_dir() for entry in entries)
    assert tests.cli.main(["clean", "--all"]) == 0
    assert tuple(path for path in result_root.iterdir() if path.name != ".cleanup.lock") == ()


def test_clean_rejects_nonpositive_keep_count() -> None:
    with pytest.raises(SystemExit) as exc_info:
        tests.cli.main(["clean", "--keep", "0"])

    assert exc_info.value.code == 2
