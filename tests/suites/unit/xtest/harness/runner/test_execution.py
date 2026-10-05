"""Canonical test-suite command-line behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import xtest.cli
import xtest.harness.runner.execution
import xtest.harness.runner.plan
import xtest.harness.runner.selection
from xkit import ResourceRequirements


def test_model_suite_collects_only_its_literal_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_directory = "tests/suites/models/Qwen/Qwen3-0.6B"
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
        ) -> None:
            del repository_root, run_directory, strict_requirements
            assert catalogue_path == Path.cwd().resolve() / "tests/tests.toml"
            collected_selectors.append(selectors)

        def collect(self) -> xtest.harness.runner.plan.TestPlan:
            return xtest.harness.runner.plan.TestPlan((collected,))

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
    assert collected_selectors == [(model_directory,)]
    assert "models" not in xtest.cli.SUITE_ORDER
