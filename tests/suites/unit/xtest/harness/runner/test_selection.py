"""Canonical test-suite command-line behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import xtest.cli
import xtest.harness.runner.execution
import xtest.harness.runner.plan
import xtest.harness.runner.selection


def test_select_model_suite_by_canonical_identity(tmp_path: Path) -> None:
    (tmp_path / "tests/suites/models/Qwen/Qwen3-0.6B").mkdir(parents=True)
    assert xtest.harness.runner.selection.select_suites(tmp_path, ("Qwen/Qwen3-0.6B",), (), None) == (
        "Qwen/Qwen3-0.6B",
    )


@pytest.mark.parametrize("name", ["org/bad name", "org/模型", "org/model%2Fname"])
def test_model_suite_directory_requires_canonical_identity(tmp_path: Path, name: str) -> None:
    (tmp_path / "tests/suites/models" / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown model suite"):
        xtest.harness.runner.selection.select_suites(tmp_path, (name,), (), None)


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
    monkeypatch.setattr(xtest.harness.runner.selection, "INTEGRATIONS", ("sglang", "vllm"))

    selected = tuple(
        path
        for suite in ("integration", "e2e", "Qwen/Qwen3-0.6B")
        for path in xtest.harness.runner.selection.suite_selectors(
            suite,
            repository_root=tmp_path,
            model_suites=("Qwen/Qwen3-0.6B",),
            integration="sglang",
        )
    )
    assert tuple(sorted(selected)) == tuple(
        sorted(
            path for path in paths if "/vllm/" not in path and "test_vllm_" not in path and "/Qwen3-14B/" not in path
        )
    )
