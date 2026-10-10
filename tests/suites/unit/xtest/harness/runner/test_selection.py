"""Canonical test-suite command-line behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import xtest.harness.runner.selection


def test_select_model_suite_by_canonical_identity(tmp_path: Path) -> None:
    (tmp_path / "tests/suites/models/Qwen/Qwen3-0.6B").mkdir(parents=True)
    assert xtest.harness.runner.selection.select_suites(tmp_path, ("Qwen/Qwen3-0.6B",)) == ("Qwen/Qwen3-0.6B",)


@pytest.mark.parametrize("name", ["org/bad name", "org/模型", "org/model%2Fname"])
def test_model_suite_directory_requires_canonical_identity(tmp_path: Path, name: str) -> None:
    (tmp_path / "tests/suites/models" / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown model suite"):
        xtest.harness.runner.selection.select_suites(tmp_path, (name,))
