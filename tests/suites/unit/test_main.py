"""Canonical test-suite command-line behavior."""

from __future__ import annotations

import os

import pytest

import tests.cli


def test_main_rejects_duplicate_suites(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject duplicate stage selection instead of silently rewriting it."""

    with pytest.raises(SystemExit) as exc_info:
        tests.cli.main(["run", "--suite", "unit", "--suite", "unit"])

    assert exc_info.value.code == 2
    assert "--suite cannot select the same suite more than once" in capsys.readouterr().err


def test_clean_defaults_to_twenty_retained_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the documented retention default only for explicit cleanup."""

    monkeypatch.setattr(tests.cli, "REPOSITORY_ROOT", tmp_path)
    result_root = tmp_path / ".xpool-cache" / "test-runs"
    for index in range(21):
        entry = result_root / f"legacy-{index:02d}"
        entry.mkdir(parents=True)
        os.utime(entry, ns=(index + 1, index + 1))

    assert tests.cli.main(["clean"]) == 0
    assert len(tuple(path for path in result_root.iterdir() if path.name != ".cleanup.lock")) == 20


def test_clean_applies_explicit_dry_run_and_all(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tests.cli, "REPOSITORY_ROOT", tmp_path)
    result_root = tmp_path / ".xpool-cache" / "test-runs"
    entries = tuple(result_root / f"legacy-{index}" for index in range(3))
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
