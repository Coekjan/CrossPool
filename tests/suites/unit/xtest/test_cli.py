"""Canonical test-suite command-line behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import xtest.cli


def test_main_rejects_duplicate_suites(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject duplicate stage selection instead of silently rewriting it."""

    with pytest.raises(SystemExit) as exc_info:
        xtest.cli.main(["run", "--suite", "unit", "--suite", "unit"])

    assert exc_info.value.code == 2
    assert "--suite cannot select the same suite more than once" in capsys.readouterr().err


def test_main_rejects_unknown_model_suite(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        xtest.cli.main(["run", "--suite", "Qwen/Unknown-Model"])

    assert exc_info.value.code == 2
    assert "unknown model suite: Qwen/Unknown-Model" in capsys.readouterr().err


def test_clean_rejects_nonpositive_keep_count() -> None:
    with pytest.raises(SystemExit) as exc_info:
        xtest.cli.main(["clean", "--keep", "0"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("command", ["list", "run"])
def test_source_commands_reject_nonrepository_directory(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    assert xtest.cli.main([command]) == 2
    assert "repository root" in capsys.readouterr().err
