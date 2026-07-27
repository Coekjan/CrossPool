"""Canonical test-suite command-line behavior."""

from __future__ import annotations

import pytest

import tests.__main__


def test_main_rejects_duplicate_suites(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject duplicate stage selection instead of silently rewriting it."""

    with pytest.raises(SystemExit) as exc_info:
        tests.__main__.main(["--suite", "unit", "--suite", "unit"])

    assert exc_info.value.code == 2
    assert "--suite cannot select the same suite more than once" in capsys.readouterr().err
