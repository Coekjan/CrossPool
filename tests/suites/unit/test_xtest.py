from __future__ import annotations

from pathlib import Path

import pytest

import tests.cli
import xtest


def test_xtest_forwards_arguments_from_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str] | None] = []
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    monkeypatch.setattr(tests.cli, "main", lambda arguments=None: received.append(arguments) or 7)

    assert xtest.main(["run", "--suite", "unit"]) == 7
    assert received == [["run", "--suite", "unit"]]


def test_xtest_rejects_nonrepository_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert xtest.main(["clean"]) == 2
    assert "repository root" in capsys.readouterr().err
