from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import cast

import pytest

import xkit.process
import xkit.supervisor
from xkit.process import (
    OwnedProcess,
    OwnedProcessGroup,
)


def test_terminate_escalates_and_requires_live_group_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 12345

    process = FakeProcess()
    signals: list[int] = []
    waits = iter((False, True))
    monkeypatch.setattr(os, "killpg", lambda pid, signal_number: signals.append(signal_number))
    monkeypatch.setattr(xkit.process, "wait_for_process_group", lambda process, timeout: next(waits))

    OwnedProcessGroup(name="fake", process=cast(subprocess.Popen[str], process)).terminate()

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_terminate_reports_group_that_survives_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 12345

    process = FakeProcess()
    monkeypatch.setattr(os, "killpg", lambda pid, signal_number: None)
    monkeypatch.setattr(xkit.process, "wait_for_process_group", lambda process, timeout: False)
    owner = OwnedProcessGroup(name="fake", process=cast(subprocess.Popen[str], process))

    with pytest.raises(RuntimeError, match="retained live members after SIGKILL"):
        owner.terminate()


def test_tail_remains_available_after_log_owner_closes(tmp_path: Path) -> None:
    class FakeProcess:
        stdout = None
        stderr = None

    log_path = tmp_path / "process.log"
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("retained diagnostic")
    owner = OwnedProcess(
        name="fake",
        process=cast(subprocess.Popen[str], FakeProcess()),
        log_path=log_path,
        log_file=log_file,
    )

    owner.close()

    assert owner.tail() == "retained diagnostic"
