from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import cast

import pytest

import tests.harness.runner.process
import tests.harness.runner.supervisor
from tests.harness.runner.process import (
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
    monkeypatch.setattr(tests.harness.runner.process, "wait_for_process_group", lambda process, timeout: next(waits))

    OwnedProcessGroup(name="fake", process=cast(subprocess.Popen[str], process)).terminate()

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_terminate_reports_group_that_survives_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 12345

    process = FakeProcess()
    monkeypatch.setattr(os, "killpg", lambda pid, signal_number: None)
    monkeypatch.setattr(tests.harness.runner.process, "wait_for_process_group", lambda process, timeout: False)
    owner = OwnedProcessGroup(name="fake", process=cast(subprocess.Popen[str], process))

    with pytest.raises(RuntimeError, match="retained live members after SIGKILL"):
        owner.terminate()


def test_collect_process_output_after_timeout_closes_stuck_pipes() -> None:
    class FakePipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakePipe()
            self.stderr = FakePipe()

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            assert timeout is not None
            raise subprocess.TimeoutExpired(
                cmd="probe",
                timeout=timeout,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

    process = FakeProcess()
    owner = OwnedProcess(name="fake", process=cast(subprocess.Popen[str], process))

    stdout, stderr, status = owner.collect_output_after_timeout()

    assert stdout == "partial stdout"
    assert stderr == "partial stderr"
    assert status == "stdout/stderr pipes did not close after cleanup"
    assert process.stdout.closed
    assert process.stderr.closed


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
