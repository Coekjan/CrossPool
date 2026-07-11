from __future__ import annotations

import os
import signal
import subprocess
from typing import cast

import pytest

from tests.harness.sglang.process import collect_process_output_after_timeout, terminate_process_group


def test_terminate_process_group_reports_unreaped_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> None:
            assert timeout is not None
            raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)

    process = FakeProcess()
    signals: list[int] = []

    def fake_killpg(pid: int, signal_number: int) -> None:
        assert pid == process.pid
        signals.append(signal_number)

    monkeypatch.setattr(os, "killpg", fake_killpg)

    status = terminate_process_group(cast(subprocess.Popen[str], process))

    assert status == "process group still running after SIGKILL"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


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

    stdout, stderr, status = collect_process_output_after_timeout(cast(subprocess.Popen[str], process))

    assert stdout == "partial stdout"
    assert stderr == "partial stderr"
    assert status == "stdout/stderr pipes did not close after cleanup"
    assert process.stdout.closed
    assert process.stderr.closed
