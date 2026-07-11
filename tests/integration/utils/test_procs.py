from __future__ import annotations

import subprocess
import sys
import time

import pytest

from xpool.utils.procs import ProcUniqId


def test_process_identity_treats_unreaped_zombie_as_not_alive() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.1)"],
    )
    process_id = ProcUniqId(process.pid)
    try:
        time.sleep(0.2)
        assert not process_id.is_alive()
    finally:
        process.wait(timeout=5.0)


def test_pid_liveness_requires_matching_create_time(monkeypatch: pytest.MonkeyPatch) -> None:
    proc_id = ProcUniqId.current()

    assert proc_id.is_alive() is True

    class ReusedPidProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def create_time(self) -> float:
            return proc_id.create_time + 1

    monkeypatch.setattr("xpool.utils.procs.psutil.Process", ReusedPidProcess)

    assert proc_id.is_alive() is False
