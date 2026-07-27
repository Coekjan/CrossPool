from __future__ import annotations

import os
import subprocess
import time

import pytest

from tests.harness.process_probe import command
from xpool.utils.procs import PROCESS_KILL_WAIT_S, ProcUniqId


def test_process_identity_treats_unreaped_zombie_as_not_alive() -> None:
    process = subprocess.Popen(
        command("sleep", seconds=0.1),
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


def test_kill_tree_directly_kills_captured_root_and_child_within_bound() -> None:
    process = subprocess.Popen(
        command("spawn-child", seconds=60),
        stdout=subprocess.PIPE,
        text=True,
    )
    root = ProcUniqId(process.pid)
    child: ProcUniqId | None = None
    try:
        assert process.stdout is not None
        child = ProcUniqId(int(process.stdout.readline()))
        started_at = time.monotonic()

        assert root.kill_tree()

        process.wait(timeout=5.0)
        assert time.monotonic() - started_at <= PROCESS_KILL_WAIT_S + 1.0
        assert not root.is_alive()
        assert not child.is_alive()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        if child is not None and child.is_alive():
            child.kill_tree()


@pytest.mark.parametrize("title", ["xpool::daemon", "xpool::atnagent", "xpool::ffnagent"])
def test_process_title_is_visible_without_hiding_environment(title: str) -> None:
    probe_name = "XPOOL_PROCESS_TITLE_ENVIRONMENT_PROBE"
    environment = os.environ.copy()
    environment[probe_name] = title
    process = subprocess.Popen(
        [*command("title"), "--title", title],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"

        proc_root = f"/proc/{process.pid}"
        with open(f"{proc_root}/comm", encoding="utf-8") as stream:
            assert stream.read().strip() == title
        with open(f"{proc_root}/cmdline", "rb") as stream:
            assert stream.read().split(b"\0", maxsplit=1)[0] == title.encode()
        with open(f"{proc_root}/environ", "rb") as stream:
            assert f"{probe_name}={title}".encode() in stream.read().split(b"\0")
    finally:
        if process.poll() is None:
            process.communicate(input="\n", timeout=5.0)
        else:
            process.wait(timeout=5.0)
