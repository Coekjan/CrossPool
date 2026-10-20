from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

import pytest

from xpool.utils.procs import ProcUniqId
from xpool.utils.sighandler import defer_signal_exceptions, sighandle


@pytest.mark.parametrize("body_failure", [False, True])
def test_signal_exception_waits_for_publication_and_preserves_operation_error(body_failure: bool) -> None:
    events: list[str] = []

    def interrupt(signum: int, frame: FrameType | None) -> None:
        events.append("cancel-requested")
        raise InterruptedError("cancelled")

    with sighandle(signal.SIGTERM, interrupt):
        with pytest.raises(ValueError if body_failure else InterruptedError):
            with defer_signal_exceptions():
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("published")
                if body_failure:
                    raise ValueError("creation failed")
        assert signal.getsignal(signal.SIGTERM) is interrupt
    assert events == ["cancel-requested", "published"]


def test_spawned_child_protects_import_and_waits_for_cooperative_stop(tmp_path: Path) -> None:
    probe = tmp_path / "signal_probe.py"
    gate = tmp_path / "continue-import"
    probe.write_text(
        """import multiprocessing
import os
from pathlib import Path
import sys
import time

from xpool.utils.sighandler import block_signals, initialize_child_signals
import signal

if __name__ == "__mp_main__":
    Path(sys.argv[1] + ".importing").touch()
    while not Path(sys.argv[1]).exists():
        time.sleep(0.01)

def child(connection, mask):
    initialize_child_signals(mask)
    connection.send("ready")
    assert connection.recv() == "stop"
    connection.close()

if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    parent, endpoint = context.Pipe()
    with block_signals({signal.SIGINT, signal.SIGTERM}) as mask:
        process = context.Process(target=child, args=(endpoint, mask))
        process.start()
        endpoint.close()
    print(process.pid, flush=True)
    assert parent.recv() == "ready"
    parent.send("stop")
    process.join(10)
    assert process.exitcode == 0
    print("stopped", flush=True)
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(probe), str(gate)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline())
        deadline = time.monotonic() + 10
        while not gate.with_suffix(".importing").exists():
            assert time.monotonic() < deadline, "spawn child did not reach module import"
            time.sleep(0.01)
        for signum in (signal.SIGINT, signal.SIGTERM):
            os.kill(child_pid, signum)
        gate.touch()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        assert stdout == "stopped\n"
    finally:
        if process.poll() is None:
            ProcUniqId(process.pid).kill_tree()
            process.wait(timeout=5)
