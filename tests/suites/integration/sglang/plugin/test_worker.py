from __future__ import annotations

import multiprocessing
import signal
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.server_args import PortArgs

from xpool.integrations.sglang import worker
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.sglang import fakes


def test_scheduler_entry_keeps_one_owner_until_confirmed_local_release(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[worker.WorkerLifecycle] = []

    def run(*args: object) -> None:
        lifecycle = worker.WorkerLifecycle.current()
        observed.append(lifecycle)
        lifecycle.loop_returned = True
        lifecycle.normal_release_complete = True

    monkeypatch.setattr(worker.scheduler, "run_scheduler_process", run)
    monkeypatch.setattr(worker, "worker_lifecycle", None)
    reader, writer = multiprocessing.Pipe(duplex=False)
    try:
        worker.run_scheduler_process(fakes.server_args(), PortArgs.__new__(PortArgs), 0, 0, 0, 0, 0, 0, None, writer)
    finally:
        reader.close()
        writer.close()
    assert len(observed) == 1
    assert observed[0].parent.pid > 0
    with pytest.raises(RuntimeError, match="no worker lifecycle"):
        worker.WorkerLifecycle.current()


def test_later_terminal_error_requires_exit_without_replacing_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list[signal.Signals] = []
    lifecycle = worker.WorkerLifecycle(ProcUniqId.current())
    first = RuntimeError("background transport failed")
    terminal = torch.AcceleratorError("terminal device result")
    # Torch's native translator attaches this metadata dynamically.
    setattr(terminal, "error_code", 719)

    def notify(self: ProcUniqId, signum: signal.Signals) -> None:
        notified.append(signum)
        raise OSError("parent notification unavailable")

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(ProcUniqId, "send_signal", notify)
    monkeypatch.setattr(worker.os, "_exit", exit_process)
    lifecycle.fail(first)
    with pytest.raises(SystemExit) as exit_error:
        lifecycle.fail(terminal)
    assert exit_error.value.code == 1
    assert lifecycle.failure is first
    assert notified == [signal.SIGQUIT]


@pytest.mark.parametrize("reported_failure", [False, True])
def test_upstream_return_without_local_release_retains_owner(
    monkeypatch: pytest.MonkeyPatch,
    reported_failure: bool,
) -> None:
    class Retained(Exception):
        pass

    def wait(seconds: float) -> None:
        raise Retained

    signals: list[signal.Signals] = []
    observed: list[worker.WorkerLifecycle] = []
    original = RuntimeError("upstream swallowed scheduler failure")

    def run(*args: object) -> None:
        lifecycle = worker.WorkerLifecycle.current()
        observed.append(lifecycle)
        if reported_failure:
            lifecycle.fail(original)

    monkeypatch.setattr(worker, "worker_lifecycle", None)
    monkeypatch.setattr(worker.scheduler, "run_scheduler_process", run)
    monkeypatch.setattr(ProcUniqId, "send_signal", lambda self, signum: signals.append(signum))
    monkeypatch.setattr(worker, "time", SimpleNamespace(sleep=wait))
    reader, writer = multiprocessing.Pipe(duplex=False)
    with reader, writer, pytest.raises(Retained):
        worker.run_scheduler_process(fakes.server_args(), PortArgs.__new__(PortArgs), 0, 0, 0, 0, 0, 0, None, writer)
    if reported_failure:
        assert observed[0].failure is original
    else:
        assert isinstance(observed[0].failure, RuntimeError)
    assert signals == [signal.SIGQUIT]
