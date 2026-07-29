"""Pinned SGLang orderly-departure hook contracts."""

from __future__ import annotations

import os
import signal
from collections.abc import Callable, Iterator
from types import FrameType, SimpleNamespace
from typing import cast

import pytest
import sglang.cli.serve
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import SignalHandler
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import xpool.integrations.sglang.plugin
from xpool.integrations.sglang.adapter import XpoolModelRuntime
from xpool.integrations.sglang.plugin import (
    KILL_PROCESS_TREE,
    SchedulerShutdownFailure,
    after_sigterm_handler,
    around_kill_process_tree,
    around_scheduler_run_event_loop,
)


@pytest.fixture
def reset_orderly_shutdown_marker() -> Iterator[None]:
    xpool.integrations.sglang.plugin.orderly_shutdown_requested.clear()
    yield
    xpool.integrations.sglang.plugin.orderly_shutdown_requested.clear()


pytestmark = pytest.mark.usefixtures(reset_orderly_shutdown_marker.__name__)


def test_parent_sigterm_marks_orderly_shutdown() -> None:
    handler = cast(SignalHandler, SimpleNamespace())

    after_sigterm_handler(None, handler, signal.SIGTERM, None)

    assert xpool.integrations.sglang.plugin.orderly_shutdown_requested.is_set()


def test_scheduler_sigterm_synchronizes_then_detaches(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    model_runner = SimpleNamespace(device=3)
    scheduler = cast(Scheduler, SimpleNamespace(tp_worker=SimpleNamespace(model_runner=model_runner)))

    def run(scheduler: Scheduler) -> None:
        events.append("run")
        handler = cast(Callable[[int, FrameType | None], object], signal.getsignal(signal.SIGTERM))
        handler(signal.SIGTERM, None)

    def synchronize(device: object) -> None:
        assert device == 3
        events.append("synchronize")

    def detach(runner: object) -> None:
        assert runner is model_runner
        events.append("detach")
        handler = cast(Callable[[int, FrameType | None], object], signal.getsignal(signal.SIGTERM))
        handler(signal.SIGTERM, None)

    runtime = SimpleNamespace(detach=detach)
    monkeypatch.setattr(xpool.integrations.sglang.plugin.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(XpoolModelRuntime, "require", staticmethod(lambda runner: runtime))

    around_scheduler_run_event_loop(run, scheduler)

    assert events == ["run", "synchronize", "detach"]


def test_scheduler_normal_return_does_not_detach(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = cast(Scheduler, SimpleNamespace())
    monkeypatch.setattr(
        XpoolModelRuntime,
        "require",
        staticmethod(lambda runner: pytest.fail("normal event-loop return must not detach")),
    )

    around_scheduler_run_event_loop(lambda scheduler: None, scheduler)


def test_scheduler_cleanup_failure_stays_outside_sglang_exception_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = SimpleNamespace(device=3)
    scheduler = cast(Scheduler, SimpleNamespace(tp_worker=SimpleNamespace(model_runner=model_runner)))

    def run(scheduler: Scheduler) -> None:
        handler = cast(Callable[[int, FrameType | None], object], signal.getsignal(signal.SIGTERM))
        handler(signal.SIGTERM, None)

    def synchronize(device: object) -> None:
        raise RuntimeError("synchronize failed")

    monkeypatch.setattr(xpool.integrations.sglang.plugin.torch.cuda, "synchronize", synchronize)

    with pytest.raises(SchedulerShutdownFailure):
        around_scheduler_run_event_loop(run, scheduler)


def test_parent_waits_for_all_schedulers_before_original_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    schedulers = [FakeSchedulerProcess(101, events), FakeSchedulerProcess(102, events)]
    xpool.integrations.sglang.plugin.orderly_shutdown_requested.set()
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "collect_scheduler_processes", lambda: schedulers)

    def wait_procs(processes: list[FakeSchedulerProcess], timeout: float) -> tuple[list[FakeSchedulerProcess], list]:
        assert processes == schedulers
        assert timeout == xpool.integrations.sglang.plugin.SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS
        events.append("wait")
        return processes, []

    monkeypatch.setattr(xpool.integrations.sglang.plugin.psutil, "wait_procs", wait_procs)

    def original(
        parent_pid: int | None, include_parent: bool, skip_pid: int | None, wait_timeout: float | None
    ) -> None:
        assert (parent_pid, include_parent, skip_pid, wait_timeout) == (os.getpid(), False, 314, 4.0)
        events.append("original")

    hooks = HookRegistry._hooks.copy()
    patched = HookRegistry._patched.copy()
    HookRegistry.reset()
    monkeypatch.setattr(sglang.cli.serve, "kill_process_tree", original)
    try:
        HookRegistry.register(KILL_PROCESS_TREE, around_kill_process_tree, HookType.AROUND)
        HookRegistry.apply_hooks()
        sglang.cli.serve.kill_process_tree(os.getpid(), False, 314, 4.0)
    finally:
        HookRegistry.reset()
        HookRegistry._hooks.update(hooks)
        HookRegistry._patched.update(patched)

    assert events == ["signal:101", "signal:102", "wait", "original"]


def test_parent_cleanup_runs_after_orderly_drain_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    xpool.integrations.sglang.plugin.orderly_shutdown_requested.set()

    def fail_discovery() -> list[FakeSchedulerProcess]:
        raise RuntimeError("scheduler discovery failed")

    monkeypatch.setattr(xpool.integrations.sglang.plugin, "collect_scheduler_processes", fail_discovery)

    around_kill_process_tree(lambda *args: events.append("original"), os.getpid(), False)

    assert events == ["original"]


def test_non_orderly_cleanup_preserves_sglang_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin,
        "collect_scheduler_processes",
        lambda: pytest.fail("forced cleanup must not wait for schedulers"),
    )

    around_kill_process_tree(lambda *args: events.append("original"), os.getpid())

    assert events == ["original"]


class FakeSchedulerProcess:
    """Minimal psutil process surface used by the parent hook."""

    def __init__(self, pid: int, events: list[str]) -> None:
        self.pid = pid
        self.returncode = 0
        self.events = events

    def send_signal(self, signum: int) -> None:
        assert signum == signal.SIGTERM
        self.events.append(f"signal:{self.pid}")
