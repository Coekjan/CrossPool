from __future__ import annotations

import signal
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import SimpleNamespace
from typing import cast

import psutil
import pytest

import xkit.process
import xkit.supervisor
from xkit.supervisor import (
    SupervisedTaskScope,
    TaskCompletion,
    TaskCompletionKind,
    TaskProtectionState,
    TaskScopeState,
    drain_subreaper_descendants,
)


@pytest.mark.parametrize("resource_state", (TaskProtectionState.ACTIVE, TaskProtectionState.RETIRING))
def test_supervisor_loss_retains_resource_protection(resource_state: TaskProtectionState) -> None:
    class DeadSupervisor:
        def is_alive(self) -> bool:
            return False

    class RootChannel:
        def poll(self, timeout: float = 0.0) -> bool:
            raise AssertionError("lost supervision cannot acknowledge resource cleanup")

    scope = SupervisedTaskScope(
        name="supervisor-lost",
        supervisor=cast(BaseProcess, DeadSupervisor()),
        connection=cast(Connection, RootChannel()),
        root_connection=cast(Connection, RootChannel()),
        state=TaskScopeState.FAILED,
        resource_state=resource_state,
    )

    scope.service_root_control()

    assert scope.is_protected
    with pytest.raises(RuntimeError, match="cannot close unproven"):
        scope.close()


def test_scope_cancellation_accepts_completion_after_broken_pipe() -> None:
    completion = TaskCompletion(TaskCompletionKind.EXITED, 0, None)

    class CompletingConnection:
        completed = False

        def poll(self, timeout: float = 0.0) -> bool:
            return self.completed

        def recv(self) -> TaskCompletion:
            return completion

        def send(self, message: object) -> None:
            self.completed = True
            raise BrokenPipeError

    class LiveSupervisor:
        def is_alive(self) -> bool:
            return True

    scope = SupervisedTaskScope(
        name="completed-during-cancel",
        supervisor=cast(BaseProcess, LiveSupervisor()),
        connection=cast(Connection, CompletingConnection()),
        state=TaskScopeState.RUNNING,
    )

    SupervisedTaskScope.terminate_all((scope,))

    assert scope.completion == completion
    assert scope.state is TaskScopeState.COMPLETED


@pytest.mark.parametrize(
    ("kind", "returncode", "diagnostics"),
    (
        (TaskCompletionKind.EXITED, None, None),
        (TaskCompletionKind.EXITED, 0, "unexpected"),
        (TaskCompletionKind.TIMED_OUT, 1, "deadline"),
        (TaskCompletionKind.LEAKED, None, "descendant"),
        (TaskCompletionKind.LEAKED, 1, None),
        (TaskCompletionKind.INFRASTRUCTURE_FAILED, 1, "internal failure"),
        (TaskCompletionKind.INFRASTRUCTURE_FAILED, None, None),
    ),
)
def test_task_completion_rejects_inconsistent_payloads(
    kind: TaskCompletionKind,
    returncode: int | None,
    diagnostics: str | None,
) -> None:
    with pytest.raises(ValueError):
        TaskCompletion(kind, returncode, diagnostics)


def test_scope_rejects_poll_and_close_before_empty_proof() -> None:
    class LiveSupervisor:
        def is_alive(self) -> bool:
            return True

    class EmptyConnection:
        def poll(self, timeout: float = 0.0) -> bool:
            return False

    scope = SupervisedTaskScope(
        name="starting",
        supervisor=cast(BaseProcess, LiveSupervisor()),
        connection=cast(Connection, EmptyConnection()),
    )

    with pytest.raises(RuntimeError, match="not running"):
        scope.poll()
    with pytest.raises(RuntimeError, match="cannot close unproven"):
        scope.close()


def test_subreaper_drain_signals_each_identity_once_per_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProcessIdentity:
        def __init__(self) -> None:
            self.signals: list[signal.Signals] = []

        def send_signal(self, signal_number: signal.Signals) -> None:
            self.signals.append(signal_number)

    target = ProcessIdentity()
    now = 0.0
    polls = {"term": 0, "kill": 0}

    def roots(excluded: frozenset[object]) -> tuple[ProcessIdentity, ...]:
        phase = "term" if now < 1.0 else "kill"
        polls[phase] += 1
        return (target,) if now < 2.0 else ()

    def advance(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(xkit.supervisor, "PROCESS_TERMINATE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(xkit.supervisor, "PROCESS_KILL_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(xkit.supervisor, "subreaper_direct_roots", roots)
    monkeypatch.setattr(xkit.supervisor, "process_tree_ids", lambda process_roots: (target, target))
    monkeypatch.setattr(xkit.supervisor, "time", SimpleNamespace(monotonic=lambda: now, sleep=advance))

    drain_subreaper_descendants(frozenset(), context="test descendants")

    assert target.signals == [signal.SIGTERM, signal.SIGKILL]
    assert polls["term"] >= 2 and polls["kill"] >= 2


def test_managed_fallback_keeps_expired_clock_until_domain_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProcessIdentity:
        def __init__(self) -> None:
            self.signals: list[signal.Signals] = []

        def send_signal(self, signum: signal.Signals) -> None:
            self.signals.append(signum)

    target = ProcessIdentity()
    now = 10.0
    observations = 0

    def roots(excluded: frozenset[object]) -> tuple[ProcessIdentity, ...]:
        nonlocal observations
        observations += 1
        if observations == 1:
            raise psutil.AccessDenied(123)
        return (target,) if now < 10.2 else ()

    def advance(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(xkit.supervisor, "subreaper_direct_roots", roots)
    monkeypatch.setattr(xkit.supervisor, "process_tree_ids", lambda process_roots: (target,))
    monkeypatch.setattr(xkit.supervisor, "time", SimpleNamespace(monotonic=lambda: now, sleep=advance))

    drain_subreaper_descendants(frozenset(), context="managed descendants", cleanup_deadline=9.0)

    assert not target.signals
    assert observations >= 3
    assert now >= 10.2
