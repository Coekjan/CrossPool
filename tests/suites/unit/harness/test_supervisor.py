from __future__ import annotations

import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import cast

import pytest

import tests.harness.runner.process
import tests.harness.runner.supervisor
from tests.harness.runner.supervisor import (
    SupervisedTaskScope,
    TaskCompletion,
    TaskCompletionKind,
    TaskScopeState,
    drain_subreaper_descendants,
    prepare_task_supervision,
)
from xpool.utils.procs import ProcUniqId


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
    "completion",
    (
        TaskCompletion(TaskCompletionKind.EXITED, 0, None),
        TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "deadline"),
        TaskCompletion(TaskCompletionKind.LEAKED, 1, "descendant"),
        TaskCompletion(TaskCompletionKind.INFRASTRUCTURE_FAILED, None, "internal failure"),
    ),
)
def test_task_completion_accepts_kind_specific_payloads(completion: TaskCompletion) -> None:
    assert isinstance(completion.kind, TaskCompletionKind)


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


def test_prepare_task_supervision_protects_only_warmup_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = cast(ProcUniqId, object())
    helper = cast(ProcUniqId, object())

    class WarmupProcess:
        def join(self, timeout: float) -> None:
            assert timeout > 0

        def is_alive(self) -> bool:
            return False

        def close(self) -> None:
            pass

    class SpawnContext:
        def Process(self, **kwargs: object) -> WarmupProcess:
            return WarmupProcess()

    children = iter(((existing,), (existing, helper)))
    monkeypatch.setattr(tests.harness.runner.supervisor, "protected_subreaper_process_ids", None)
    monkeypatch.setattr(tests.harness.runner.supervisor, "set_child_subreaper", lambda: None)
    monkeypatch.setattr(tests.harness.runner.supervisor, "direct_child_process_ids", lambda: next(children))
    monkeypatch.setattr(tests.harness.runner.supervisor.multiprocessing, "get_context", lambda method: SpawnContext())
    monkeypatch.setattr(tests.harness.runner.supervisor, "start_spawn_process", lambda process: None)

    prepare_task_supervision()

    assert tests.harness.runner.supervisor.protected_subreaper_process_ids == frozenset((helper,))


def test_subreaper_drain_signals_each_identity_once_per_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProcessIdentity:
        def __init__(self) -> None:
            self.signals: list[signal.Signals] = []

        def send_signal(self, signal_number: signal.Signals) -> None:
            self.signals.append(signal_number)

    target = cast(ProcUniqId, ProcessIdentity())
    roots = iter(((target,), (target,), ()))
    clock = iter((0.0, 1.0, 2.0, 11.0, 12.0, 13.0, 21.0))
    monkeypatch.setattr(tests.harness.runner.supervisor, "PROCESS_TERMINATE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(tests.harness.runner.supervisor, "PROCESS_KILL_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(tests.harness.runner.supervisor, "subreaper_direct_roots", lambda excluded: next(roots))
    monkeypatch.setattr(tests.harness.runner.supervisor, "process_tree_ids", lambda process_roots: (target, target))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    drain_subreaper_descendants(frozenset(), context="test descendants")

    assert cast(ProcessIdentity, target).signals == [signal.SIGTERM, signal.SIGKILL]
