"""Supervised task-scope lifecycle and descendant-tree ownership for tests."""

from __future__ import annotations

import ctypes
import logging
import math
import multiprocessing
import multiprocessing.process
import multiprocessing.reduction
import multiprocessing.resource_tracker
import os
import signal
import subprocess
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Self, TextIO, cast

import psutil

from xkit.child import start_spawn_process
from xkit.process import (
    LOG_TAIL_CHARS,
    PROCESS_KILL_TIMEOUT_SECONDS,
    PROCESS_POLL_INTERVAL_SECONDS,
    PROCESS_TERMINATE_TIMEOUT_SECONDS,
)
from xkit.task import (
    TASK_ROOT_CONTROL_FD_ENV,
    TaskCancellation,
    TaskExecutionWindow,
    TaskResourceEvent,
    TaskRootAcknowledged,
    TaskRootUpdate,
)
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S
from xpool.utils.procs import ProcUniqId

TASK_NATURAL_DRAIN_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_START_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS = 5.0
PR_SET_CHILD_SUBREAPER = 36
task_supervision_lock = threading.Lock()
protected_subreaper_process_ids: frozenset[ProcUniqId] | None = None
logger = logging.getLogger(__name__)


class TaskCompletionKind(StrEnum):
    """Proven-empty terminal outcome of one supervised device task."""

    EXITED = "exited"
    TIMED_OUT = "timed_out"
    LEAKED = "leaked"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class TaskScopeState(StrEnum):
    """Parent-side lifecycle of one supervised task scope."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    DRAINED = "drained"
    FAILED = "failed"
    CLOSED = "closed"


class TaskProtectionState(StrEnum):
    """Protection retained across its commits and supervisor failure."""

    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    RETIRING = "retiring"
    CLEANED = "cleaned"


@dataclass(frozen=True, slots=True)
class TaskCompletion:
    """Terminal task result published only after its descendant scope is empty."""

    kind: TaskCompletionKind
    returncode: int | None
    diagnostics: str | None

    def __post_init__(self) -> None:
        if self.kind is TaskCompletionKind.EXITED:
            if self.returncode is None or self.diagnostics is not None:
                raise ValueError("EXITED completion requires a return code and no diagnostics")
        elif self.kind is TaskCompletionKind.TIMED_OUT:
            if self.returncode is not None:
                raise ValueError("TIMED_OUT completion cannot carry a return code")
        elif self.kind is TaskCompletionKind.LEAKED:
            if self.returncode is None or not self.diagnostics:
                raise ValueError("LEAKED completion requires a return code and diagnostics")
        elif self.returncode is not None or not self.diagnostics:
            raise ValueError("INFRASTRUCTURE_FAILED completion requires diagnostics and no return code")


class TaskScopeFailure(RuntimeError):
    """Raised when a supervisor cannot prove that its complete task scope is empty."""


class TaskStartFailure(RuntimeError):
    """Raised when task startup failed after rollback proved the scope empty."""


class TaskSupervisionFailure(RuntimeError):
    """Raised when normal supervision failed and runner fallback is required."""


@dataclass(frozen=True, slots=True)
class TaskSupervisorStarted:
    """Launched root identity retained before task-control activation."""

    root: ProcUniqId


@dataclass(frozen=True, slots=True)
class TerminateTaskScope:
    """Infrastructure cancellation command with shared absolute deadlines."""

    term_deadline: float
    kill_deadline: float
    cleanup_deadline: float | None = None


@dataclass(frozen=True, slots=True)
class TaskScopeDrained:
    """Acknowledgement that infrastructure cancellation proved the scope empty."""


@dataclass(frozen=True, slots=True)
class TaskSupervisorFailed:
    """Typed supervisor failure that cannot be represented as task completion."""

    diagnostics: str


TaskSupervisorMessage = (
    TaskSupervisorStarted
    | TaskCompletion
    | TaskScopeDrained
    | TaskSupervisorFailed
    | TaskRootAcknowledged
    | TaskCancellation
)


@dataclass(slots=True)
class TaskProtection:
    """Supervisor-local commit and first retirement clock for one exact root."""

    root: ProcUniqId
    event: TaskResourceEvent | None = None
    cleanup_deadline: float | None = None

    def commit(self, update: TaskRootUpdate, connection: Connection) -> None:
        """Commit before acknowledging the runner's protection handoff."""

        if update.root != self.root:
            raise TaskScopeFailure("task protection root identity does not match")
        if update.event is TaskResourceEvent.ACTIVE:
            if self.event is not None:
                raise TaskScopeFailure("task protection cannot activate twice")
        elif self.event is not TaskResourceEvent.ACTIVE:
            raise TaskScopeFailure("task cleanup proof requires prior protection")
        self.event = update.event
        connection.send(TaskRootAcknowledged(update.event))

    def retire(self, deadline: float | None = None) -> TaskCancellation:
        """Retain the first absolute budget for cooperative owner retirement."""

        if self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S if deadline is None else deadline
        elif deadline is not None:
            self.cleanup_deadline = min(self.cleanup_deadline, deadline)
        return TaskCancellation(self.cleanup_deadline)


@dataclass(slots=True)
class SupervisedTaskScope:
    """Own one child-subreaper supervisor and its complete descendant domain."""

    name: str
    supervisor: multiprocessing.process.BaseProcess
    connection: Connection
    root_connection: Connection | None = None
    root: ProcUniqId | None = None
    state: TaskScopeState = TaskScopeState.STARTING
    completion: TaskCompletion | None = None
    resource_state: TaskProtectionState = TaskProtectionState.INACTIVE
    cleanup_deadline: float | None = None

    @property
    def is_protected(self) -> bool:
        """Return whether resource proof still prohibits generic domain signals."""

        return self.resource_state in (
            TaskProtectionState.ACTIVATING,
            TaskProtectionState.ACTIVE,
            TaskProtectionState.RETIRING,
        )

    def service_root_control(self) -> None:
        """Relay root updates while the supervision owner remains available."""

        connection = self.root_connection
        if connection is None or (self.is_protected and not self.supervisor.is_alive()):
            return
        while connection.poll():
            try:
                update = connection.recv()
            except (EOFError, OSError):
                connection.close()
                self.root_connection = None
                return
            if isinstance(update, TaskCancellation):
                self.request_retirement(update.deadline)
                if self.supervisor.is_alive():
                    try:
                        self.connection.send(update)
                    except (BrokenPipeError, EOFError, OSError):
                        self.state = TaskScopeState.FAILED
                continue
            if isinstance(update, TaskExecutionWindow):
                if update.root != self.root:
                    raise TaskSupervisionFailure(f"{self.name} execution window root identity does not match")
                if self.supervisor.is_alive():
                    try:
                        self.connection.send(update)
                    except (BrokenPipeError, EOFError, OSError) as error:
                        self.state = TaskScopeState.FAILED
                        raise TaskSupervisionFailure(f"{self.name} execution timing relay failed: {error}") from error
                continue
            if not isinstance(update, TaskRootUpdate) or update.root != self.root:
                raise TaskSupervisionFailure(f"{self.name} task root sent invalid update {update!r}")
            if update.event is TaskResourceEvent.ACTIVE:
                if self.resource_state is not TaskProtectionState.INACTIVE:
                    raise TaskSupervisionFailure(f"{self.name} task protection already activated")
                if self.cleanup_deadline is not None:
                    connection.send(TaskCancellation(self.cleanup_deadline))
                    continue
                self.resource_state = TaskProtectionState.ACTIVATING
                try:
                    self.connection.send(update)
                except (BrokenPipeError, EOFError, OSError) as error:
                    self.state = TaskScopeState.FAILED
                    raise TaskSupervisionFailure(f"{self.name} protection commit failed: {error}") from error
            else:
                if not self.is_protected and self.cleanup_deadline is None:
                    raise TaskSupervisionFailure(f"{self.name} unexpected task cleanup proof")
                never_committed = self.resource_state is TaskProtectionState.INACTIVE
                self.resource_state = TaskProtectionState.RETIRING
                if not never_committed:
                    try:
                        self.connection.send(update)
                    except (BrokenPipeError, EOFError, OSError):
                        self.state = TaskScopeState.FAILED
                    continue
                self.resource_state = TaskProtectionState.CLEANED
                connection.send(TaskRootAcknowledged(TaskResourceEvent.CLEANED))

    def request_retirement(self, deadline: float | None = None) -> None:
        """Request the same owner-driven operation without refreshing its clock."""

        if self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S if deadline is None else deadline
        elif deadline is not None:
            self.cleanup_deadline = min(self.cleanup_deadline, deadline)
        if self.root_connection is not None and self.is_protected:
            try:
                self.root_connection.send(TaskCancellation(self.cleanup_deadline))
            except (BrokenPipeError, EOFError, OSError) as error:
                logger.error("task cancellation channel unavailable name=%s detail=%s", self.name, error)
                self.root_connection.close()
                self.root_connection = None

    def accept_message(self, message: TaskSupervisorMessage) -> TaskCompletion | None:
        """Apply one supervision transition, keeping proof distinct from verdict."""

        if isinstance(message, TaskRootAcknowledged):
            if message.event is TaskResourceEvent.ACTIVE:
                if self.resource_state is TaskProtectionState.RETIRING:
                    return None
                if self.resource_state is not TaskProtectionState.ACTIVATING:
                    raise TaskSupervisionFailure(f"{self.name} unexpected protection acknowledgement")
                self.resource_state = TaskProtectionState.ACTIVE
                if self.cleanup_deadline is not None:
                    self.request_retirement()
                    return None
            elif self.resource_state is TaskProtectionState.RETIRING:
                self.resource_state = TaskProtectionState.CLEANED
            else:
                raise TaskSupervisionFailure(f"{self.name} unexpected cleanup acknowledgement")
            if self.root_connection is not None:
                self.root_connection.send(message)
            return None
        if isinstance(message, TaskCancellation):
            self.request_retirement(message.deadline)
            return None
        if isinstance(message, (TaskCompletion, TaskScopeDrained)):
            if self.is_protected:
                raise TaskSupervisionFailure(f"{self.name} reported empty scope without resource cleanup proof")
            if isinstance(message, TaskCompletion):
                self.completion = message
                self.state = TaskScopeState.COMPLETED
                return message
            self.state = TaskScopeState.DRAINED
            return None
        self.state = TaskScopeState.FAILED
        if isinstance(message, TaskSupervisorFailed):
            raise TaskSupervisionFailure(f"{self.name} task supervisor failed: {message.diagnostics}")
        raise TaskSupervisionFailure(f"{self.name} task supervisor sent invalid message: {message!r}")

    @classmethod
    def run(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float | None,
    ) -> TaskCompletion:
        """Run one complete serial task lifecycle and return only after cleanup."""

        try:
            scope = cls.start(
                name,
                command,
                cwd=cwd,
                env=env,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
            )
        except TaskStartFailure as error:
            return TaskCompletion(TaskCompletionKind.INFRASTRUCTURE_FAILED, None, str(error))

        try:
            completion = scope.wait()
        except TaskSupervisionFailure as error:
            diagnostics = str(error)
            cls.terminate_all((scope,))
            scope.close()
            return TaskCompletion(TaskCompletionKind.INFRASTRUCTURE_FAILED, None, diagnostics)
        except BaseException as error:
            if scope.state not in (TaskScopeState.CLOSED, TaskScopeState.FAILED):
                try:
                    cls.terminate_all((scope,))
                    scope.close()
                except TaskScopeFailure as cleanup_error:
                    raise cleanup_error from error
            raise
        scope.close()
        return completion

    @classmethod
    def start(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float | None,
    ) -> Self:
        """Start a supervisor and transfer the task's cleanup-control channel."""

        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("supervised task timeout_seconds must be positive and finite, or None")
        prepare_task_supervision()
        baseline_children = frozenset(direct_child_process_ids())
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        root_connection, task_connection = context.Pipe(duplex=True)
        supervisor = context.Process(
            name=f"xpool-task-supervisor:{name}",
            target=run_task_supervisor,
            args=(
                child_connection,
                name,
                command,
                cwd,
                dict(env),
                log_path,
                timeout_seconds,
            ),
        )
        scope = cls(name=name, supervisor=supervisor, connection=parent_connection, root_connection=root_connection)
        try:
            start_spawn_process(supervisor)
            if supervisor.pid is None:
                raise TaskStartFailure("task supervisor has no process identity for descriptor transfer")
            multiprocessing.reduction.send_handle(parent_connection, task_connection.fileno(), supervisor.pid)
            task_connection.close()
            child_connection.close()
            if not parent_connection.poll(TASK_SUPERVISOR_START_TIMEOUT_SECONDS):
                raise TaskSupervisionFailure(f"{name} task supervisor did not acknowledge startup")
            message = scope.receive_message()
            if isinstance(message, TaskSupervisorStarted):
                scope.root = message.root
                scope.state = TaskScopeState.RUNNING
                return scope
            if isinstance(message, TaskSupervisorFailed):
                raise TaskSupervisionFailure(f"{name} task supervisor failed during startup: {message.diagnostics}")
            raise TaskSupervisionFailure(f"{name} task supervisor sent an invalid startup message: {message!r}")
        except BaseException as startup_error:
            task_connection.close()
            child_connection.close()
            try:
                scope.abort_startup(baseline_children)
            except BaseException as cleanup_error:
                raise TaskScopeFailure(
                    f"{name} task startup failed ({startup_error}) and its attempted domain could not be proved "
                    f"empty: {cleanup_error}"
                ) from startup_error
            raise TaskStartFailure(f"{name} task startup failed: {startup_error}") from startup_error

    def poll(self) -> TaskCompletion | None:
        """Return a completion without blocking, or fail if the supervisor vanished."""

        if self.state is TaskScopeState.COMPLETED:
            assert self.completion is not None
            return self.completion
        if self.state is TaskScopeState.DRAINED:
            raise RuntimeError(f"{self.name} task scope was drained by infrastructure cancellation")
        if self.state is TaskScopeState.FAILED:
            raise TaskSupervisionFailure(f"{self.name} task supervisor previously failed")
        if self.state is TaskScopeState.CLOSED:
            raise RuntimeError(f"{self.name} task scope is closed")
        if self.state is not TaskScopeState.RUNNING:
            raise RuntimeError(f"{self.name} task scope is not running: {self.state}")
        self.service_root_control()
        if self.is_protected and self.cleanup_deadline is not None and time.monotonic() >= self.cleanup_deadline:
            self.state = TaskScopeState.FAILED
            raise TaskSupervisionFailure(f"{self.name} cleanup expired; retaining owner and device grant")
        if self.connection.poll():
            try:
                message = self.receive_message()
                return self.accept_message(message)
            except TaskSupervisionFailure:
                self.state = TaskScopeState.FAILED
                raise
        if not self.supervisor.is_alive():
            if self.connection.poll():
                return self.poll()
            self.state = TaskScopeState.FAILED
            raise TaskSupervisionFailure(
                f"{self.name} task supervisor exited with code {self.supervisor.exitcode} before proving scope empty"
            )
        return None

    def wait(self) -> TaskCompletion:
        """Wait without a parent-side deadline for supervisor-owned completion."""

        while True:
            completion = self.poll()
            if completion is not None:
                return completion
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)

    @classmethod
    def terminate_all(cls, scopes: Sequence[Self]) -> None:
        """Fan out cancellation and require every supplied scope to become empty."""

        active = tuple(scope for scope in scopes if scope.state in (TaskScopeState.RUNNING, TaskScopeState.FAILED))
        if not active:
            return
        term_deadline = time.monotonic() + PROCESS_TERMINATE_TIMEOUT_SECONDS
        kill_deadline = term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
        cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
        supervision_failures: list[str] = []
        for scope in active:
            try:
                scope.service_root_control()
                scope.request_retirement(cleanup_deadline)
                if scope.state is TaskScopeState.FAILED:
                    supervision_failures.append(f"{scope.name} task scope requires runner fallback")
                    continue
                if scope.poll() is not None:
                    continue
            except (TaskSupervisionFailure, EOFError, OSError) as error:
                scope.request_retirement(cleanup_deadline)
                scope.state = TaskScopeState.FAILED
                supervision_failures.append(str(error))
                continue
            try:
                scope.connection.send(TerminateTaskScope(term_deadline, kill_deadline, scope.cleanup_deadline))
            except (BrokenPipeError, EOFError, OSError) as error:
                try:
                    if scope.poll() is not None:
                        continue
                except TaskSupervisionFailure as reconciliation_error:
                    scope.state = TaskScopeState.FAILED
                    supervision_failures.append(
                        f"{scope.name}: cancellation send failed: {error}; "
                        f"completion reconciliation failed: {reconciliation_error}"
                    )
                else:
                    scope.state = TaskScopeState.FAILED
                    supervision_failures.append(f"{scope.name}: cancellation send failed: {error}")
                continue
        pending = list(active)
        expiry_reported: set[str] = set()
        response_deadline = kill_deadline + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS
        while pending:
            for scope in tuple(pending):
                try:
                    scope.service_root_control()
                    if scope.connection.poll():
                        scope.accept_message(scope.receive_message())
                    if scope.state in (TaskScopeState.DRAINED, TaskScopeState.COMPLETED):
                        pending.remove(scope)
                        continue
                    if not scope.supervisor.is_alive():
                        raise TaskSupervisionFailure(f"{scope.name} task supervisor exited during cancellation")
                    if time.monotonic() >= response_deadline and not scope.is_protected:
                        raise TaskSupervisionFailure(f"{scope.name} task supervisor exceeded its cancellation bound")
                except Exception as error:
                    if scope.state is not TaskScopeState.FAILED:
                        supervision_failures.append(str(error))
                    scope.state = TaskScopeState.FAILED
                if scope.state is TaskScopeState.FAILED and not scope.is_protected:
                    pending.remove(scope)
                elif (
                    scope.cleanup_deadline is not None
                    and time.monotonic() >= scope.cleanup_deadline
                    and scope.name not in expiry_reported
                ):
                    logger.error(
                        "task cleanup expired; retaining owner name=%s; manual resolution required", scope.name
                    )
                    expiry_reported.add(scope.name)
            if pending:
                time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
        if not supervision_failures and any(scope.state is TaskScopeState.FAILED for scope in active):
            supervision_failures.append("task cancellation requires runner fallback")
        if supervision_failures:
            # Protected scopes remain in the wait above. Generic fallback only
            # reaches tasks whose resource protection was never committed or
            # whose cleanup has already been acknowledged by the supervisor.
            retained_deadline = min(
                (
                    scope.cleanup_deadline
                    for scope in active
                    if scope.resource_state is not TaskProtectionState.INACTIVE and scope.cleanup_deadline is not None
                ),
                default=None,
            )
            fallback_failures: list[str] = []
            try:
                reap_task_supervisors(scopes, cleanup_deadline=retained_deadline)
            except TaskScopeFailure as error:
                fallback_failures.append(str(error))
            try:
                drain_unprotected_subreaper_descendants(cleanup_deadline=retained_deadline)
            except TaskScopeFailure as error:
                fallback_failures.append(str(error))
            if not fallback_failures:
                for scope in scopes:
                    if scope.state is TaskScopeState.FAILED:
                        scope.state = TaskScopeState.DRAINED
                return
            raise TaskScopeFailure(
                "task-scope fallback failed after "
                + "; ".join(supervision_failures)
                + ": "
                + "; ".join(fallback_failures)
            )

    def close(self) -> None:
        """Release IPC and supervisor handles after the scope is proven empty."""

        if self.state is TaskScopeState.CLOSED:
            return
        if self.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
            raise RuntimeError(f"cannot close unproven task scope {self.name}")
        if self.is_protected:
            raise RuntimeError(f"cannot close protected task scope {self.name}")
        self.supervisor.join(TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        if self.supervisor.is_alive():
            self.state = TaskScopeState.FAILED
            raise TaskSupervisionFailure(f"{self.name} task supervisor did not exit after proving scope empty")
        self.connection.close()
        if self.root_connection is not None:
            self.root_connection.close()
        self.supervisor.close()
        self.state = TaskScopeState.CLOSED

    def receive_message(self) -> TaskSupervisorMessage:
        """Receive one typed supervisor message or classify a broken channel."""

        try:
            message = self.connection.recv()
        except (EOFError, OSError) as error:
            raise TaskSupervisionFailure(f"{self.name} task supervisor channel failed: {error}") from error
        if not isinstance(
            message,
            (
                TaskSupervisorStarted,
                TaskCompletion,
                TaskScopeDrained,
                TaskSupervisorFailed,
                TaskRootAcknowledged,
                TaskCancellation,
            ),
        ):
            raise TaskSupervisionFailure(f"{self.name} task supervisor sent unknown message {message!r}")
        return message

    def kill_unresponsive_supervisor(self) -> None:
        """Bound parent-side cleanup to a supervisor that never established its contract."""

        if self.supervisor.pid is None:
            return
        if self.supervisor.is_alive():
            self.supervisor.kill()
        self.supervisor.join(TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        if self.supervisor.is_alive():
            raise TaskScopeFailure(f"{self.name} task supervisor survived SIGKILL during startup rollback")

    def abort_startup(self, baseline_children: frozenset[ProcUniqId]) -> None:
        """Reap an unacknowledged Supervisor and every child it may have launched."""

        try:
            self.kill_unresponsive_supervisor()
            drain_new_subreaper_descendants(baseline_children)
        except BaseException:
            self.state = TaskScopeState.FAILED
            raise
        finally:
            self.connection.close()
            if self.root_connection is not None:
                self.root_connection.close()
            if self.supervisor.pid is None or not self.supervisor.is_alive():
                self.supervisor.close()
        self.state = TaskScopeState.CLOSED


def run_task_supervisor(
    connection: Connection,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float | None,
) -> None:
    """Launch and supervise one root process from a dedicated child subreaper."""

    root: subprocess.Popen[str] | None = None
    log_file: TextIO | None = None
    started = False
    task_connection: Connection | None = None
    protection: TaskProtection | None = None
    exit_code = 0
    try:
        os.setsid()
        set_child_subreaper()
        task_connection = Connection(multiprocessing.reduction.recv_handle(connection))
        env[TASK_ROOT_CONTROL_FD_ENV] = str(task_connection.fileno())
        log_file = log_path.open("w", encoding="utf-8")
        root = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            process_group=0,
            pass_fds=(task_connection.fileno(),),
            text=True,
        )
        task_connection.close()
        task_connection = None
        task_deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        protection = TaskProtection(ProcUniqId(root.pid))
        connection.send(TaskSupervisorStarted(protection.root))
        started = True
        connection.send(supervise_task(connection, root, protection=protection, task_deadline=task_deadline))
    except BaseException:
        exit_code = 1
        diagnostics = traceback.format_exc()
        scope_empty = root is None
        if root is not None:
            now = time.monotonic()
            try:
                if protection is not None and protection.event is not None:
                    with suppress(BrokenPipeError, EOFError, OSError):
                        connection.send(protection.retire())
                    wait_for_task_cleanup(connection, protection)
                    drain_retired_task_scope(root, protection)
                else:
                    drain_task_scope(
                        root,
                        term_deadline=now + PROCESS_TERMINATE_TIMEOUT_SECONDS,
                        kill_deadline=now + PROCESS_TERMINATE_TIMEOUT_SECONDS + PROCESS_KILL_TIMEOUT_SECONDS,
                    )
                scope_empty = True
            except BaseException:
                diagnostics += "\nTask Supervisor local cleanup failed:\n" + traceback.format_exc()
        with suppress(BrokenPipeError, EOFError, OSError):
            if started and scope_empty:
                connection.send(
                    TaskCompletion(
                        TaskCompletionKind.INFRASTRUCTURE_FAILED,
                        None,
                        diagnostics[-LOG_TAIL_CHARS:],
                    )
                )
            else:
                connection.send(TaskSupervisorFailed(diagnostics[-LOG_TAIL_CHARS:]))
    if root is not None:
        root.poll()
    if log_file is not None:
        log_file.close()
    if task_connection is not None:
        task_connection.close()
    connection.close()
    os._exit(exit_code)


def supervise_task(
    connection: Connection,
    root: subprocess.Popen[str],
    *,
    protection: TaskProtection,
    task_deadline: float | None,
) -> TaskCompletion | TaskScopeDrained:
    """Drive timeout, natural drain, leak cleanup, and cancellation for one task."""

    natural_drain_deadline: float | None = None
    item_deadline: float | None = None
    item_expired = False
    while True:
        if connection.poll():
            command = connection.recv()
            if isinstance(command, TaskRootUpdate):
                protection.commit(command, connection)
                continue
            if isinstance(command, TaskExecutionWindow):
                if command.root != protection.root:
                    raise TaskScopeFailure("task execution window root identity does not match")
                if (
                    protection.event is TaskResourceEvent.ACTIVE
                    and item_deadline is not None
                    and command.changed_at >= item_deadline
                ):
                    item_expired = True
                item_deadline = command.deadline
                continue
            if isinstance(command, TaskCancellation):
                # The invocation finished and entered final retirement. Cleanup
                # is timed by its original envelope rather than execution time.
                protection.retire(command.deadline)
                task_deadline = None
                item_deadline = None
                continue
            if not isinstance(command, TerminateTaskScope):
                raise TaskScopeFailure(f"task supervisor received invalid command {command!r}")
            if protection.event is not None:
                retirement = protection.retire(command.cleanup_deadline)
                if protection.event is TaskResourceEvent.ACTIVE:
                    connection.send(retirement)
                    wait_for_task_cleanup(connection, protection)
                drain_retired_task_scope(root, protection)
                return TaskScopeDrained()
            drain_task_scope(root, term_deadline=command.term_deadline, kill_deadline=command.kill_deadline)
            return TaskScopeDrained()

        root_returncode = root.poll()
        scope_empty = reap_task_children(root)
        descendants = descendant_process_ids()
        now = time.monotonic()
        if protection.event is not None and (item_expired or (item_deadline is not None and now >= item_deadline)):
            connection.send(protection.retire())
            if protection.event is TaskResourceEvent.ACTIVE:
                wait_for_task_cleanup(connection, protection)
            drain_retired_task_scope(root, protection)
            return TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "task item exceeded its execution timeout")
        if root_returncode is not None and protection.event is TaskResourceEvent.ACTIVE:
            connection.send(protection.retire())
            wait_for_task_cleanup(connection, protection)
            drain_retired_task_scope(root, protection)
            return TaskCompletion(TaskCompletionKind.EXITED, root_returncode, None)
        if root_returncode is None:
            if task_deadline is not None and now >= task_deadline:
                if protection.event is TaskResourceEvent.ACTIVE:
                    connection.send(protection.retire())
                    wait_for_task_cleanup(connection, protection)
                    drain_retired_task_scope(root, protection)
                    return TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "task exceeded its execution timeout")
                term_deadline = now + PROCESS_TERMINATE_TIMEOUT_SECONDS
                drain_task_scope(
                    root, term_deadline=term_deadline, kill_deadline=term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
                )
                return TaskCompletion(
                    TaskCompletionKind.TIMED_OUT,
                    None,
                    f"task exceeded its timeout with live processes {format_process_ids(descendants)}",
                )
        elif scope_empty:
            return TaskCompletion(TaskCompletionKind.EXITED, root_returncode, None)
        elif natural_drain_deadline is None:
            natural_drain_deadline = now + TASK_NATURAL_DRAIN_TIMEOUT_SECONDS
        elif now >= natural_drain_deadline:
            diagnostics = f"descendants outlived the task root: {format_process_ids(descendants)}"
            term_deadline = now + PROCESS_TERMINATE_TIMEOUT_SECONDS
            if protection.event is TaskResourceEvent.CLEANED:
                drain_retired_task_scope(root, protection)
            else:
                drain_task_scope(
                    root, term_deadline=term_deadline, kill_deadline=term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
                )
            return TaskCompletion(TaskCompletionKind.LEAKED, root_returncode, diagnostics)
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def wait_for_task_cleanup(connection: Connection, protection: TaskProtection) -> None:
    """Retain supervision until exact-root proof permits generic drain."""

    expiry_reported = False
    last_error: str | None = None
    channel_available = True
    while protection.event is TaskResourceEvent.ACTIVE:
        try:
            if channel_available and connection.poll(PROCESS_POLL_INTERVAL_SECONDS):
                command = connection.recv()
                if isinstance(command, TaskRootUpdate):
                    protection.commit(command, connection)
                elif isinstance(command, TerminateTaskScope):
                    connection.send(protection.retire(command.cleanup_deadline))
                elif isinstance(command, TaskCancellation):
                    protection.retire(command.deadline)
                elif isinstance(command, TaskExecutionWindow):
                    # Retirement already owns timing; item teardown may still
                    # publish the end of its former execution interval.
                    if command.root != protection.root:
                        raise TaskScopeFailure("task execution window root identity does not match")
                else:
                    raise TaskScopeFailure(f"task supervisor received invalid retirement command {command!r}")
        except Exception as error:
            if isinstance(error, EOFError | OSError):
                channel_available = False
            diagnostic = str(error)
            if diagnostic != last_error:
                logger.error(
                    "task resource proof unavailable; retaining supervisor root=%s detail=%s",
                    protection.root,
                    diagnostic,
                )
                last_error = diagnostic
        if (
            protection.cleanup_deadline is not None
            and time.monotonic() >= protection.cleanup_deadline
            and not expiry_reported
        ):
            logger.error(
                "task cleanup expired; retaining supervisor root=%s; manual resolution required", protection.root
            )
            expiry_reported = True
        if not channel_available:
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def drain_retired_task_scope(root: subprocess.Popen[str], protection: TaskProtection) -> None:
    """Reap after resource proof within the unchanged task-retirement envelope."""

    deadline = protection.retire().deadline
    now = time.monotonic()
    try:
        drain_task_scope(
            root,
            term_deadline=min(now + PROCESS_TERMINATE_TIMEOUT_SECONDS, deadline),
            kill_deadline=min(now + PROCESS_TERMINATE_TIMEOUT_SECONDS + PROCESS_KILL_TIMEOUT_SECONDS, deadline),
        )
    except Exception as error:
        logger.error("task domain incomplete; retaining supervisor root=%s detail=%s", protection.root, error)
        last_error = str(error)
    else:
        return
    while True:
        try:
            if reap_task_children(root):
                return
        except Exception as observation_error:
            diagnostic = str(observation_error)
            if diagnostic != last_error:
                logger.error(
                    "task domain unconfirmed; retaining supervisor root=%s detail=%s", protection.root, diagnostic
                )
                last_error = diagnostic
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def drain_task_scope(root: subprocess.Popen[str], *, term_deadline: float, kill_deadline: float) -> None:
    """Repeatedly enumerate, signal, and reap one complete descendant domain."""

    term_signaled: set[ProcUniqId] = set()
    while time.monotonic() < term_deadline:
        if reap_task_children(root):
            return
        targets = set(descendant_process_ids())
        for process_id in targets - term_signaled:
            process_id.send_signal(signal.SIGTERM)
            term_signaled.add(process_id)
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, term_deadline - time.monotonic())))

    kill_signaled: set[ProcUniqId] = set()
    while time.monotonic() < kill_deadline:
        if reap_task_children(root):
            return
        targets = set(descendant_process_ids())
        for process_id in targets - kill_signaled:
            process_id.send_signal(signal.SIGKILL)
            kill_signaled.add(process_id)
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, kill_deadline - time.monotonic())))

    if reap_task_children(root):
        return
    survivors = descendant_process_ids()
    raise TaskScopeFailure(f"task scope survived SIGKILL: {format_process_ids(survivors)}")


def descendant_process_ids() -> tuple[ProcUniqId, ...]:
    """Strictly capture every live descendant of the current supervisor."""

    try:
        descendants = psutil.Process(os.getpid()).children(recursive=True)
    except psutil.Error as error:
        raise TaskScopeFailure(f"failed to enumerate task descendants: {error}") from error
    process_ids: set[ProcUniqId] = set()
    for descendant in descendants:
        try:
            process_id = ProcUniqId(descendant.pid)
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, ValueError) as error:
            raise TaskScopeFailure(f"failed to identify task descendant {descendant.pid}: {error}") from error
        if process_id.is_alive():
            process_ids.add(process_id)
    return tuple(sorted(process_ids, key=lambda process_id: (process_id.pid, process_id.create_time)))


def reap_task_children(root: subprocess.Popen[str]) -> bool:
    """Reap adopted children and return the kernel-proven empty-scope state."""

    if root.poll() is None:
        return False
    while True:
        try:
            child_pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if child_pid == 0:
            return False


def format_process_ids(process_ids: Sequence[ProcUniqId]) -> str:
    """Return bounded PID/create-time diagnostics for one descendant snapshot."""

    rendered = ", ".join(f"{process_id.pid}@{process_id.create_time:.6f}" for process_id in process_ids)
    return rendered[-LOG_TAIL_CHARS:] or "<none>"


def set_child_subreaper() -> None:
    """Make the current Linux process adopt orphaned descendants."""

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def prepare_task_supervision() -> None:
    """Enable runner subreaping and identify multiprocessing infrastructure children."""

    global protected_subreaper_process_ids
    if protected_subreaper_process_ids is not None:
        return
    with task_supervision_lock:
        if protected_subreaper_process_ids is not None:
            return
        set_child_subreaper()
        multiprocessing.resource_tracker.ensure_running()
        # Pinned CPython exposes the exact helper identity only on its tracker
        # object; typeshed does not declare the PID field.
        # An inherited tracker belongs to an ancestor and has no local PID.
        tracker_pid = cast(int | None, getattr(multiprocessing.resource_tracker._resource_tracker, "_pid"))
        protected_subreaper_process_ids = frozenset() if tracker_pid is None else frozenset((ProcUniqId(tracker_pid),))


def direct_child_process_ids() -> tuple[ProcUniqId, ...]:
    """Strictly capture live direct children of the current process."""

    try:
        children = psutil.Process(os.getpid()).children(recursive=False)
    except psutil.Error as error:
        raise TaskScopeFailure(f"failed to enumerate direct child processes: {error}") from error
    process_ids: list[ProcUniqId] = []
    for child in children:
        try:
            process_id = ProcUniqId(child.pid)
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, ValueError) as error:
            raise TaskScopeFailure(f"failed to identify direct child {child.pid}: {error}") from error
        if process_id.is_alive():
            process_ids.append(process_id)
    return tuple(sorted(process_ids, key=lambda process_id: (process_id.pid, process_id.create_time)))


def subreaper_direct_roots(excluded: frozenset[ProcUniqId]) -> tuple[ProcUniqId, ...]:
    """Reap unexcluded zombies and return every live unexcluded direct child."""

    try:
        children = psutil.Process(os.getpid()).children(recursive=False)
    except psutil.Error as error:
        raise TaskScopeFailure(f"failed to enumerate subreaper roots: {error}") from error
    roots: list[ProcUniqId] = []
    for child in children:
        try:
            process_id = ProcUniqId(child.pid)
            status = child.status()
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, ValueError) as error:
            raise TaskScopeFailure(f"failed to inspect subreaper root {child.pid}: {error}") from error
        if process_id in excluded:
            continue
        if status == psutil.STATUS_ZOMBIE:
            with suppress(ChildProcessError, ProcessLookupError):
                os.waitpid(child.pid, os.WNOHANG)
            continue
        if process_id.is_alive():
            roots.append(process_id)
    return tuple(sorted(roots, key=lambda process_id: (process_id.pid, process_id.create_time)))


def unprotected_subreaper_roots() -> tuple[ProcUniqId, ...]:
    """Return live runner direct roots other than exact multiprocessing helpers."""

    return subreaper_direct_roots(protected_subreaper_process_ids or frozenset())


def process_tree_ids(roots: Sequence[ProcUniqId]) -> tuple[ProcUniqId, ...]:
    """Capture complete currently visible trees rooted at exact process identities."""

    process_ids: set[ProcUniqId] = set()
    for root in roots:
        process_ids.update(root.child_process_ids())
        if root.is_alive():
            process_ids.add(root)
    return tuple(sorted(process_ids, key=lambda process_id: (process_id.pid, process_id.create_time)))


def drain_subreaper_descendants(
    excluded_roots: frozenset[ProcUniqId],
    *,
    context: str,
    cleanup_deadline: float | None = None,
) -> None:
    """Drain adopted trees, consuming a managed task's existing retirement clock.

    Without a managed clock, existing bounded generic cleanup applies. With
    one, expiry or observation failure retains this owner until domain proof;
    it never grants another signalling phase.
    """

    term_deadline = time.monotonic() + PROCESS_TERMINATE_TIMEOUT_SECONDS
    kill_deadline = term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
    if cleanup_deadline is not None:
        term_deadline = min(term_deadline, cleanup_deadline)
        kill_deadline = min(kill_deadline, cleanup_deadline)
    last_error: str | None = None
    for signum, phase_deadline in ((signal.SIGTERM, term_deadline), (signal.SIGKILL, kill_deadline)):
        signaled: set[ProcUniqId] = set()
        while time.monotonic() < phase_deadline:
            try:
                roots = subreaper_direct_roots(excluded_roots)
                if not roots:
                    return
                for process_id in set(process_tree_ids(roots)) - signaled:
                    process_id.send_signal(signum)
                    signaled.add(process_id)
            except Exception as error:
                if cleanup_deadline is None:
                    raise
                diagnostic = str(error)
                if diagnostic != last_error:
                    logger.error("%s unconfirmed; retaining owner detail=%s", context, diagnostic)
                    last_error = diagnostic
            time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, phase_deadline - time.monotonic())))

    expiry_reported = False
    while True:
        try:
            survivors = subreaper_direct_roots(excluded_roots)
            if not survivors:
                return
            if cleanup_deadline is None:
                raise TaskScopeFailure(f"{context} survived SIGKILL: {format_process_ids(survivors)}")
        except Exception as error:
            if cleanup_deadline is None:
                raise
            diagnostic = str(error)
            if diagnostic != last_error:
                logger.error("%s unconfirmed; retaining owner detail=%s", context, diagnostic)
                last_error = diagnostic
        if not expiry_reported:
            logger.error("%s incomplete; retaining owner; manual resolution required", context)
            expiry_reported = True
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def drain_new_subreaper_descendants(baseline_children: frozenset[ProcUniqId]) -> None:
    """Drain direct-root trees adopted while one Supervisor startup was attempted."""

    excluded = baseline_children | (protected_subreaper_process_ids or frozenset())
    drain_subreaper_descendants(excluded, context="task startup descendants")


def drain_unprotected_subreaper_descendants(*, cleanup_deadline: float | None = None) -> None:
    """Clean descendants adopted after a Task Supervisor infrastructure failure."""

    drain_subreaper_descendants(
        protected_subreaper_process_ids or frozenset(),
        context="runner-adopted descendants",
        cleanup_deadline=cleanup_deadline,
    )


def reap_task_supervisors(scopes: Sequence[SupervisedTaskScope], *, cleanup_deadline: float | None = None) -> None:
    """Cancel and reap every known Supervisor before runner-level fallback."""

    known = tuple(
        scope for scope in scopes if scope.state is not TaskScopeState.CLOSED and scope.supervisor.pid is not None
    )
    for scope in known:
        if scope.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED, TaskScopeState.CLOSED):
            scope.state = TaskScopeState.FAILED
            try:
                if scope.supervisor.is_alive() and (cleanup_deadline is None or time.monotonic() < cleanup_deadline):
                    with suppress(ProcessLookupError):
                        scope.supervisor.kill()
            except Exception as error:
                if cleanup_deadline is None:
                    raise
                logger.error("task supervisor unconfirmed; retaining owner name=%s detail=%s", scope.name, error)

    if cleanup_deadline is not None:
        wait_for_supervisors(known, cleanup_deadline, retain_after_expiry=True)
        return
    wait_for_supervisors(known, time.monotonic() + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
    survivors = tuple(scope for scope in known if scope.supervisor.is_alive())
    for scope in survivors:
        scope.state = TaskScopeState.FAILED
        with suppress(ProcessLookupError):
            scope.supervisor.kill()
    wait_for_supervisors(survivors, time.monotonic() + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
    survivors = tuple(scope.name for scope in survivors if scope.supervisor.is_alive())
    if survivors:
        raise TaskScopeFailure(f"Task Supervisors survived SIGKILL: {survivors}")


def wait_for_supervisors(
    scopes: Sequence[SupervisedTaskScope], deadline: float, *, retain_after_expiry: bool = False
) -> None:
    """Reap known Supervisor processes under one shared deadline."""

    pending = list(scopes)
    expiry_reported = False
    last_error: str | None = None
    while pending and (time.monotonic() < deadline or retain_after_expiry):
        for scope in tuple(pending):
            try:
                scope.supervisor.join(0)
                if not scope.supervisor.is_alive():
                    pending.remove(scope)
            except Exception as error:
                if not retain_after_expiry:
                    raise
                diagnostic = str(error)
                if diagnostic != last_error:
                    logger.error(
                        "task supervisor unconfirmed; retaining owner name=%s detail=%s", scope.name, diagnostic
                    )
                    last_error = diagnostic
        if pending:
            if time.monotonic() >= deadline and not expiry_reported:
                logger.error("task supervisors incomplete; retaining owner; manual resolution required")
                expiry_reported = True
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
