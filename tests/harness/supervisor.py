"""Supervised task-scope lifecycle and descendant-tree ownership for tests."""

from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.process
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
from typing import Self, TextIO

import psutil

from tests.harness.process import (
    LOG_TAIL_CHARS,
    PROCESS_KILL_TIMEOUT_SECONDS,
    PROCESS_POLL_INTERVAL_SECONDS,
    PROCESS_TERMINATE_TIMEOUT_SECONDS,
    start_spawn_process,
)
from xpool.utils.procs import ProcUniqId

TASK_NATURAL_DRAIN_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_START_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS = 5.0
PR_SET_CHILD_SUBREAPER = 36
task_supervision_lock = threading.Lock()
protected_subreaper_process_ids: frozenset[ProcUniqId] | None = None


class TaskCompletionKind(StrEnum):
    """Proven-empty terminal outcome of one supervised GPU task."""

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
    """Supervisor acknowledgement sent after the pytest root is launched."""


@dataclass(frozen=True, slots=True)
class TerminateTaskScope:
    """Infrastructure cancellation command with shared absolute deadlines."""

    term_deadline: float
    kill_deadline: float


@dataclass(frozen=True, slots=True)
class TaskScopeDrained:
    """Acknowledgement that infrastructure cancellation proved the scope empty."""


@dataclass(frozen=True, slots=True)
class TaskSupervisorFailed:
    """Typed supervisor failure that cannot be represented as task completion."""

    diagnostics: str


TaskSupervisorMessage = TaskSupervisorStarted | TaskCompletion | TaskScopeDrained | TaskSupervisorFailed


@dataclass(slots=True)
class SupervisedTaskScope:
    """Own one child-subreaper supervisor and its complete descendant domain."""

    name: str
    supervisor: multiprocessing.process.BaseProcess
    connection: Connection
    state: TaskScopeState = TaskScopeState.STARTING
    completion: TaskCompletion | None = None

    @classmethod
    def run(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float,
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
            try:
                completion = scope.wait()
                scope.close()
                return completion
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

    @classmethod
    def start(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float,
    ) -> Self:
        """Start one clean supervisor and wait until it launches the task root."""

        if timeout_seconds <= 0:
            raise ValueError("supervised task timeout_seconds must be positive")
        prepare_task_supervision()
        baseline_children = frozenset(direct_child_process_ids())
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
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
        scope = cls(name=name, supervisor=supervisor, connection=parent_connection)
        try:
            start_spawn_process(supervisor)
        except BaseException as startup_error:
            child_connection.close()
            try:
                scope.abort_startup(baseline_children)
            except BaseException as cleanup_error:
                raise TaskScopeFailure(
                    f"{name} task supervisor spawn failed ({startup_error}) and its attempted domain could not be "
                    f"proved empty: {cleanup_error}"
                ) from startup_error
            raise TaskStartFailure(f"{name} task supervisor spawn failed: {startup_error}") from startup_error
        child_connection.close()
        try:
            if not parent_connection.poll(TASK_SUPERVISOR_START_TIMEOUT_SECONDS):
                raise TaskSupervisionFailure(f"{name} task supervisor did not acknowledge startup")
            message = scope.receive_message()
            if isinstance(message, TaskSupervisorStarted):
                scope.state = TaskScopeState.RUNNING
                return scope
            if isinstance(message, TaskSupervisorFailed):
                raise TaskSupervisionFailure(f"{name} task supervisor failed during startup: {message.diagnostics}")
            raise TaskSupervisionFailure(f"{name} task supervisor sent an invalid startup message: {message!r}")
        except BaseException as startup_error:
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
        if self.connection.poll():
            try:
                message = self.receive_message()
            except TaskSupervisionFailure:
                self.state = TaskScopeState.FAILED
                raise
            if isinstance(message, TaskCompletion):
                self.completion = message
                self.state = TaskScopeState.COMPLETED
                return message
            if isinstance(message, TaskSupervisorFailed):
                self.state = TaskScopeState.FAILED
                raise TaskSupervisionFailure(f"{self.name} task supervisor failed: {message.diagnostics}")
            self.state = TaskScopeState.FAILED
            raise TaskSupervisionFailure(f"{self.name} task supervisor sent an invalid completion message: {message!r}")
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
        supervision_failures: list[str] = []
        waiting: list[SupervisedTaskScope] = []
        for scope in active:
            if scope.state is TaskScopeState.FAILED:
                supervision_failures.append(f"{scope.name} task scope requires runner fallback")
                continue
            try:
                if scope.poll() is not None:
                    continue
            except TaskSupervisionFailure as error:
                scope.state = TaskScopeState.FAILED
                supervision_failures.append(str(error))
                continue
            try:
                scope.connection.send(TerminateTaskScope(term_deadline, kill_deadline))
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
            waiting.append(scope)
        for scope in waiting:
            try:
                scope.wait_for_termination(kill_deadline)
            except TaskSupervisionFailure as error:
                supervision_failures.append(str(error))
        if supervision_failures:
            fallback_failures: list[str] = []
            try:
                reap_task_supervisors(scopes)
            except TaskScopeFailure as error:
                fallback_failures.append(str(error))
            try:
                drain_unprotected_subreaper_descendants()
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
        self.supervisor.join(TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        if self.supervisor.is_alive():
            self.state = TaskScopeState.FAILED
            raise TaskSupervisionFailure(f"{self.name} task supervisor did not exit after proving scope empty")
        self.connection.close()
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
            (TaskSupervisorStarted, TaskCompletion, TaskScopeDrained, TaskSupervisorFailed),
        ):
            raise TaskSupervisionFailure(f"{self.name} task supervisor sent unknown message {message!r}")
        return message

    def wait_for_termination(self, kill_deadline: float) -> None:
        """Wait for cancellation acknowledgement under the shared cleanup deadline."""

        response_deadline = kill_deadline + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS
        while time.monotonic() < response_deadline:
            if self.connection.poll(PROCESS_POLL_INTERVAL_SECONDS):
                try:
                    message = self.receive_message()
                except TaskSupervisionFailure:
                    self.state = TaskScopeState.FAILED
                    raise
                if isinstance(message, TaskScopeDrained):
                    self.state = TaskScopeState.DRAINED
                    return
                if isinstance(message, TaskCompletion):
                    self.completion = message
                    self.state = TaskScopeState.COMPLETED
                    return
                if isinstance(message, TaskSupervisorFailed):
                    self.state = TaskScopeState.FAILED
                    raise TaskSupervisionFailure(f"{self.name} task supervisor failed: {message.diagnostics}")
                self.state = TaskScopeState.FAILED
                raise TaskSupervisionFailure(
                    f"{self.name} task supervisor sent invalid cancellation message {message!r}"
                )
            if not self.supervisor.is_alive():
                self.state = TaskScopeState.FAILED
                raise TaskSupervisionFailure(
                    f"{self.name} task supervisor exited with code {self.supervisor.exitcode} during cancellation"
                )
        self.state = TaskScopeState.FAILED
        raise TaskSupervisionFailure(f"{self.name} task supervisor exceeded the shared cleanup deadline")

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
    timeout_seconds: float,
) -> None:
    """Launch and supervise one root process from a dedicated child subreaper."""

    root: subprocess.Popen[str] | None = None
    log_file: TextIO | None = None
    started = False
    try:
        os.setsid()
        set_child_subreaper()
        log_file = log_path.open("w", encoding="utf-8")
        root = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            process_group=0,
            text=True,
        )
        task_deadline = time.monotonic() + timeout_seconds
        connection.send(TaskSupervisorStarted())
        started = True
        connection.send(supervise_task(connection, root, task_deadline=task_deadline))
    except BaseException:
        diagnostics = traceback.format_exc()
        scope_empty = root is None
        if root is not None:
            now = time.monotonic()
            try:
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
        connection.close()
        os._exit(1)
    if root is not None:
        root.poll()
    if log_file is not None:
        log_file.close()
    connection.close()
    os._exit(0)


def supervise_task(
    connection: Connection,
    root: subprocess.Popen[str],
    *,
    task_deadline: float,
) -> TaskCompletion | TaskScopeDrained:
    """Drive timeout, natural drain, leak cleanup, and cancellation for one task."""

    natural_drain_deadline: float | None = None
    while True:
        if connection.poll():
            command = connection.recv()
            if not isinstance(command, TerminateTaskScope):
                raise TaskScopeFailure(f"task supervisor received invalid command {command!r}")
            drain_task_scope(root, term_deadline=command.term_deadline, kill_deadline=command.kill_deadline)
            return TaskScopeDrained()

        root_returncode = root.poll()
        scope_empty = reap_task_children(root)
        descendants = descendant_process_ids()
        now = time.monotonic()
        if root_returncode is None:
            if now >= task_deadline:
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
            drain_task_scope(
                root, term_deadline=term_deadline, kill_deadline=term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
            )
            return TaskCompletion(TaskCompletionKind.LEAKED, root_returncode, diagnostics)
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def drain_task_scope(root: subprocess.Popen[str], *, term_deadline: float, kill_deadline: float) -> None:
    """Repeatedly enumerate, signal, and reap one complete descendant domain."""

    try:
        root_identity = ProcUniqId(root.pid)
    except (psutil.Error, ValueError):
        root_identity = None
    term_signaled: set[ProcUniqId] = set()
    while time.monotonic() < term_deadline:
        if reap_task_children(root):
            return
        targets = set(descendant_process_ids())
        if root_identity is not None and root_identity.is_alive():
            targets.add(root_identity)
        for process_id in targets - term_signaled:
            process_id.send_signal(signal.SIGTERM)
            term_signaled.add(process_id)
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, term_deadline - time.monotonic())))

    kill_signaled: set[ProcUniqId] = set()
    while time.monotonic() < kill_deadline:
        if reap_task_children(root):
            return
        targets = set(descendant_process_ids())
        if root_identity is not None and root_identity.is_alive():
            targets.add(root_identity)
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
        children_before = frozenset(direct_child_process_ids())
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=time.sleep, args=(0,))
        start_spawn_process(process)
        process.join(TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
            process.close()
            raise TaskScopeFailure("failed to initialize multiprocessing spawn supervision")
        process.close()
        children_after = frozenset(direct_child_process_ids())
        protected_subreaper_process_ids = children_after - children_before


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
) -> None:
    """Drain complete trees rooted at every unexcluded direct subreaper child."""

    term_deadline = time.monotonic() + PROCESS_TERMINATE_TIMEOUT_SECONDS
    kill_deadline = term_deadline + PROCESS_KILL_TIMEOUT_SECONDS
    term_signaled: set[ProcUniqId] = set()
    while time.monotonic() < term_deadline:
        roots = subreaper_direct_roots(excluded_roots)
        if not roots:
            return
        for process_id in set(process_tree_ids(roots)) - term_signaled:
            process_id.send_signal(signal.SIGTERM)
            term_signaled.add(process_id)
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, term_deadline - time.monotonic())))

    kill_signaled: set[ProcUniqId] = set()
    while time.monotonic() < kill_deadline:
        roots = subreaper_direct_roots(excluded_roots)
        if not roots:
            return
        for process_id in set(process_tree_ids(roots)) - kill_signaled:
            process_id.send_signal(signal.SIGKILL)
            kill_signaled.add(process_id)
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, kill_deadline - time.monotonic())))

    survivors = subreaper_direct_roots(excluded_roots)
    if survivors:
        raise TaskScopeFailure(f"{context} survived SIGKILL: {format_process_ids(survivors)}")


def drain_new_subreaper_descendants(baseline_children: frozenset[ProcUniqId]) -> None:
    """Drain direct-root trees adopted while one Supervisor startup was attempted."""

    excluded = baseline_children | (protected_subreaper_process_ids or frozenset())
    drain_subreaper_descendants(excluded, context="task startup descendants")


def drain_unprotected_subreaper_descendants() -> None:
    """Clean descendants adopted after a Task Supervisor infrastructure failure."""

    drain_subreaper_descendants(
        protected_subreaper_process_ids or frozenset(),
        context="runner-adopted descendants",
    )


def reap_task_supervisors(scopes: Sequence[SupervisedTaskScope]) -> None:
    """Cancel and reap every known Supervisor before runner-level fallback."""

    known = tuple(scope for scope in scopes if scope.supervisor.pid is not None)
    for scope in known:
        if scope.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED, TaskScopeState.CLOSED):
            scope.state = TaskScopeState.FAILED
            if scope.supervisor.is_alive():
                with suppress(ProcessLookupError):
                    scope.supervisor.kill()

    deadline = time.monotonic() + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS
    wait_for_supervisors(known, deadline)
    survivors = tuple(scope for scope in known if scope.supervisor.is_alive())
    for scope in survivors:
        scope.state = TaskScopeState.FAILED
        with suppress(ProcessLookupError):
            scope.supervisor.kill()
    wait_for_supervisors(survivors, time.monotonic() + TASK_SUPERVISOR_EXIT_TIMEOUT_SECONDS)
    survivors = tuple(scope.name for scope in survivors if scope.supervisor.is_alive())
    if survivors:
        raise TaskScopeFailure(f"Task Supervisors survived SIGKILL: {survivors}")


def wait_for_supervisors(scopes: Sequence[SupervisedTaskScope], deadline: float) -> None:
    """Reap known Supervisor processes under one shared deadline."""

    pending = list(scopes)
    while pending and time.monotonic() < deadline:
        for scope in tuple(pending):
            scope.supervisor.join(0)
            if not scope.supervisor.is_alive():
                pending.remove(scope)
        if pending:
            time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
