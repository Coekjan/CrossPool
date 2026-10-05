"""Invocation-wide resource proof and cooperative task-root cancellation."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from types import FrameType
from typing import Self

from xkit.process import PROCESS_POLL_INTERVAL_SECONDS
from xpool.utils.background import BackgroundThread
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S
from xpool.utils.procs import ProcUniqId

TASK_ROOT_CONTROL_FD_ENV = "XPOOL_TASK_ROOT_CONTROL_FD"

__all__ = ("TaskCancelled", "TaskCleanupScope", "TaskRoot", "get_task_root")

logger = logging.getLogger(__name__)
current_task_root: TaskRoot | None = None


class TaskResourceEvent(StrEnum):
    """Root notifications that respectively prohibit and permit generic drain."""

    ACTIVE = "active"
    CLEANED = "cleaned"


@dataclass(frozen=True, slots=True)
class TaskRootUpdate:
    """Resource state from the exact root, independent of its program verdict."""

    root: ProcUniqId
    event: TaskResourceEvent


@dataclass(frozen=True, slots=True)
class TaskRootAcknowledged:
    """Confirmation of a committed protection transition."""

    event: TaskResourceEvent


@dataclass(frozen=True, slots=True)
class TaskCancellation:
    """Cooperative retirement under the first absolute monotonic deadline."""

    deadline: float


@dataclass(frozen=True, slots=True)
class TaskExecutionWindow:
    """Current item limit, with producer time for a completed interval.

    ``deadline=None`` ends the interval. The supervisor checks ``changed_at``
    against the previous limit before clearing it, so a completed overdue item
    cannot escape expiry merely because both messages arrived in one poll.
    """

    root: ProcUniqId
    deadline: float | None
    changed_at: float


class TaskCancelled(KeyboardInterrupt):
    """Stop execution through normal Python unwinding, preserving teardown."""


@dataclass(eq=False, slots=True)
class TaskCleanupScope:
    """Outstanding proof owned before controller creation or nested launch.

    This value performs no cleanup. Its actual owner calls ``complete`` only
    after verified ordered close, including failed startup rollback.
    """

    root: TaskRoot

    def complete(self) -> None:
        """Record verified owner cleanup once, leaving failed cleanup pending."""

        with self.root.condition:
            self.root.scopes.discard(self)
            self.root.condition.notify_all()


class TaskRoot:
    """Own task-lifetime protection and aggregate actual owner cleanup proofs.

    The internal worker installs this before invoking its program. ``finish``
    follows complete invocation teardown. Outstanding scopes and missing runner
    acknowledgement retain this live process; expiry never authorizes exit.
    Controller lifecycle and cleanup remain at each actual resource owner.
    """

    def __init__(self, connection: Connection) -> None:
        """Bind a transferred channel and install cooperative cancellation."""

        self.connection = connection
        self.identity = ProcUniqId.current()
        self.condition = threading.Condition()
        self.scopes: set[TaskCleanupScope] = set()
        self.sealed = False
        self.requested_event: TaskResourceEvent | None = None
        self.acknowledged: TaskResourceEvent | None = None
        self.cleanup_deadline: float | None = None
        self.interrupt_pending = False
        self.handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        self.reader = BackgroundThread(
            name="xpool-task-control",
            target=self.read_control,
            join_timeout_s=1.0,
        )
        for signum in self.handlers:
            signal.signal(signum, self.handle_cancel)
        self.reader.start()

    @classmethod
    def from_environment(cls) -> Self | None:
        """Consume the exec descriptor and install this invocation's root.

        Direct invocations return ``None``. Present invalid metadata fails;
        descendants receive neither the metadata nor a second root owner.

        Raises:
            ValueError: Descriptor metadata is invalid or a root is installed.
            OSError: The actual transferred descriptor is unavailable.
        """

        global current_task_root
        value = os.environ.pop(TASK_ROOT_CONTROL_FD_ENV, None)
        if value is None:
            return None
        if current_task_root is not None:
            raise ValueError("task root is already installed")
        connection = Connection(int(value))
        try:
            os.fstat(connection.fileno())
            root = cls(connection)
        except BaseException:
            connection.close()
            raise
        current_task_root = root
        return root

    def register_scope(self) -> TaskCleanupScope:
        """Retain an actual owner's proof before attempting resource creation.

        Raises:
            TaskCancelled: Retirement has sealed further scope creation.
        """

        with self.condition:
            if self.sealed:
                raise TaskCancelled("task retirement prohibits further resource creation")
            scope = TaskCleanupScope(self)
            self.scopes.add(scope)
            return scope

    def activate(self) -> None:
        """Require supervisor-committed protection before resource startup.

        Activation is sticky through the complete invocation. Its enclosing
        owner has already retained its cleanup scope. Aggregate execution timing
        continues at supervision; this handshake adds no independent timeout.

        Raises:
            TaskCancelled: Cancellation precedes resource startup.
            OSError: The control channel fails before acknowledgement.
            RuntimeError: The control reader fails or rejects a message.
        """

        with self.condition:
            if self.sealed:
                raise TaskCancelled("task is retiring")
            if not self.scopes:
                raise RuntimeError("task activation requires a retained owner cleanup scope")
            if self.requested_event is None:
                self.requested_event = TaskResourceEvent.ACTIVE
                self.connection.send(TaskRootUpdate(self.identity, TaskResourceEvent.ACTIVE))
            while self.acknowledged is not TaskResourceEvent.ACTIVE:
                if self.sealed:
                    raise TaskCancelled("task is retiring")
                self.reader.raise_if_failed()
                self.condition.wait(PROCESS_POLL_INTERVAL_SECONDS)
            if self.sealed:
                raise TaskCancelled("task is retiring")

    def handle_cancel(self, signum: int, frame: FrameType | None) -> None:
        """Deliver one interruption, leaving later signals out of cleanup."""

        if self.consume_cancellation():
            raise TaskCancelled(f"task cancelled by signal {signum}")

    def consume_cancellation(self) -> bool:
        """Seal resource creation and consume the first cancellation delivery.

        A program's existing signal adapter calls this before interrupting its
        own execution, including asyncio task cancellation. A queued control
        request has already installed its deadline. Direct signals establish
        the first deadline here. ``False`` leaves later signals out of cleanup.
        """

        with self.condition:
            if self.sealed and not self.interrupt_pending:
                return False
            self.interrupt_pending = False
            self.sealed = True
            if self.cleanup_deadline is None:
                self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            return True

    def read_control(self, stop_event: threading.Event) -> None:
        """Receive commits and wake the program once for cooperative retirement."""

        while not stop_event.is_set():
            if not self.connection.poll(PROCESS_POLL_INTERVAL_SECONDS):
                continue
            try:
                message = self.connection.recv()
            except (EOFError, OSError):
                with self.condition:
                    interrupt = self.requested_event is not None and not self.sealed
                    self.sealed = True
                    self.interrupt_pending = self.interrupt_pending or interrupt
                    if self.cleanup_deadline is None:
                        self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
                if interrupt:
                    os.kill(self.identity.pid, signal.SIGINT)
                raise
            if isinstance(message, TaskRootAcknowledged):
                with self.condition:
                    if message.event is not self.requested_event:
                        raise RuntimeError("task root received an unexpected acknowledgement")
                    self.acknowledged = message.event
                    self.condition.notify_all()
            elif isinstance(message, TaskCancellation):
                with self.condition:
                    interrupt = not self.sealed
                    self.sealed = True
                    if self.cleanup_deadline is None:
                        self.cleanup_deadline = message.deadline
                    else:
                        self.cleanup_deadline = min(self.cleanup_deadline, message.deadline)
                    self.interrupt_pending = self.interrupt_pending or interrupt
                    self.condition.notify_all()
                if interrupt:
                    # A real process-local signal also wakes asyncio's signal
                    # fd; no client, descendant or process group is signalled.
                    os.kill(self.identity.pid, signal.SIGINT)
            else:
                raise RuntimeError(f"task root received invalid control message {message!r}")

    def finish(self) -> None:
        """Seal after full teardown and commit CLEANED only with every proof.

        Failed cleanup retains its scope. Ordinary expiry or channel failure
        retains this owner for manual resolution under the original deadline.
        This operation never calls resource cleanup or infers it from root exit.
        """

        global current_task_root
        with self.condition:
            self.sealed = True
            self.interrupt_pending = False
            for signum in self.handlers:
                signal.signal(signum, self.handle_cancel)
            if self.cleanup_deadline is None:
                self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            deadline = self.cleanup_deadline
            if self.requested_event is not None:
                try:
                    self.connection.send(TaskCancellation(deadline))
                except (BrokenPipeError, EOFError, OSError) as error:
                    logger.error("task retirement channel unavailable pid=%s detail=%s", self.identity.pid, error)
            expiry_reported = False
            while self.scopes:
                if time.monotonic() >= deadline and not expiry_reported:
                    logger.error(
                        "task cleanup expired; retaining owner pid=%s; manual resolution required", self.identity.pid
                    )
                    expiry_reported = True
                self.condition.wait(PROCESS_POLL_INTERVAL_SECONDS)
            if self.requested_event is not None:
                sent = False
                failure_reported = False
                while self.acknowledged is not TaskResourceEvent.CLEANED:
                    try:
                        self.reader.raise_if_failed()
                        if not sent:
                            self.requested_event = TaskResourceEvent.CLEANED
                            self.connection.send(TaskRootUpdate(self.identity, TaskResourceEvent.CLEANED))
                            sent = True
                    except Exception as error:
                        if not failure_reported:
                            logger.error(
                                "task cleanup proof unacknowledged; retaining owner pid=%s detail=%s",
                                self.identity.pid,
                                error,
                            )
                            failure_reported = True
                    self.condition.wait(PROCESS_POLL_INTERVAL_SECONDS)
        self.reader.close()
        self.connection.close()
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)
        if current_task_root is self:
            current_task_root = None


def get_task_root() -> TaskRoot | None:
    """Return this worker's invocation owner, absent during direct execution."""

    return current_task_root
