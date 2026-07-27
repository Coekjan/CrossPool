"""Host process identity and lifecycle helpers."""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import NoReturn, Self

import psutil

__all__ = ["ProcUniqId", "bail", "set_process_title"]

PROCESS_KILL_WAIT_S = 1.0


def bail(
    logger: logging.Logger | None = None,
    message: str | None = None,
    *args: object,
    code: int = 1,
) -> NoReturn:
    """Log an optional critical failure and terminate the process with ``os._exit``.

    Args:
        logger: Logger that owns the failing runtime path. When omitted and
            ``message`` is provided, the process utility logger is used.
        message: Optional critical log message format string.
        *args: Positional format arguments for ``message``.
        code: Process exit status.

    Side Effects:
        Terminates the current process immediately without running Python
        cleanup handlers. Use only for fail-closed paths where raising would
        leave an unsafe process alive.
    """

    if message is not None:
        (logger or logging.getLogger(__name__)).critical(message, *args)
    os._exit(code)


def set_process_title(title: str) -> None:
    """Set the current Linux process title without hiding its environment.

    Args:
        title: Complete process title exposed through ``ps`` and ``/proc``.

    Side Effects:
        Sets the setproctitle import option that preserves
        ``/proc/<pid>/environ``, imports its native extension, and replaces the
        current process title.
    """

    # setproctitle reads this option while importing its native extension.
    os.environ["SPT_NOENV"] = "1"
    import setproctitle

    setproctitle.setproctitle(title)


@dataclass(frozen=True, slots=True, init=False)
class ProcUniqId:
    """Stable host process unique id based on pid and process creation time.

    Attributes:
        pid: Host process id captured during construction.
        create_time: Process creation timestamp reported by psutil.
    """

    pid: int
    create_time: float

    def __init__(self, pid: int) -> None:
        """Capture a process unique id from a live host pid."""

        if pid <= 0:
            raise ValueError(f"pid must be positive, got {pid}")
        process = psutil.Process(pid)
        object.__setattr__(self, "pid", process.pid)
        object.__setattr__(self, "create_time", process.create_time())

    @classmethod
    def current(cls) -> Self:
        """Return the unique id for the current host process."""

        return cls(os.getpid())

    def is_alive(self) -> bool:
        """Return whether this exact pid/create-time pair is still alive."""

        try:
            process = psutil.Process(self.pid)
            return process.create_time() == self.create_time and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.Error, ValueError):
            return False

    def terminate_tree(self, *, term_grace_s: float) -> bool:
        """Terminate this process and its currently known children.

        Args:
            term_grace_s: Seconds to wait after SIGTERM before sending SIGKILL.

        Returns:
            Whether this exact root process is no longer alive after the
            termination attempt.

        Side Effects:
            Sends SIGTERM, waits for ``term_grace_s``, then sends SIGKILL to the
            root process and recursive children that still match their captured
            pid/create-time identities.
        """

        if not self.is_alive():
            return True
        children = self.child_process_ids()
        targets = [*children, self]
        for target in targets:
            target.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + term_grace_s
        while time.monotonic() < deadline:
            if not any(target.is_alive() for target in targets):
                return True
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        for target in targets:
            target.send_signal(signal.SIGKILL)
        return not self.is_alive()

    def kill_tree(self) -> bool:
        """Directly kill this exact process tree and report whether it exited.

        Returns:
            Whether every captured process identity is gone after the bounded
            confirmation interval.

        Side Effects:
            Sends SIGKILL to live descendants and then the root process. It
            never sends SIGTERM or runs target cleanup handlers.
        """

        if not self.is_alive():
            return True
        targets = [*self.child_process_ids(), self]
        for target in targets:
            target.send_signal(signal.SIGKILL)
        deadline = time.monotonic() + PROCESS_KILL_WAIT_S
        while any(target.is_alive() for target in targets) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not any(target.is_alive() for target in targets)

    def send_signal(self, sig: signal.Signals) -> None:
        """Send a signal only if this exact process identity remains alive."""

        if not self.is_alive():
            return
        try:
            psutil.Process(self.pid).send_signal(sig)
        except psutil.Error:
            return

    def child_process_ids(self) -> list[ProcUniqId]:
        """Return recursively captured identities for current child processes."""

        try:
            process = psutil.Process(self.pid)
            if ProcUniqId(process.pid) != self:
                return []
            children = process.children()
        except psutil.Error:
            return []
        result: list[ProcUniqId] = []
        for child in children:
            try:
                child_id = ProcUniqId(child.pid)
            except (psutil.Error, ValueError):
                continue
            result.append(child_id)
            result.extend(child_id.child_process_ids())
        return result
