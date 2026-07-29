"""Typed Python child-process ownership for the test harness."""

from __future__ import annotations

import multiprocessing
import multiprocessing.process
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Self

from tests.harness.runner.process import (
    LOG_TAIL_CHARS,
    PROCESS_KILL_TIMEOUT_SECONDS,
    PROCESS_POLL_INTERVAL_SECONDS,
    PROCESS_TERMINATE_TIMEOUT_SECONDS,
    signal_process_group,
)

TASK_SUPERVISOR_START_TIMEOUT_SECONDS = 30.0
REPO_ROOT = Path(__file__).resolve().parents[3]
spawn_environment_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class PythonChildStarted:
    """Child acknowledgement emitted after process-group and log setup."""


@dataclass(frozen=True, slots=True)
class PythonChildFailure:
    """Exception traceback sent by a spawned child before nonzero exit."""

    traceback: str


@dataclass(slots=True)
class PythonChildProcess:
    """Own one typed-Pipe child started with a fresh Python interpreter."""

    name: str
    process: multiprocessing.process.BaseProcess
    connection: Connection
    log_path: Path
    closed: bool = False

    @classmethod
    def start[T](
        cls,
        name: str,
        target: Callable[[Connection, T], None],
        spec: T,
        *,
        log_path: Path,
    ) -> Self:
        """Start one process-group child and wait for typed bootstrap."""

        log_path.parent.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            name=f"xpool-test-child:{name}",
            target=run_python_child,
            args=(child_connection, target, spec, log_path),
        )
        try:
            start_spawn_process(process)
        except BaseException:
            parent_connection.close()
            child_connection.close()
            raise
        child_connection.close()
        owned = cls(name=name, process=process, connection=parent_connection, log_path=log_path)
        try:
            owned.receive(PythonChildStarted, timeout_seconds=TASK_SUPERVISOR_START_TIMEOUT_SECONDS)
        except BaseException:
            cls.terminate_all((owned,))
            owned.close()
            raise
        return owned

    def send(self, message: object) -> None:
        """Send one typed subsystem command to the live child."""

        if self.closed or not self.process.is_alive():
            raise RuntimeError(f"spawned process {self.name} is not live")
        try:
            self.connection.send(message)
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError(f"spawned process {self.name} command channel failed: {error}") from error

    def receive[T](self, expected: type[T], *, timeout_seconds: float) -> T:
        """Receive one expected typed message under a bounded deadline."""

        if timeout_seconds <= 0:
            raise ValueError("spawned process receive timeout_seconds must be positive")
        if not self.connection.poll(timeout_seconds):
            if not self.process.is_alive():
                self.process.join(0)
                raise RuntimeError(
                    f"spawned process {self.name} exited with code {self.process.exitcode} "
                    f"before replying\n{self.tail()}"
                )
            raise RuntimeError(f"spawned process {self.name} timed out waiting for {expected.__name__}")
        try:
            message = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"spawned process {self.name} response channel failed: {error}\n{self.tail()}"
            ) from error
        if isinstance(message, PythonChildFailure):
            raise RuntimeError(f"spawned process {self.name} failed:\n{message.traceback}\n{self.tail()}")
        if not isinstance(message, expected):
            raise RuntimeError(
                f"spawned process {self.name} sent {type(message).__name__}, expected {expected.__name__}"
            )
        return message

    def wait(self, *, timeout_seconds: float) -> None:
        """Require normal child exit under a bounded deadline."""

        if timeout_seconds <= 0:
            raise ValueError("spawned process wait timeout_seconds must be positive")
        self.process.join(timeout_seconds)
        if self.process.is_alive():
            raise RuntimeError(f"spawned process {self.name} did not exit within {timeout_seconds}s")
        if self.process.exitcode == 0:
            return
        if self.connection.poll():
            try:
                message = self.connection.recv()
            except (EOFError, OSError):
                message = None
            if isinstance(message, PythonChildFailure):
                raise RuntimeError(f"spawned process {self.name} failed:\n{message.traceback}\n{self.tail()}")
        raise RuntimeError(f"spawned process {self.name} exited with code {self.process.exitcode}\n{self.tail()}")

    @classmethod
    def terminate_all(cls, processes: Sequence[Self]) -> None:
        """Terminate process groups concurrently under shared TERM/KILL deadlines."""

        active = tuple(process for process in processes if process.process.is_alive())
        for process in active:
            process_id = process.process.pid
            if process_id is None:
                raise RuntimeError(f"spawned process {process.name} has no process id")
            signal_process_group(process_id, signal.SIGTERM)
        term_deadline = time.monotonic() + PROCESS_TERMINATE_TIMEOUT_SECONDS
        wait_for_python_children(active, term_deadline)
        survivors = tuple(process for process in active if process.process.is_alive())
        for process in survivors:
            process_id = process.process.pid
            if process_id is None:
                raise RuntimeError(f"spawned process {process.name} has no process id")
            signal_process_group(process_id, signal.SIGKILL)
        kill_deadline = time.monotonic() + PROCESS_KILL_TIMEOUT_SECONDS
        wait_for_python_children(survivors, kill_deadline)
        survivors = tuple(process.name for process in survivors if process.process.is_alive())
        if survivors:
            raise RuntimeError(f"spawned process groups survived SIGKILL: {survivors}")

    def close(self) -> None:
        """Release Pipe and multiprocessing handles after child reap."""

        if self.closed:
            return
        if self.process.is_alive():
            raise RuntimeError(f"cannot close live spawned process {self.name}")
        self.process.join(0)
        self.connection.close()
        self.process.close()
        self.closed = True

    def tail(self) -> str:
        """Return a bounded diagnostic tail from the child log."""

        try:
            return self.log_path.read_text(encoding="utf-8")[-LOG_TAIL_CHARS:]
        except OSError as error:
            return f"<failed to read {self.log_path}: {error}>"


def start_spawn_process(process: multiprocessing.process.BaseProcess) -> None:
    """Start a spawn process with the repository test package importable."""

    with spawn_environment_lock:
        previous_pythonpath = os.environ.get("PYTHONPATH")
        previous_sys_path = list(sys.path)
        pythonpath = os.pathsep.join(
            (str(REPO_ROOT), previous_pythonpath) if previous_pythonpath else (str(REPO_ROOT),)
        )
        os.environ["PYTHONPATH"] = pythonpath
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        try:
            process.start()
        finally:
            sys.path[:] = previous_sys_path
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath


def run_python_child[T](
    connection: Connection,
    target: Callable[[Connection, T], None],
    spec: T,
    log_path: Path,
) -> None:
    """Bootstrap one typed child with isolated logs and failure propagation."""

    try:
        os.setsid()
        with log_path.open("w", encoding="utf-8") as log_file:
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())
            connection.send(PythonChildStarted())
            target(connection, spec)
    except BaseException:
        failure = PythonChildFailure(traceback.format_exc())
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(failure)
        raise
    finally:
        connection.close()


def wait_for_python_children(processes: Sequence[PythonChildProcess], deadline: float) -> None:
    """Poll a process set until every child exits or one shared deadline expires."""

    pending = list(processes)
    while pending and time.monotonic() < deadline:
        for process in tuple(pending):
            process.process.join(0)
            if not process.process.is_alive():
                pending.remove(process)
        if pending:
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
