"""Generic subprocess, process-group, and spawned-child ownership for tests."""

from __future__ import annotations

import multiprocessing
import multiprocessing.process
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Self, TextIO

import psutil

PROCESS_TERMINATE_TIMEOUT_SECONDS = 30.0
PROCESS_KILL_TIMEOUT_SECONDS = 30.0
PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_START_TIMEOUT_SECONDS = 30.0
PROCESS_POLL_INTERVAL_SECONDS = 0.05
LOG_TAIL_CHARS = 12000
REPO_ROOT = Path(__file__).resolve().parents[2]
spawn_environment_lock = threading.Lock()


@dataclass(slots=True)
class OwnedProcess:
    """Own one direct subprocess together with its optional output storage."""

    name: str
    process: subprocess.Popen[str]
    log_path: Path | None = None
    log_file: TextIO | None = None

    @classmethod
    def spawn_captured(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> Self:
        """Start one direct process with captured text output."""

        return cls(
            name=name,
            process=subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )

    @classmethod
    def spawn_logged(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> Self:
        """Start one direct process whose combined output is written to a log."""

        log_file = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(env),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except BaseException:
            log_file.close()
            raise
        return cls(name=name, process=process, log_path=log_path, log_file=log_file)

    def collect_output_after_timeout(self) -> tuple[str, str, str]:
        """Drain captured output, closing stuck pipes after the deadline."""

        try:
            stdout, stderr = self.process.communicate(timeout=PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            self.close_pipes()
            return (
                self.output_text(error.output),
                self.output_text(error.stderr),
                "stdout/stderr pipes did not close after cleanup",
            )
        return stdout, stderr, "stdout/stderr drained after cleanup"

    def tail(self) -> str:
        """Return the diagnostic tail of the owned log file."""

        if self.log_path is None:
            return "<process has no log file>"
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.flush()
        try:
            return self.log_path.read_text(encoding="utf-8")[-LOG_TAIL_CHARS:]
        except OSError as error:
            return f"<failed to read {self.log_path}: {error}>"

    def close(self) -> None:
        """Close owned pipes and log storage after the process is reaped."""

        self.close_pipes()
        if self.log_file is not None:
            self.log_file.close()

    def close_pipes(self) -> None:
        """Close captured stdout and stderr pipes when present."""

        for pipe in (self.process.stdout, self.process.stderr):
            if pipe is not None:
                with suppress(OSError, ValueError):
                    pipe.close()

    @staticmethod
    def output_text(output: str | bytes | None) -> str:
        """Normalize partial ``TimeoutExpired`` output to text."""

        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output


@dataclass(slots=True)
class OwnedProcessGroup(OwnedProcess):
    """Own one process together with its dedicated POSIX process group."""

    @classmethod
    def spawn_captured(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> Self:
        """Start one dedicated process group with captured text output."""

        return cls(
            name=name,
            process=subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True,
            ),
        )

    @classmethod
    def spawn_logged(
        cls,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> Self:
        """Start one dedicated process group with combined logged output."""

        log_file = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(env),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        except BaseException:
            log_file.close()
            raise
        return cls(name=name, process=process, log_path=log_path, log_file=log_file)

    def terminate(self) -> None:
        """Terminate every live group member and reap the direct leader."""

        signal_process_group(self.process.pid, signal.SIGTERM)
        if wait_for_process_group(self.process, PROCESS_TERMINATE_TIMEOUT_SECONDS):
            return
        signal_process_group(self.process.pid, signal.SIGKILL)
        if wait_for_process_group(self.process, PROCESS_KILL_TIMEOUT_SECONDS):
            return
        raise RuntimeError(f"{self.name} process group {self.process.pid} retained live members after SIGKILL")


@dataclass(frozen=True, slots=True)
class SpawnedProcessStarted:
    """Child acknowledgement emitted after process-group and log setup."""


@dataclass(frozen=True, slots=True)
class SpawnedProcessFailure:
    """Exception traceback sent by a spawned child before nonzero exit."""

    traceback: str


@dataclass(slots=True)
class SpawnedProcess:
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
            target=run_spawned_process,
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
            owned.receive(SpawnedProcessStarted, timeout_seconds=TASK_SUPERVISOR_START_TIMEOUT_SECONDS)
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
        if isinstance(message, SpawnedProcessFailure):
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
            if isinstance(message, SpawnedProcessFailure):
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
        wait_for_spawned_processes(active, term_deadline)
        survivors = tuple(process for process in active if process.process.is_alive())
        for process in survivors:
            process_id = process.process.pid
            if process_id is None:
                raise RuntimeError(f"spawned process {process.name} has no process id")
            signal_process_group(process_id, signal.SIGKILL)
        kill_deadline = time.monotonic() + PROCESS_KILL_TIMEOUT_SECONDS
        wait_for_spawned_processes(survivors, kill_deadline)
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


def run_spawned_process[T](
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
            connection.send(SpawnedProcessStarted())
            target(connection, spec)
    except BaseException:
        failure = SpawnedProcessFailure(traceback.format_exc())
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(failure)
        raise
    finally:
        connection.close()


def wait_for_spawned_processes(processes: Sequence[SpawnedProcess], deadline: float) -> None:
    """Poll a process set until every child exits or one shared deadline expires."""

    pending = list(processes)
    while pending and time.monotonic() < deadline:
        for process in tuple(pending):
            process.process.join(0)
            if not process.process.is_alive():
                pending.remove(process)
        if pending:
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def signal_process_group(process_group: int, signal_number: int) -> None:
    """Send a signal to a process group when it still exists."""

    with suppress(ProcessLookupError):
        os.killpg(process_group, signal_number)


def wait_for_process_group(process: subprocess.Popen[str], timeout_seconds: float) -> bool:
    """Wait until no live group member remains and reap the direct leader."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()
        if process.returncode is not None and not process_group_has_live_members(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))


def process_group_has_live_members(process_group: int) -> bool:
    """Return whether one process group retains a non-zombie member."""

    for process in psutil.process_iter(("status",)):
        try:
            if os.getpgid(process.pid) == process_group and process.info["status"] != psutil.STATUS_ZOMBIE:
                return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (PermissionError, psutil.AccessDenied):
            return True
    return False
