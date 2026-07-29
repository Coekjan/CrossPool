"""Generic subprocess, process-group, and spawned-child ownership for tests."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TextIO

import psutil

PROCESS_TERMINATE_TIMEOUT_SECONDS = 30.0
PROCESS_KILL_TIMEOUT_SECONDS = 30.0
PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS = 30.0
TASK_SUPERVISOR_START_TIMEOUT_SECONDS = 30.0
PROCESS_POLL_INTERVAL_SECONDS = 0.05
LOG_TAIL_CHARS = 12000
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
