"""Concurrent-safe lifecycle for durable test-run result directories."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

RUN_ID_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]+-[0-9]+")
RETENTION_ENVIRONMENT_VARIABLE = "XPOOL_TEST_KEEP_RUNS"


@dataclass(frozen=True, slots=True)
class TestResultRetention:
    """Optional count of test runs retained on disk."""

    keep_runs: int

    @classmethod
    def from_environment(cls) -> TestResultRetention | None:
        """Parse optional retention from the test-runner environment."""

        value = os.environ.get(RETENTION_ENVIRONMENT_VARIABLE)
        if value is None:
            return None
        try:
            keep_runs = int(value)
        except ValueError as error:
            raise ValueError(f"{RETENTION_ENVIRONMENT_VARIABLE} must be a positive integer") from error
        if keep_runs <= 0:
            raise ValueError(f"{RETENTION_ENVIRONMENT_VARIABLE} must be a positive integer")
        return cls(keep_runs)


@dataclass(slots=True)
class TestRun:
    """One live result directory protected from concurrent retention cleanup."""

    directory: Path
    lock_file: BinaryIO
    closed: bool = False

    def complete(self) -> None:
        """Mark the run complete and release its exclusive lifecycle lock."""

        if self.closed:
            return
        try:
            (self.directory / ".completed").touch(exist_ok=False)
        finally:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            self.closed = True


@dataclass(frozen=True, slots=True)
class TestResultStore:
    """Own result-directory creation and optional inactive-run retention."""

    root: Path
    retention: TestResultRetention | None

    @classmethod
    def from_environment(cls, root: Path) -> TestResultStore:
        """Construct a store using the optional runner retention setting."""

        return cls(root, TestResultRetention.from_environment())

    def start(self, run_id: str) -> TestRun:
        """Create and exclusively lock one validated run directory."""

        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"invalid test run id: {run_id!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / run_id
        directory.mkdir(exist_ok=False)
        lock_file = (directory / ".run.lock").open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock_file.close()
            raise
        return TestRun(directory, lock_file)

    def cleanup(self) -> None:
        """Best-effort delete inactive runs older than the retained suffix."""

        if self.retention is None:
            return
        try:
            with (self.root / ".cleanup.lock").open("a+b") as cleanup_lock:
                fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
                runs = sorted(
                    self.run_directories(),
                    key=lambda path: ((path / ".run.lock").stat().st_mtime_ns, path.name),
                    reverse=True,
                )
                for directory in runs[self.retention.keep_runs :]:
                    self.remove_inactive(directory)
        except OSError as error:
            warnings.warn(f"failed to clean old xpool test results: {error}", RuntimeWarning, stacklevel=2)

    def run_directories(self) -> tuple[Path, ...]:
        """Return validated run directories that may be inactive."""

        result: list[Path] = []
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or RUN_ID_PATTERN.fullmatch(directory.name) is None:
                continue
            lock = directory / ".run.lock"
            if lock.is_file() and not lock.is_symlink():
                result.append(directory)
        return tuple(result)

    @staticmethod
    def remove_inactive(directory: Path) -> None:
        """Delete one run directory only after proving it is inactive."""

        lock_path = directory / ".run.lock"
        with lock_path.open("a+b") as run_lock:
            try:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            shutil.rmtree(directory)
