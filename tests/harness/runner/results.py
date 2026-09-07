"""Concurrent-safe lifecycle for durable test-run result directories."""

from __future__ import annotations

import fcntl
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

RUN_ID_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]+-[0-9]+")
RUN_LOCK_NAME = ".run.lock"
CLEANUP_LOCK_NAME = ".cleanup.lock"


@dataclass(frozen=True, slots=True)
class TestResultCleanup:
    """One explicit cleanup decision over test-result entries."""

    removable: tuple[Path, ...]
    retained: tuple[Path, ...]
    active: tuple[Path, ...]


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
    """Own concurrent-safe test-result creation and explicit cleanup."""

    root: Path

    def start(self, run_id: str) -> TestRun:
        """Create and exclusively lock one validated run directory."""

        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"invalid test run id: {run_id!r}")
        root = self.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / CLEANUP_LOCK_NAME).open("a+b") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            directory = root / run_id
            directory.mkdir(exist_ok=False)
            lock_file = (directory / RUN_LOCK_NAME).open("a+b")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                lock_file.close()
                shutil.rmtree(directory)
                raise
        return TestRun(directory, lock_file)

    def cleanup(self, *, keep_runs: int, dry_run: bool = False) -> TestResultCleanup:
        """Select or remove inactive entries below one resolved result root."""

        if keep_runs < 0:
            raise ValueError("keep_runs must be non-negative")
        root = self.root.resolve()
        if not root.exists():
            return TestResultCleanup((), (), ())

        with (root / CLEANUP_LOCK_NAME).open("a+b") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            inactive: list[Path] = []
            active: list[Path] = []
            acquired_run_locks: list[BinaryIO] = []
            try:
                for entry in root.iterdir():
                    if entry.name == CLEANUP_LOCK_NAME:
                        continue
                    if entry.is_dir() and not entry.is_symlink():
                        run_lock_path = entry / RUN_LOCK_NAME
                        if run_lock_path.is_file() and not run_lock_path.is_symlink():
                            run_lock = run_lock_path.open("a+b")
                            try:
                                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                run_lock.close()
                                active.append(entry)
                                continue
                            acquired_run_locks.append(run_lock)
                    inactive.append(entry)

                newest_first = sorted(inactive, key=lambda path: (path.lstat().st_mtime_ns, path.name), reverse=True)
                retained = tuple(newest_first[:keep_runs])
                removable = tuple(newest_first[keep_runs:])
                if not dry_run:
                    for entry in removable:
                        if entry.is_symlink() or not entry.is_dir():
                            entry.unlink()
                        else:
                            shutil.rmtree(entry)
                return TestResultCleanup(
                    removable=removable,
                    retained=retained,
                    active=tuple(sorted(active, key=lambda path: path.name)),
                )
            finally:
                for run_lock in acquired_run_locks:
                    run_lock.close()
