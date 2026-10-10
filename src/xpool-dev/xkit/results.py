"""Concurrent-safe lifecycle and read protection for durable tool results."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from pydantic import JsonValue

__all__ = ["RunCleanup", "RunEntry", "RunStore", "write_json", "write_jsonl"]

RUN_ID_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]+-[0-9]+")
RUN_LOCK_NAME = ".run.lock"
CLEANUP_LOCK_NAME = ".cleanup.lock"


def write_json(path: Path, value: JsonValue) -> None:
    """Atomically replace one owner-written JSON checkpoint, rejecting NaN values."""

    contents = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def write_jsonl(path: Path, values: Iterable[JsonValue]) -> None:
    """Create a new UTF-8 JSONL file and flush/fsync it after the final record.

    The caller supplies serialized values and an existing parent directory.
    Existing files are preserved; write failures may leave a partial prefix.
    Non-finite JSON numbers are rejected.
    """

    with path.open("x", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


@dataclass(frozen=True, slots=True)
class RunCleanup:
    """One explicit cleanup decision over tool invocation entries."""

    removable: tuple[Path, ...]
    retained: tuple[Path, ...]
    active: tuple[Path, ...]

    def print(self) -> None:
        """Print decisions in remove/keep/active order without changing files."""
        for action, paths in (("remove", self.removable), ("keep", self.retained), ("active", self.active)):
            for path in paths:
                print(f"{action} {path}")


@dataclass(slots=True)
class RunEntry:
    """One live result directory protected from concurrent retention cleanup."""

    directory: Path
    lock_file: BinaryIO
    closed: bool = False

    def reopen(self) -> None:
        """Withdraw a completion marker while holding this entry's exclusive lock."""
        if self.closed:
            raise RuntimeError("cannot reopen a closed run entry")
        (self.directory / ".completed").unlink(missing_ok=True)

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
class RunStore:
    """Own run creation, read protection and explicit retention cleanup."""

    root: Path

    def start(self, run_id: str) -> RunEntry:
        """Create and exclusively lock one validated run directory."""

        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"invalid run id: {run_id!r}")
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
        return RunEntry(directory, lock_file)

    def acquire(self, run_id: str, *, exclusive: bool = True) -> RunEntry:
        """Acquire an existing entry without creating or changing evidence.

        Active runs and linked entry/lock files are rejected. Existing lock files
        are opened without creating or modifying source evidence; cleanup uses
        the same lock ordering as creation. Exclusive access also serializes
        derived-report publication with other readers and writers.
        """

        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"invalid run id: {run_id!r}")
        root = self.root.resolve()
        directory = root / run_id
        cleanup_path = root / CLEANUP_LOCK_NAME
        run_lock_path = directory / RUN_LOCK_NAME
        if directory.is_symlink() or cleanup_path.is_symlink() or run_lock_path.is_symlink():
            raise ValueError("result entries and lock files must not be symbolic links")
        with cleanup_path.open("rb") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            run_lock = run_lock_path.open("rb")
            try:
                fcntl.flock(run_lock.fileno(), (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BaseException:
                run_lock.close()
                raise
        return RunEntry(directory, run_lock)

    @contextmanager
    def read(self, run_id: str, *, exclusive: bool = False) -> Generator[Path]:
        """Protect retained evidence for reading or exclusive report publication."""
        entry = self.acquire(run_id, exclusive=exclusive)
        try:
            yield entry.directory
        finally:
            entry.lock_file.close()

    @contextmanager
    def resume(self, run_id: str) -> Generator[RunEntry]:
        """Hold an existing run exclusively through validation, reopening and execution.

        Acquisition does not change evidence. The caller reopens only after its
        own input validation; abandoning or completing the entry releases protection.
        """
        entry = self.acquire(run_id)
        try:
            yield entry
        finally:
            entry.lock_file.close()

    def cleanup(self, *, keep_runs: int, dry_run: bool = False) -> RunCleanup:
        """Select or remove inactive entries below one resolved result root."""

        if keep_runs < 0:
            raise ValueError("keep_runs must be non-negative")
        root = self.root.resolve()
        if not root.exists():
            return RunCleanup((), (), ())

        with (root / CLEANUP_LOCK_NAME).open("a+b") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            inactive: list[Path] = []
            active: list[Path] = []
            acquired_run_locks: list[BinaryIO] = []
            ages: dict[Path, int] = {}
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
                            if RUN_ID_PATTERN.fullmatch(entry.name) is not None:
                                ages[entry] = run_lock_path.stat().st_mtime_ns
                    inactive.append(entry)

                newest_first = sorted(
                    inactive,
                    key=lambda path: (ages[path] if path in ages else path.lstat().st_mtime_ns, path.name),
                    reverse=True,
                )
                retained = tuple(newest_first[:keep_runs])
                removable = tuple(newest_first[keep_runs:])
                if not dry_run:
                    for entry in removable:
                        if entry.is_symlink() or not entry.is_dir():
                            entry.unlink()
                        else:
                            shutil.rmtree(entry)
                return RunCleanup(
                    removable=removable,
                    retained=retained,
                    active=tuple(sorted(active, key=lambda path: path.name)),
                )
            finally:
                for run_lock in acquired_run_locks:
                    run_lock.close()

    def inactive_runs(self) -> tuple[Path, ...]:
        """Enumerate recognized inactive entries without creating or changing files.

        A later reader reacquires protection; this inventory is not a lease.
        Vanished entries and currently held run locks are omitted.
        """
        if not self.root.is_dir():
            return ()
        inactive = []
        for entry in sorted(self.root.iterdir()):
            if RUN_ID_PATTERN.fullmatch(entry.name) is None or not entry.is_dir() or entry.is_symlink():
                continue
            if not (entry / RUN_LOCK_NAME).is_file() or (entry / RUN_LOCK_NAME).is_symlink():
                continue
            try:
                with self.read(entry.name, exclusive=True) as directory:
                    inactive.append(directory)
            except (BlockingIOError, FileNotFoundError):
                continue
        return tuple(inactive)
