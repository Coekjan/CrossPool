"""CUDA-free GPU task metadata, UUID normalization, and whole-run leases."""

from __future__ import annotations

import fcntl
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

GPU_LOCK_DIRECTORY = Path("/tmp/xpool-test-gpu-locks")
NVIDIA_SMI_COMMAND = "nvidia-smi"


@dataclass(frozen=True, slots=True)
class GpuLease:
    """Ordered physical GPU UUIDs exclusively held by one running task."""

    uuids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.uuids:
            raise ValueError("GPU lease must contain at least one UUID")
        if len(self.uuids) != len(set(self.uuids)):
            raise ValueError("GPU lease UUIDs must be unique")


class GpuPool:
    """Whole-run locked physical GPUs with deterministic task-local leasing."""

    def __init__(self, uuids: tuple[str, ...], lock_files: tuple[TextIO, ...]) -> None:
        self.uuids = uuids
        self.available_uuids = list(uuids)
        self.active_leases: set[GpuLease] = set()
        self.lock_files = lock_files
        self.closed = False

    @classmethod
    def from_environment(cls) -> GpuPool:
        """Normalize explicit CUDA visibility and lock every eligible GPU."""

        raw_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw_visibility is None or not raw_visibility.strip():
            raise RuntimeError("GPU runner requires an explicit nonempty CUDA_VISIBLE_DEVICES")
        entries = tuple(entry.strip() for entry in raw_visibility.split(","))
        if any(not entry for entry in entries):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains an empty GPU entry")
        if any(entry.startswith("MIG-") or "/MIG-" in entry for entry in entries):
            raise RuntimeError("GPU runner does not accept MIG identifiers")

        gpu_by_index = query_physical_gpus()
        known_uuids = frozenset(gpu_by_index.values())
        normalized: list[str] = []
        unknown: list[str] = []
        for entry in entries:
            if entry in gpu_by_index:
                normalized.append(gpu_by_index[entry])
            elif entry in known_uuids:
                normalized.append(entry)
            else:
                unknown.append(entry)
        if unknown:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES contains unknown physical GPUs: {tuple(unknown)}")
        if len(normalized) != len(set(normalized)):
            raise RuntimeError("CUDA_VISIBLE_DEVICES resolves to duplicate physical GPU UUIDs")

        lock_files = acquire_gpu_locks(tuple(normalized))
        return cls(tuple(normalized), lock_files)

    @property
    def available_count(self) -> int:
        """Return the number of GPUs not leased to active tasks."""

        return len(self.available_uuids)

    def try_lease(self, count: int) -> GpuLease | None:
        """Lease the first available GPUs in user-declared order when possible."""

        if self.closed:
            raise RuntimeError("GPU pool is closed")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("GPU lease count must be a positive integer")
        if count > self.available_count:
            return None
        selected = tuple(self.available_uuids[:count])
        del self.available_uuids[:count]
        lease = GpuLease(selected)
        self.active_leases.add(lease)
        return lease

    def release(self, lease: GpuLease) -> None:
        """Return one active task allocation without releasing whole-run locks."""

        if lease not in self.active_leases:
            raise RuntimeError("GPU lease is not active in this pool")
        self.active_leases.remove(lease)
        leased = set(lease.uuids)
        self.available_uuids = [uuid for uuid in self.uuids if uuid in leased or uuid in self.available_uuids]

    def close(self) -> None:
        """Release whole-run advisory locks after every task lease is returned."""

        if self.active_leases:
            raise RuntimeError("cannot close GPU pool while task leases remain active")
        if self.closed:
            return
        for lock_file in reversed(self.lock_files):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        self.closed = True


def query_physical_gpus() -> dict[str, str]:
    """Query host ordinal-to-UUID mappings once without importing CUDA libraries."""

    try:
        result = subprocess.run(
            [NVIDIA_SMI_COMMAND, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"failed to query physical GPUs with {NVIDIA_SMI_COMMAND}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"failed to query physical GPUs with {NVIDIA_SMI_COMMAND}: {detail}")

    gpu_by_index: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned invalid physical GPU row: {line!r}")
        index, uuid = fields
        if index in gpu_by_index or uuid in gpu_by_index.values():
            raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned duplicate physical GPU row: {line!r}")
        gpu_by_index[index] = uuid
    if not gpu_by_index:
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned no physical GPUs")
    return gpu_by_index


def acquire_gpu_locks(uuids: tuple[str, ...]) -> tuple[TextIO, ...]:
    """Acquire nonblocking whole-run advisory locks in stable UUID order."""

    GPU_LOCK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock_files: list[TextIO] = []
    conflicts: list[str] = []
    for uuid in sorted(uuids):
        lock_file = (GPU_LOCK_DIRECTORY / f"{uuid}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            conflicts.append(uuid)
            lock_file.close()
        else:
            lock_files.append(lock_file)
    if conflicts:
        for lock_file in reversed(lock_files):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        raise RuntimeError(f"physical GPUs are locked by another xpool test run: {tuple(conflicts)}")
    return tuple(lock_files)
