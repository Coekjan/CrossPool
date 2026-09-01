"""CUDA-free GPU inventory normalization and in-run task leases."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

NVIDIA_SMI_COMMAND = "nvidia-smi"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


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
    """Visible physical GPUs with deterministic task-local leasing."""

    def __init__(self, uuids: tuple[str, ...]) -> None:
        self.uuids = uuids
        self.available_uuids = list(uuids)
        self.active_leases: set[GpuLease] = set()
        self.closed = False

    @classmethod
    def from_environment(cls) -> GpuPool:
        """Normalize the explicitly visible, externally exclusive GPU pool."""

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

        return cls(tuple(normalized))

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
        """Return one active task allocation to this pool."""

        if lease not in self.active_leases:
            raise RuntimeError("GPU lease is not active in this pool")
        self.active_leases.remove(lease)
        leased = set(lease.uuids)
        self.available_uuids = [uuid for uuid in self.uuids if uuid in leased or uuid in self.available_uuids]

    def close(self) -> None:
        """Close this pool after every task lease is returned."""

        if self.active_leases:
            raise RuntimeError("cannot close GPU pool while task leases remain active")
        if self.closed:
            return
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


def query_physical_gpu_links() -> dict[tuple[str, str], str]:
    """Return directed physical-GPU link tokens keyed by UUID pair."""

    gpu_by_index = query_physical_gpus()
    try:
        result = subprocess.run(
            [NVIDIA_SMI_COMMAND, "topo", "-m"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to query physical GPU links with {NVIDIA_SMI_COMMAND}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"failed to query physical GPU links with {NVIDIA_SMI_COMMAND}: {detail}")

    rows = tuple(ANSI_ESCAPE_PATTERN.sub("", line).split() for line in result.stdout.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned no topology rows")
    gpu_columns = tuple(value for value in rows[0] if value.startswith("GPU") and value[3:].isdigit())
    expected_columns = tuple(f"GPU{index}" for index in range(len(gpu_by_index)))
    if gpu_columns != expected_columns:
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned invalid topology columns: {gpu_columns}")

    tokens_by_row = {row[0]: row[1 : len(gpu_columns) + 1] for row in rows[1:] if row[0] in gpu_columns}
    if set(tokens_by_row) != set(gpu_columns) or any(
        len(tokens) != len(gpu_columns) for tokens in tokens_by_row.values()
    ):
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned an incomplete physical GPU topology matrix")
    links: dict[tuple[str, str], str] = {}
    for source_index, source in enumerate(gpu_columns):
        for destination_index, destination in enumerate(gpu_columns):
            if source_index == destination_index:
                continue
            source_uuid = gpu_by_index[str(source_index)]
            destination_uuid = gpu_by_index[str(destination_index)]
            token = tokens_by_row[source][destination_index]
            if not token or token == "X":
                raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned invalid link {source}->{destination}: {token!r}")
            links[(source_uuid, destination_uuid)] = token
    return links
