"""Context-free device inventory and tool-local task allocations."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass

from xpool.utils.device import query_uuids_mapping, visible_uuids

NVIDIA_SMI_COMMAND = "nvidia-smi"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class PhysicalDevice:
    """CUDA-free device identity and capacity; memory is in bytes."""

    index: str
    uuid: str
    name: str | None
    pci_bus_id: str | None
    total_memory_bytes: int | None
    driver_version: str | None


@dataclass(frozen=True, slots=True)
class DeviceTopology:
    """Raw directed link tokens and nullable CPU/NUMA affinity by physical UUID."""

    links: dict[tuple[str, str], str]
    affinities: dict[str, tuple[str | None, str | None]]


@dataclass(frozen=True, slots=True, eq=False)
class DeviceLease:
    """Tool-local allocation retained until resource and task-domain cleanup."""

    uuids: tuple[str, ...]


class DevicePool:
    """Allocate eligible devices among tasks within one tool invocation.

    Inventory creates no contexts. The runner returns each allocation only
    after proving its complete resource and task-domain cleanup.
    """

    def __init__(
        self,
        uuids: tuple[str, ...],
        physical_index_by_uuid: dict[str, int],
    ) -> None:
        self.uuids = uuids
        self.physical_index_by_uuid = physical_index_by_uuid
        self.available_uuids = list(uuids)
        self.active_leases: set[DeviceLease] = set()
        self.closed = False

    @classmethod
    def from_environment(cls) -> DevicePool:
        """Resolve explicit deployment visibility for this tool's local allocation."""

        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise RuntimeError("device runner requires an explicit nonempty CUDA_VISIBLE_DEVICES")
        uuids = visible_uuids()
        inventory = query_uuids_mapping()
        return cls(uuids, {uuid: index for index, uuid in inventory.items()})

    @property
    def available_count(self) -> int:
        """Return the number of devices not leased to active tasks."""

        return len(self.available_uuids)

    def try_lease(self, count: int) -> DeviceLease | None:
        """Allocate the first available devices, or return None at capacity."""

        if self.closed:
            raise RuntimeError("device pool is closed")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("device lease count must be a positive integer")
        if count > self.available_count:
            return None
        selected = tuple(self.available_uuids[:count])
        lease = DeviceLease(selected)
        del self.available_uuids[:count]
        self.active_leases.add(lease)
        return lease

    def release(self, lease: DeviceLease) -> None:
        """Return one active task allocation to this pool."""

        if lease not in self.active_leases:
            raise RuntimeError("device lease is not active in this pool")
        self.active_leases.remove(lease)
        leased = set(lease.uuids)
        self.available_uuids = [uuid for uuid in self.uuids if uuid in leased or uuid in self.available_uuids]

    def close(self) -> None:
        """Close this pool after every task lease is returned."""

        if self.active_leases:
            raise RuntimeError("cannot close device pool while task leases remain active")
        if self.closed:
            return
        self.closed = True


def query_visible_device_total_memory_bytes() -> tuple[int, ...]:
    """Return selected capacities in deployment order; unknown capacity raises RuntimeError."""

    memory_by_uuid = {device.uuid: device.total_memory_bytes for device in query_physical_device_details()}
    capacities = []
    for uuid in visible_uuids():
        capacity = memory_by_uuid[uuid]
        if capacity is None:
            raise RuntimeError(f"physical device capacity is unavailable: {uuid}")
        capacities.append(capacity)
    return tuple(capacities)


def query_physical_device_details() -> tuple[PhysicalDevice, ...]:
    """Observe identity/capacity with a ten-second bound, without initializing CUDA.

    Unavailable optional nvidia-smi fields remain unknown. Command, identity or
    capacity syntax failures raise RuntimeError for the observation owner.
    """

    try:
        result = subprocess.run(
            [
                NVIDIA_SMI_COMMAND,
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to query physical device details: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(f"failed to query physical device details: {result.stderr.strip() or result.stdout.strip()}")
    devices = []
    indices: set[str] = set()
    uuids: set[str] = set()
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        values = [value.strip() for value in row]
        if len(values) != 6 or not values[0].isdigit() or not values[1].startswith("GPU-"):
            raise RuntimeError(f"invalid physical device detail row: {row!r}")
        index, uuid = values[:2]
        if index in indices or uuid in uuids:
            raise RuntimeError(f"duplicate physical device detail row: {row!r}")
        indices.add(index)
        uuids.add(uuid)
        name, pci, memory, driver = (None if value in {"", "N/A", "[N/A]"} else value for value in values[2:])
        if memory is not None and (not memory.isdigit() or int(memory) <= 0):
            raise RuntimeError(f"invalid physical device capacity: {row!r}")
        devices.append(PhysicalDevice(index, uuid, name, pci, int(memory) * 1024 * 1024 if memory else None, driver))
    if not devices:
        raise RuntimeError("physical device details returned no devices")
    return tuple(devices)


def query_physical_device_topology(device_by_index: dict[str, str]) -> DeviceTopology:
    """Observe device links/affinity for a validated physical index map, bounded to ten seconds."""
    try:
        result = subprocess.run(
            [NVIDIA_SMI_COMMAND, "topo", "-m"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to query physical device links with {NVIDIA_SMI_COMMAND}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"failed to query physical device links with {NVIDIA_SMI_COMMAND}: {detail}")

    rows = tuple(ANSI_ESCAPE_PATTERN.sub("", line).split() for line in result.stdout.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned no topology rows")
    device_columns = tuple(value for value in rows[0] if value.startswith("GPU") and value[3:].isdigit())
    expected_columns = tuple(f"GPU{index}" for index in device_by_index)
    if device_columns != expected_columns:
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned invalid topology columns: {device_columns}")

    tokens_by_row = {row[0]: row[1 : len(device_columns) + 1] for row in rows[1:] if row[0] in device_columns}
    if set(tokens_by_row) != set(device_columns) or any(
        len(tokens) != len(device_columns) for tokens in tokens_by_row.values()
    ):
        raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned an incomplete physical device topology matrix")
    links: dict[tuple[str, str], str] = {}
    header = " ".join(rows[0])
    matrix_columns = tuple(value for value in rows[0] if re.fullmatch(r"(?:GPU|NIC)\d+", value))
    affinities: dict[str, tuple[str | None, str | None]] = {}
    for row in rows[1:]:
        if row[0] not in device_columns:
            continue
        tail = row[len(matrix_columns) + 1 :]
        cpu = tail[0] if "CPU Affinity" in header and tail else None
        numa = tail[1] if "NUMA Affinity" in header and len(tail) > 1 else None
        affinities[device_by_index[row[0][3:]]] = (
            None if cpu in {None, "N/A", "[N/A]"} else cpu,
            None if numa in {None, "N/A", "[N/A]"} else numa,
        )
    for source_index, source in enumerate(device_columns):
        for destination_index, destination in enumerate(device_columns):
            if source_index == destination_index:
                continue
            source_uuid = device_by_index[source[3:]]
            destination_uuid = device_by_index[destination[3:]]
            token = tokens_by_row[source][destination_index]
            if not token or token == "X":
                raise RuntimeError(f"{NVIDIA_SMI_COMMAND} returned invalid link {source}->{destination}: {token!r}")
            links[(source_uuid, destination_uuid)] = token
    return DeviceTopology(links, affinities)
