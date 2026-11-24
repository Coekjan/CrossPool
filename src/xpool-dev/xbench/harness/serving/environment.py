"""Record server startup and device details for benchmark reports."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from xbench.harness.serving.case import BenchValue, ServingEndpoint
from xkit.device import query_physical_device_details, query_physical_device_topology
from xkit.results import write_json
from xpool.model import ModelId


class ServingDevice(BenchValue):
    uuid: str = Field(min_length=1)
    name: str | None = None
    total_memory_bytes: int | None = Field(default=None, gt=0)
    pci_bus_id: str | None = None
    cpu_affinity: str | None = None
    numa_affinity: str | None = None


class ServingDeviceLink(BenchValue):
    source_uuid: str
    destination_uuid: str
    link: str


class ServingSoftware(BenchValue):
    python: str
    prefix: str
    packages: dict[str, str]


class ServingCommand(BenchValue):
    """Launch arguments, selected environment values, and available software versions.

    Hosted launchers choose the environment values in their command records.
    """

    argv: tuple[str, ...] = Field(min_length=1)
    environment: dict[str, str]
    software: ServingSoftware | None = None


class HostedLaunchRecord(BenchValue):
    """Diagnostic command snapshot written by the launcher before starting its services."""

    commands: tuple[ServingCommand, ...] = Field(min_length=1)


class ServingEnvironment(BenchValue):
    """Server endpoints, launch commands and device details for a managed attempt."""

    endpoints: dict[ModelId, ServingEndpoint]
    commands: tuple[ServingCommand, ...] | None
    devices: tuple[ServingDevice, ...]
    links: tuple[ServingDeviceLink, ...] | None = None
    role_device_uuids: dict[Literal["atn", "ffn"], tuple[str, ...]] | None = None
    driver_version: str | None = None
    runtime_config: Path | None = None


ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "LD_LIBRARY_PATH",
    "SGLANG_PLUGINS",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


def package_versions(names: tuple[str, ...], errors: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
            errors.append(f"package version unavailable: {name}: {error}")
    return versions


def capture_serving_environment(
    endpoints: dict[ModelId, ServingEndpoint],
    uuids: tuple[str, ...],
    *,
    commands: tuple[ServingCommand, ...] | None,
    errors: list[str],
    role_device_uuids: dict[Literal["atn", "ffn"], tuple[str, ...]] | None = None,
    runtime_config: Path | None = None,
) -> ServingEnvironment:
    """Record details for the assigned devices before measurement starts.

    If a query fails, keep the device UUIDs and any details already collected,
    and append the error to `errors`. Device queries run without initializing
    CUDA or changing device visibility.
    """

    devices = {uuid: ServingDevice(uuid=uuid) for uuid in uuids}
    links = None
    driver_version = None
    try:
        inventory = query_physical_device_details() if uuids else ()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))
        inventory = ()
    if inventory:
        selected = [device for device in inventory if device.uuid in devices]
        for device in selected:
            devices[device.uuid] = ServingDevice(
                uuid=device.uuid,
                name=device.name,
                total_memory_bytes=device.total_memory_bytes,
                pci_bus_id=device.pci_bus_id,
            )
        if len(selected) != len(uuids):
            errors.append("device details unavailable for part of the assigned lease")
        drivers = {device.driver_version for device in selected}
        driver_version = next(iter(drivers)) if len(drivers) == 1 else None
        try:
            topology = query_physical_device_topology({device.index: device.uuid for device in inventory})
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(str(error))
        else:
            for uuid in uuids:
                cpu, numa = topology.affinities.get(uuid, (None, None))
                devices[uuid] = devices[uuid].model_copy(update={"cpu_affinity": cpu, "numa_affinity": numa})
            links = tuple(
                ServingDeviceLink(source_uuid=source, destination_uuid=destination, link=link)
                for (source, destination), link in topology.links.items()
                if source in devices and destination in devices
            )
    return ServingEnvironment(
        endpoints=endpoints,
        commands=commands,
        runtime_config=runtime_config,
        devices=tuple(devices[uuid] for uuid in uuids),
        links=links,
        role_device_uuids=role_device_uuids,
        driver_version=driver_version,
    )


def write_environment(directory: Path, environment: dict[str, JsonValue]) -> None:
    try:
        write_json(directory / "environment.json", environment)
    except OSError as error:
        # Measurement evidence remains authoritative if the diagnostic disk write fails.
        print(f"xbench environment snapshot unavailable: {error}", file=sys.stderr)
