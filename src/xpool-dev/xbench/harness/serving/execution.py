"""Reusable serving warmup and environment observations for benchmark programs."""

from __future__ import annotations

import asyncio
import importlib.metadata
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import httpx
from pydantic import JsonValue

from xbench.harness.serving.api import create_api_adapter
from xbench.harness.serving.case import BenchCase, ServingDevice, ServingDeviceLink, ServingMetadata
from xbench.harness.serving.client import MeasurementRecorder, RequestState, observe_system, send_request
from xbench.harness.serving.workload import PreparedWorkload
from xkit.device import query_physical_device_details, query_physical_device_topology
from xkit.results import write_json
from xpool.model import ModelId

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


async def warmup(
    case: BenchCase,
    workload: PreparedWorkload,
    endpoints: Mapping[ModelId, str],
    directory: Path,
    *,
    check_alive: Callable[[], None] | None,
) -> None:
    """Complete every target's independent warmup before creating measured T0."""

    recorder = MeasurementRecorder(directory / "warmup")
    prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
    targets = {target.model_id: target for target in case.targets}
    origin = time.perf_counter()
    stopped = asyncio.Event()
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False) as client:

            async def dispatch() -> None:
                for request in workload.warmup:
                    now = time.perf_counter() - origin
                    target = targets[request.model_id]
                    adapter = create_api_adapter(target.api)
                    state = RequestState(request, recorder, adapter, enqueued_at=now)
                    state.event("enqueue", now)
                    record = await send_request(
                        client,
                        endpoints[request.model_id],
                        adapter.payload(request, prompts[request.model_id, request.prompt_id], target),
                        state,
                        origin=origin,
                        timeout_seconds=case.request_timeout_seconds,
                    )
                    if record.outcome != "success":
                        raise RuntimeError(f"{request.model_id}: warmup failed ({record.error_kind})")
                stopped.set()

            async with asyncio.TaskGroup() as group:
                group.create_task(dispatch())
                group.create_task(observe_system(check_alive, stopped))
    finally:
        try:
            recorder.close()
        finally:
            write_json(
                directory / "warmup.json",
                {"requests": [record.model_dump(mode="json") for record in recorder.requests]},
            )


def package_versions(names: tuple[str, ...], errors: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
            errors.append(f"package version unavailable: {name}: {error}")
    return versions


def capture_owned_metadata(
    uuids: tuple[str, ...],
    *,
    target_device_uuids: dict[ModelId, tuple[str, ...]],
    role_device_uuids: dict[Literal["atn", "ffn"], tuple[str, ...]],
    errors: list[str],
) -> ServingMetadata:
    """Observe once before timing and project host facts onto a validated lease.

    Observation failures preserve known placement with unknown device fields and
    append diagnostics; this snapshot never starts CUDA or changes visibility.
    """

    devices = {uuid: ServingDevice(uuid=uuid) for uuid in uuids}
    links = None
    driver_version = None
    try:
        inventory = query_physical_device_details()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))
    else:
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
    # Use available distribution metadata; do not infer CUDA builds from package suffixes.
    return ServingMetadata(
        devices=tuple(devices[uuid] for uuid in uuids),
        links=links,
        target_device_uuids=target_device_uuids,
        role_device_uuids=role_device_uuids,
        driver_version=driver_version,
        packages=package_versions(("sglang", "torch"), errors),
    )


def write_environment(directory: Path, environment: dict[str, JsonValue]) -> None:
    try:
        write_json(directory / "environment.json", environment)
    except OSError as error:
        # Measurement evidence remains authoritative if the diagnostic disk write fails.
        print(f"xbench environment snapshot unavailable: {error}", file=sys.stderr)
