"""Structured output for native transport device-phase observations."""

from __future__ import annotations

import json
import logging
import os
from functools import wraps
from pathlib import Path
from threading import Lock

from xpool.abi import (
    TRANSPORT_PHASES,
    TRANSPORT_TRACE_FIELDS,
    RuntimeRole,
    TransportArenaHandle,
    TransportTraceSnapshot,
)
from xpool.config import get_global_config
from xpool.runtime.devagent.atn import AtnArenaResource

runtime_roles = frozenset({RuntimeRole.DEVAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False


def install() -> None:
    """Install transport snapshot recording around arena destruction.

    Raises:
        MissingRequiredConfig: If no process-global config is installed.
        RuntimeError: If enabled transport observation has no output directory.
        OSError: If the configured output directory cannot be created.

    Side Effects:
        Prepares the configured output directory and monkeypatches
        :meth:`AtnArenaResource.destroy` once per process.
    """

    config = get_global_config()
    outdir = config.debug.transport_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool transport observer requires debug.transport_observer.outdir")

    global installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        if installed:
            return
        original_destroy = AtnArenaResource.destroy

        @wraps(original_destroy)
        def observed_destroy(resource: AtnArenaResource) -> TransportTraceSnapshot:
            snapshot = original_destroy(resource)
            try:
                write_transport_snapshot(
                    instance_id=resource.instance_id,
                    rank=resource.registration.rank,
                    handle=resource.handle,
                    snapshot=snapshot,
                )
            except Exception:
                logger.warning("Failed to record xpool transport observer snapshot", exc_info=True)
            return snapshot

        setattr(AtnArenaResource, "destroy", observed_destroy)
        installed = True


def transport_phase_summary(records: list[dict[str, int | dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Aggregate observer phase durations with nearest-rank percentiles.

    Args:
        records: Serialized transport records containing ``durations_ns`` maps.

    Returns:
        Per-phase count, minimum, median, p95, p99, and maximum nanoseconds.
    """

    values_by_phase: dict[str, list[int]] = {}
    for record in records:
        durations = record["durations_ns"]
        if not isinstance(durations, dict):
            continue
        for phase, duration in durations.items():
            values_by_phase.setdefault(phase, []).append(duration)
    summaries: dict[str, dict[str, int]] = {}
    for phase, values in values_by_phase.items():
        values.sort()
        last_index = len(values) - 1
        summaries[phase] = {
            "count": len(values),
            "min_ns": values[0],
            "median_ns": values[last_index // 2],
            "p95_ns": values[round(last_index * 0.95)],
            "p99_ns": values[round(last_index * 0.99)],
            "max_ns": values[-1],
        }
    return summaries


def write_transport_snapshot(
    *,
    instance_id: str,
    rank: int,
    handle: TransportArenaHandle,
    snapshot: TransportTraceSnapshot,
) -> Path:
    """Write one native observer snapshot as deterministic JSON.

    Args:
        instance_id: Configured instance associated with the arena.
        rank: Instance rank associated with the arena.
        handle: Arena identity used to distinguish resource generations.
        snapshot: Native trace ring snapshot copied at the drain boundary.

    Returns:
        Path of the written JSON document.

    Side Effects:
        Reads the process-global output directory and atomically replaces the
        process-local snapshot file.
    """

    outdir = get_global_config().debug.transport_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool transport observer requires debug.transport_observer.outdir")
    records: list[dict[str, int | dict[str, int]]] = []
    for record in snapshot.records:
        values = {field: int(getattr(record, field)) for field in TRANSPORT_TRACE_FIELDS}
        if values["trace_id"] == 0:
            continue
        durations = {
            name: values[end] - values[start]
            for name, start, end in TRANSPORT_PHASES
            if values[start] != 0 and values[end] != 0 and values[end] >= values[start]
        }
        records.append({**values, "durations_ns": durations})
    records.sort(key=lambda record: int(record["trace_id"]))
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "instance_id": instance_id,
        "rank": rank,
        "arena_handle_suffix": handle.handle[-16:],
        "sequence": snapshot.sequence,
        "dropped": snapshot.dropped,
        "incomplete": sum(1 for record in records if record["slot_recycled"] == 0),
        "phase_summary": transport_phase_summary(records),
        "records": records,
    }
    filename_instance_id = instance_id.replace("/", "--")
    path = outdir / f"xpool.transport-observer.{os.getpid()}.{filename_instance_id}.{rank}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)
    return path
