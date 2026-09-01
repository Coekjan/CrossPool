"""Structured output for native transport device-phase observations."""

from __future__ import annotations

import json
import logging
import os
from functools import wraps
from pathlib import Path
from threading import Lock

import xpool.native
from xpool.bootstrap import get_runtime_role
from xpool.config import get_global_config
from xpool.native import RuntimeRole
from xpool.runtime.atnagent import AtnAgentTransportRuntime
from xpool.runtime.instance import InstanceRankRuntime
from xpool.transport import TransportArenaHandle

runtime_roles = frozenset({RuntimeRole.INSTANCE, RuntimeRole.ATNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False
transport_trace_fields = (
    "trace_id",
    "payload_rows",
    "layer_ordinal",
    "forward_mode",
    "output_requirement",
    "dp_row_layout",
    "result_code",
    "request_staging_started",
    "request_staging_completed",
    "request_published",
    "request_observed",
    "execution_started",
    "execution_completed",
    "result_published",
    "result_observed",
    "output_copied",
    "result_acknowledged",
    "closed",
)
transport_phases = (
    ("request_staging", "request_staging_started", "request_staging_completed"),
    ("request_publication", "request_staging_completed", "request_published"),
    ("request_processing", "request_observed", "result_published"),
    ("execution", "execution_started", "execution_completed"),
    ("output_copy", "result_observed", "output_copied"),
    ("result_acknowledgement", "output_copied", "result_acknowledged"),
    ("result_acknowledged_total", "request_staging_started", "result_acknowledged"),
    ("closed_total", "request_staging_started", "closed"),
)
type TransportRecordJson = dict[str, int | str | dict[str, int]]


def install() -> None:
    """Install transport snapshot recording after process-wide quiesce.

    Raises:
        MissingRequiredConfig: If no process-global config is installed.
        RuntimeError: If enabled transport observation has no output directory.
        OSError: If the configured output directory cannot be created.

    Side Effects:
        Prepares the configured output directory and installs the role-specific
        snapshot hook once per process: Instance detach or AtnAgent quiesce.
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
        if get_runtime_role() is RuntimeRole.INSTANCE:
            install_instance_observer()
        else:
            install_atnagent_observer()
        installed = True


def install_atnagent_observer() -> None:
    """Record every AtnAgent-local endpoint after resident quiesce."""

    original_quiesce = AtnAgentTransportRuntime.quiesce

    @wraps(original_quiesce)
    def observed_quiesce(transport_runtime: AtnAgentTransportRuntime) -> None:
        resources = transport_runtime.resources
        original_quiesce(transport_runtime)
        try:
            snapshot = xpool.native.devkit.transport_observer.read()
            if snapshot is None:
                return
            config = get_global_config()
            resource_by_endpoint = {
                (config.instance_by_id[resource.instance_id].instance_index, resource.registration.rank): resource
                for resource in resources
            }
            for endpoint in snapshot.endpoints:
                resource = resource_by_endpoint[(endpoint.instance_index, endpoint.instance_rank)]
                write_transport_snapshot(
                    site="atnagent",
                    instance_id=resource.instance_id,
                    rank=resource.registration.rank,
                    handle=resource.handle,
                    snapshot=endpoint,
                )
        except Exception:
            logger.warning("Failed to record xpool transport observer snapshot", exc_info=True)

    setattr(AtnAgentTransportRuntime, "quiesce", observed_quiesce)


def install_instance_observer() -> None:
    """Record the Instance-local endpoint immediately before native detach."""

    original_detach = InstanceRankRuntime.detach_arena

    @wraps(original_detach)
    def observed_detach(runtime: InstanceRankRuntime) -> None:
        handle = runtime.arena_handle
        if handle is not None:
            try:
                snapshot = xpool.native.devkit.transport_observer.read()
                if snapshot is not None:
                    write_transport_snapshot(
                        site="instance",
                        instance_id=runtime.instance_id,
                        rank=runtime.rank,
                        handle=handle,
                        snapshot=snapshot.endpoints[0],
                    )
            except Exception:
                logger.warning("Failed to record xpool transport observer snapshot", exc_info=True)
        original_detach(runtime)

    setattr(InstanceRankRuntime, "detach_arena", observed_detach)


def transport_phase_summary(records: list[TransportRecordJson]) -> dict[str, dict[str, int]]:
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
    site: str,
    instance_id: str,
    rank: int,
    handle: TransportArenaHandle,
    snapshot: xpool.native.devkit.transport_observer.EndpointSnapshot,
) -> Path:
    """Write one native observer snapshot as deterministic JSON.

    Args:
        site: Process-local endpoint role represented by the snapshot.
        instance_id: Configured instance associated with the arena.
        rank: Instance rank associated with the arena.
        handle: Arena identity used to distinguish resource generations.
        snapshot: Native bounded-buffer trace snapshot copied at the drain boundary.

    Returns:
        Path of the written JSON document.

    Side Effects:
        Reads the process-global output directory and atomically replaces the
        process-local snapshot file.
    """

    outdir = get_global_config().debug.transport_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool transport observer requires debug.transport_observer.outdir")
    records: list[TransportRecordJson] = []
    for record in snapshot.records:
        raw_values = {field: getattr(record, field) for field in transport_trace_fields}
        numeric_values = {
            field: int(value)
            for field, value in raw_values.items()
            if field not in {"forward_mode", "output_requirement", "dp_row_layout", "result_code"}
        }
        if numeric_values["trace_id"] == 0:
            continue
        durations = {
            name: numeric_values[end] - numeric_values[start]
            for name, start, end in transport_phases
            if numeric_values[start] != 0 and numeric_values[end] != 0 and numeric_values[end] >= numeric_values[start]
        }
        values: TransportRecordJson = {
            **numeric_values,
            "forward_mode": record.forward_mode.name.lower(),
            "output_requirement": record.output_requirement.name.lower(),
            "dp_row_layout": record.dp_row_layout.name.lower(),
            "result_code": record.result_code.name.lower(),
        }
        records.append({**values, "durations_ns": durations})
    records.sort(key=lambda record: int(record["trace_id"]))
    completed_field = "result_published" if site == "atnagent" else "result_acknowledged"
    completed = sum(1 for record in records if record[completed_field] != 0)
    closed = sum(1 for record in records if record["closed"] != 0)
    if any(record["result_acknowledged"] != 0 and record["closed"] != 0 for record in records):
        raise RuntimeError("xpool Transport trace record cannot be both acknowledged and closed")
    incomplete = len(records) - completed - closed
    payload = {
        "pid": os.getpid(),
        "site": site,
        "instance_id": instance_id,
        "rank": rank,
        "arena_handle_suffix": handle.handle[-16:],
        "sequence": snapshot.sequence,
        "dropped": snapshot.dropped,
        "phase_summary": transport_phase_summary(records),
        "record_counts": {
            "retained": len(records),
            "completed": completed,
            "closed": closed,
            "incomplete": incomplete,
        },
        "records": records,
    }
    filename_instance_id = instance_id.replace("/", "--")
    path = outdir / f"xpool.transport-observer.{os.getpid()}.{site}.{filename_instance_id}.{rank}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)
    return path
