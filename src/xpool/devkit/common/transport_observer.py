"""Structured output for native transport device-phase observations."""

from __future__ import annotations

import json
import logging
import os
from functools import wraps
from pathlib import Path
from threading import Lock

import xpool.native
from xpool.abi import DpPaddingMode, FfnResultCode, FfnResultHandoff, XPoolForwardMode
from xpool.config import get_global_config
from xpool.runtime import RuntimeRole
from xpool.runtime.atnagent import AtnTransportCatalog
from xpool.transport import TransportArenaHandle

runtime_roles = frozenset({RuntimeRole.ATNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False
transport_trace_fields = (
    "trace_id",
    "payload_rows",
    "layer_ordinal",
    "forward_mode",
    "result_handoff",
    "dp_padding_mode",
    "result_code",
    "staging_started",
    "staging_completed",
    "published",
    "published_observed",
    "execution_started",
    "execution_admitted",
    "execution_completed",
    "evaluated",
    "evaluated_observed",
    "output_copied",
    "acknowledged",
    "closed",
)
transport_phases = (
    ("staging", "staging_started", "staging_completed"),
    ("publication", "staging_completed", "published"),
    ("publication_visibility", "published", "published_observed"),
    ("evaluation", "published_observed", "evaluated"),
    ("execution_admission", "execution_started", "execution_admitted"),
    ("execution", "execution_started", "execution_completed"),
    ("evaluation_visibility", "evaluated", "evaluated_observed"),
    ("output_copy", "evaluated_observed", "output_copied"),
    ("acknowledgement", "output_copied", "acknowledged"),
    ("acknowledged_total", "staging_started", "acknowledged"),
    ("closed_total", "staging_started", "closed"),
)
type TransportRecordJson = dict[str, int | str | dict[str, int]]


def install() -> None:
    """Install transport snapshot recording after process-wide quiesce.

    Raises:
        MissingRequiredConfig: If no process-global config is installed.
        RuntimeError: If enabled transport observation has no output directory.
        OSError: If the configured output directory cannot be created.

    Side Effects:
        Prepares the configured output directory and monkeypatches
        :meth:`AtnTransportCatalog.quiesce` once per process.
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
        original_quiesce = AtnTransportCatalog.quiesce

        @wraps(original_quiesce)
        def observed_quiesce(catalog: AtnTransportCatalog) -> None:
            resources = catalog.resources
            original_quiesce(catalog)
            for resource in resources:
                try:
                    snapshot = xpool.native.transport.read_trace(resource.handle.handle)
                    if snapshot is not None:
                        write_transport_snapshot(
                            instance_id=resource.instance_id,
                            rank=resource.registration.rank,
                            handle=resource.handle,
                            snapshot=snapshot,
                        )
                except Exception:
                    logger.warning("Failed to record xpool transport observer snapshot", exc_info=True)

        setattr(AtnTransportCatalog, "quiesce", observed_quiesce)
        installed = True


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
    instance_id: str,
    rank: int,
    handle: TransportArenaHandle,
    snapshot: xpool.native.TransportTraceSnapshot,
) -> Path:
    """Write one native observer snapshot as deterministic JSON.

    Args:
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
        raw_values = {field: int(getattr(record, field)) for field in transport_trace_fields}
        if raw_values["trace_id"] == 0:
            continue
        durations = {
            name: raw_values[end] - raw_values[start]
            for name, start, end in transport_phases
            if raw_values[start] != 0 and raw_values[end] != 0 and raw_values[end] >= raw_values[start]
        }
        values: TransportRecordJson = {
            **raw_values,
            "forward_mode": XPoolForwardMode(raw_values["forward_mode"]).name.lower(),
            "result_handoff": FfnResultHandoff(raw_values["result_handoff"]).name.lower(),
            "dp_padding_mode": DpPaddingMode(raw_values["dp_padding_mode"]).name.lower(),
            "result_code": FfnResultCode(raw_values["result_code"]).name.lower(),
        }
        records.append({**values, "durations_ns": durations})
    records.sort(key=lambda record: int(record["trace_id"]))
    acknowledged = sum(1 for record in records if record["acknowledged"] != 0)
    closed = sum(1 for record in records if record["closed"] != 0)
    if any(record["acknowledged"] != 0 and record["closed"] != 0 for record in records):
        raise RuntimeError("xpool Transport trace record cannot be both acknowledged and closed")
    incomplete = len(records) - acknowledged - closed
    payload = {
        "pid": os.getpid(),
        "instance_id": instance_id,
        "rank": rank,
        "arena_handle_suffix": handle.handle[-16:],
        "sequence": snapshot.sequence,
        "dropped": snapshot.dropped,
        "phase_summary": transport_phase_summary(records),
        "record_counts": {
            "retained": len(records),
            "acknowledged": acknowledged,
            "closed": closed,
            "incomplete": incomplete,
        },
        "records": records,
    }
    filename_instance_id = instance_id.replace("/", "--")
    path = outdir / f"xpool.transport-observer.{os.getpid()}.{filename_instance_id}.{rank}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)
    return path
