"""Structured output for per-PE native Fabric traces."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import cast

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationId, FabricParticipantPhase
from xpool.native import RuntimeRole
from xpool.runtime.agent import Agent

runtime_roles = frozenset({RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False


def event_timestamps(
    record: xpool.native.devkit.fabric_observer.Record,
    events: Mapping[str, object],
) -> dict[str, int]:
    """Read the complete native event family without duplicating its domain."""

    timestamp = cast(Callable[[object], int], record.timestamp)
    return {name.lower(): timestamp(event) for name, event in events.items()}


def durations(
    events_ns: dict[str, int],
    pairs: tuple[tuple[str, str, str], ...],
) -> dict[str, int]:
    """Derive valid same-record durations from explicit event pairs."""

    result: dict[str, int] = {}
    for name, start_name, end_name in pairs:
        start = events_ns[start_name]
        end = events_ns[end_name]
        if start != 0 and end >= start:
            result[name] = end - start
    return result


def atnagent_payload(record: xpool.native.devkit.fabric_observer.Record) -> tuple[dict[str, object], dict[str, int]]:
    """Project one target AtnAgent trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.AtnAgentEvent.__members__)
    facts: dict[str, object] = {
        "dp_rank_payload_rows": record.dp_rank_payload_rows,
        "forward_mode": None if record.forward_mode is None else record.forward_mode.name.lower(),
        "executor_lane_index": record.executor_lane_index,
        "executor_lease_sequence": record.executor_lease_sequence,
    }
    return facts, durations(
        events,
        (
            ("admission_wait", "submission_published", "admission_observed"),
            ("output_wait", "admission_observed", "output_commit_observed"),
            ("total", "submission_prepared", "output_acknowledgement_published"),
        ),
    )


def coordinator_payload(record: xpool.native.devkit.fabric_observer.Record) -> tuple[dict[str, object], dict[str, int]]:
    """Project one target Coordinator trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.CoordinatorEvent.__members__)
    scheduler = (
        {"policy": "random"}
        if record.ready_ticket is None
        else {
            "policy": "fifo",
            "ready_ticket": record.ready_ticket,
        }
    )
    facts: dict[str, object] = {
        "executor_lane_index": record.executor_lane_index,
        "executor_lease_sequence": record.executor_lease_sequence,
        "scheduler": scheduler,
    }
    return facts, durations(
        events,
        (
            ("scheduling", "enqueued", "scheduled"),
            ("execution", "lane_execution_published", "ffnagent_completions_observed"),
            ("active_total", "scheduled", "lane_released"),
            ("total", "enqueued", "lane_released"),
        ),
    )


def ffnagent_payload(record: xpool.native.devkit.fabric_observer.Record) -> tuple[dict[str, object], dict[str, int]]:
    """Project one target FfnAgent trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.FfnAgentEvent.__members__)
    facts: dict[str, object] = {
        "executor_lane_index": record.executor_lane_index,
        "executor_lease_sequence": record.executor_lease_sequence,
        "payload_row_capacity": record.payload_row_capacity,
        "delivery": None if record.delivery is None else record.delivery.name.lower(),
    }
    return facts, durations(
        events,
        (
            ("input_wait", "lane_execution_observed", "input_ready_observed"),
            ("compute", "compute_started", "compute_completed"),
            ("total", "lane_execution_observed", "completion_published"),
        ),
    )


def record_payload(record: xpool.native.devkit.fabric_observer.Record) -> dict[str, object]:
    """Serialize one local Fabric record and derive same-device durations."""

    match record.kind:
        case xpool.native.devkit.fabric_observer.RecordKind.ATNAGENT:
            kind = "atnagent"
            events = xpool.native.devkit.fabric_observer.AtnAgentEvent.__members__
            facts, projected_durations = atnagent_payload(record)
        case xpool.native.devkit.fabric_observer.RecordKind.COORDINATOR:
            kind = "coordinator"
            events = xpool.native.devkit.fabric_observer.CoordinatorEvent.__members__
            facts, projected_durations = coordinator_payload(record)
        case xpool.native.devkit.fabric_observer.RecordKind.FFNAGENT:
            kind = "ffnagent"
            events = xpool.native.devkit.fabric_observer.FfnAgentEvent.__members__
            facts, projected_durations = ffnagent_payload(record)
        case unexpected_kind:
            raise RuntimeError(f"xpool Fabric observer received unknown trace kind {unexpected_kind!r}")
    return {
        "local_trace_id": record.local_trace_id,
        "kind": kind,
        "instance_index": record.key.instance_index,
        "invocation_sequence": record.key.invocation_sequence,
        "layer_ordinal": record.layer_ordinal,
        "payload_rows": record.payload_rows,
        "output_requirement": record.output_requirement.name.lower(),
        "dp_row_layout": None if record.dp_row_layout is None else record.dp_row_layout.name.lower(),
        "facts": facts,
        "events_ns": event_timestamps(record, events),
        "durations_ns": projected_durations,
    }


def write_fabric_snapshot(
    generation: FabricGenerationId,
    snapshot: xpool.native.devkit.fabric_observer.Snapshot,
) -> Path:
    """Atomically write one PE's local Fabric trace snapshot as JSON."""

    outdir = get_global_config().debug.fabric_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool fabric observer requires debug.fabric_observer.outdir")
    records = [record_payload(record) for record in snapshot.records if record.local_trace_id != 0]
    records.sort(key=lambda record: int(record["local_trace_id"]))
    generation_text = generation.format()
    payload = {
        "generation": generation_text,
        "pe": snapshot.pe,
        "atnagent_count": snapshot.atnagent_count,
        "ffnagent_count": snapshot.ffnagent_count,
        "model_topologies": [
            {"atn_tp_size": topology.atn_tp_size, "atn_dp_size": topology.atn_dp_size}
            for topology in snapshot.model_topologies
        ],
        "sequence": snapshot.sequence,
        "dropped": snapshot.dropped,
        "records": records,
    }
    path = outdir / f"xpool.fabric-observer.{generation_text}.{snapshot.pe}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def install() -> None:
    """Install per-PE Fabric snapshot recording after local drain completes."""

    outdir = get_global_config().debug.fabric_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool fabric observer requires debug.fabric_observer.outdir")
    global installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        if installed:
            return
        original_advance = Agent.advance_fabric_lifecycle

        @wraps(original_advance)
        def observed_advance(agent: Agent) -> None:
            plan = agent.fabric_plan
            previous_report = agent.participant_report
            previous_phase = previous_report.phase if previous_report is not None else None
            original_advance(agent)
            updated_report = agent.participant_report
            if (
                plan is None
                or previous_phase is FabricParticipantPhase.DRAINED
                or updated_report is None
                or updated_report.phase is not FabricParticipantPhase.DRAINED
            ):
                return
            try:
                snapshot = xpool.native.devkit.fabric_observer.read()
                if snapshot is not None:
                    write_fabric_snapshot(plan.generation, snapshot)
            except Exception:
                logger.warning("Failed to record xpool Fabric observer snapshot", exc_info=True)

        setattr(Agent, "advance_fabric_lifecycle", observed_advance)
        installed = True
