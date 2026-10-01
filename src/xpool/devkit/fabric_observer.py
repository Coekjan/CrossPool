"""Structured output for per-PE native Fabric traces."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, TypedDict

from pydantic import Field

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationId, FabricParticipantPhase
from xpool.native import RuntimeRole
from xpool.runtime.agent import Agent

runtime_roles = frozenset({RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False


class FabricAtnAgentFactsJson(TypedDict):
    """AtnAgent facts, including the lease acquired after admission."""

    dp_rank_payload_rows: int
    forward_mode: str
    executor_lane_index: int | None
    executor_lease_sequence: int | None


class FabricRandomSchedulerJson(TypedDict):
    """Random scheduling facts without a FIFO ticket."""

    policy: Literal["random"]


class FabricFifoSchedulerJson(TypedDict):
    """FIFO scheduling facts with the invocation's ready ticket."""

    policy: Literal["fifo"]
    ready_ticket: int


class FabricCoordinatorFactsJson(TypedDict):
    """Coordinator scheduling facts, with a nullable pre-schedule lease."""

    executor_lane_index: int | None
    executor_lease_sequence: int | None
    scheduler: FabricRandomSchedulerJson | FabricFifoSchedulerJson


class FabricFfnAgentFactsJson(TypedDict):
    """FfnAgent lease, capacity and delivery facts."""

    executor_lane_index: int
    executor_lease_sequence: int
    payload_row_capacity: int
    delivery: str


class FabricRecordFieldsJson(TypedDict):
    """Scalar invocation fields shared by the native trace alternatives."""

    local_trace_id: int
    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    payload_rows: int
    output_requirement: str


class FabricAtnAgentRecordJson(FabricRecordFieldsJson):
    """One AtnAgent trace and its local event timings."""

    kind: Literal["atnagent"]
    dp_row_layout: str
    facts: FabricAtnAgentFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


class FabricCoordinatorRecordJson(FabricRecordFieldsJson):
    """One Coordinator trace and its local event timings."""

    kind: Literal["coordinator"]
    dp_row_layout: None
    facts: FabricCoordinatorFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


class FabricFfnAgentRecordJson(FabricRecordFieldsJson):
    """One FfnAgent trace and its local event timings."""

    kind: Literal["ffnagent"]
    dp_row_layout: None
    facts: FabricFfnAgentFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


type FabricRecordJson = Annotated[
    FabricAtnAgentRecordJson | FabricCoordinatorRecordJson | FabricFfnAgentRecordJson,
    Field(discriminator="kind"),
]


class FabricModelTopologyJson(TypedDict):
    """Attention parallelism attached to one co-indexed Model Plan."""

    atn_tp_size: int
    atn_dp_size: int


class FabricSnapshotJson(TypedDict):
    """One PE's bounded Fabric trace document."""

    generation: str
    pe: int
    atnagent_count: int
    ffnagent_count: int
    model_topologies: list[FabricModelTopologyJson]
    sequence: int
    dropped: int
    records: list[FabricRecordJson]


def event_timestamps(
    record: xpool.native.devkit.fabric_observer.Record,
    events: Mapping[
        str,
        xpool.native.devkit.fabric_observer.AtnAgentEvent
        | xpool.native.devkit.fabric_observer.CoordinatorEvent
        | xpool.native.devkit.fabric_observer.FfnAgentEvent,
    ],
) -> dict[str, int]:
    """Read the complete native event family without duplicating its domain."""

    return {name.lower(): record.timestamp(event) for name, event in events.items()}


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


def atnagent_payload(
    record: xpool.native.devkit.fabric_observer.Record,
) -> tuple[FabricAtnAgentFactsJson, dict[str, int], dict[str, int]]:
    """Project one target AtnAgent trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.AtnAgentEvent.__members__)
    dp_rank_payload_rows = record.dp_rank_payload_rows
    forward_mode = record.forward_mode
    assert dp_rank_payload_rows is not None and forward_mode is not None
    facts: FabricAtnAgentFactsJson = {
        "dp_rank_payload_rows": dp_rank_payload_rows,
        "forward_mode": forward_mode.name.lower(),
        "executor_lane_index": record.executor_lane_index,
        "executor_lease_sequence": record.executor_lease_sequence,
    }
    return (
        facts,
        events,
        durations(
            events,
            (
                ("admission_wait", "submission_published", "admission_observed"),
                ("output_wait", "admission_observed", "output_commit_observed"),
                ("total", "submission_prepared", "output_acknowledgement_published"),
            ),
        ),
    )


def coordinator_payload(
    record: xpool.native.devkit.fabric_observer.Record,
) -> tuple[FabricCoordinatorFactsJson, dict[str, int], dict[str, int]]:
    """Project one target Coordinator trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.CoordinatorEvent.__members__)
    ready_ticket = record.ready_ticket
    scheduler: FabricRandomSchedulerJson | FabricFifoSchedulerJson = (
        {"policy": "random"}
        if ready_ticket is None
        else {
            "policy": "fifo",
            "ready_ticket": ready_ticket,
        }
    )
    facts: FabricCoordinatorFactsJson = {
        "executor_lane_index": record.executor_lane_index,
        "executor_lease_sequence": record.executor_lease_sequence,
        "scheduler": scheduler,
    }
    return (
        facts,
        events,
        durations(
            events,
            (
                ("scheduling", "enqueued", "scheduled"),
                ("execution", "lane_execution_published", "ffnagent_completions_observed"),
                ("active_total", "scheduled", "lane_released"),
                ("total", "enqueued", "lane_released"),
            ),
        ),
    )


def ffnagent_payload(
    record: xpool.native.devkit.fabric_observer.Record,
) -> tuple[FabricFfnAgentFactsJson, dict[str, int], dict[str, int]]:
    """Project one target FfnAgent trace alternative."""

    events = event_timestamps(record, xpool.native.devkit.fabric_observer.FfnAgentEvent.__members__)
    executor_lane_index = record.executor_lane_index
    executor_lease_sequence = record.executor_lease_sequence
    payload_row_capacity = record.payload_row_capacity
    delivery = record.delivery
    assert executor_lane_index is not None and executor_lease_sequence is not None
    assert payload_row_capacity is not None and delivery is not None
    facts: FabricFfnAgentFactsJson = {
        "executor_lane_index": executor_lane_index,
        "executor_lease_sequence": executor_lease_sequence,
        "payload_row_capacity": payload_row_capacity,
        "delivery": delivery.name.lower(),
    }
    return (
        facts,
        events,
        durations(
            events,
            (
                ("input_wait", "lane_execution_observed", "input_ready_observed"),
                ("compute", "compute_started", "compute_completed"),
                ("total", "lane_execution_observed", "completion_published"),
            ),
        ),
    )


def record_payload(record: xpool.native.devkit.fabric_observer.Record) -> FabricRecordJson:
    """Serialize one local Fabric record and derive same-device durations."""

    common: FabricRecordFieldsJson = {
        "local_trace_id": record.local_trace_id,
        "instance_index": record.key.instance_index,
        "invocation_sequence": record.key.invocation_sequence,
        "layer_ordinal": record.layer_ordinal,
        "payload_rows": record.payload_rows,
        "output_requirement": record.output_requirement.name.lower(),
    }
    match record.kind:
        case xpool.native.devkit.fabric_observer.RecordKind.ATNAGENT:
            atn_facts, events, projected_durations = atnagent_payload(record)
            dp_row_layout = record.dp_row_layout
            assert dp_row_layout is not None
            return {
                **common,
                "kind": "atnagent",
                "dp_row_layout": dp_row_layout.name.lower(),
                "facts": atn_facts,
                "events_ns": events,
                "durations_ns": projected_durations,
            }
        case xpool.native.devkit.fabric_observer.RecordKind.COORDINATOR:
            coordinator_facts, events, projected_durations = coordinator_payload(record)
            return {
                **common,
                "kind": "coordinator",
                "dp_row_layout": None,
                "facts": coordinator_facts,
                "events_ns": events,
                "durations_ns": projected_durations,
            }
        case xpool.native.devkit.fabric_observer.RecordKind.FFNAGENT:
            ffn_facts, events, projected_durations = ffnagent_payload(record)
            return {
                **common,
                "kind": "ffnagent",
                "dp_row_layout": None,
                "facts": ffn_facts,
                "events_ns": events,
                "durations_ns": projected_durations,
            }
        case unexpected_kind:
            raise RuntimeError(f"xpool Fabric observer received unknown trace kind {unexpected_kind!r}")


def write_fabric_snapshot(
    generation: FabricGenerationId,
    snapshot: xpool.native.devkit.fabric_observer.Snapshot,
) -> Path:
    """Atomically write one PE's local Fabric trace snapshot as JSON."""

    outdir = get_global_config().debug.fabric_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool fabric observer requires debug.fabric_observer.outdir")
    records = [record_payload(record) for record in snapshot.records if record.local_trace_id != 0]
    records.sort(key=lambda record: record["local_trace_id"])
    generation_text = generation.format()
    payload: FabricSnapshotJson = {
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
                logger.warning("failed to record xpool fabric observer snapshot", exc_info=True)

        setattr(Agent, "advance_fabric_lifecycle", observed_advance)
        installed = True
