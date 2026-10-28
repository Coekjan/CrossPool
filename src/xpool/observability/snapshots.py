"""Explicit-domain earliest-N projections of retained offline evidence."""

from dataclasses import dataclass

from xpool.devkit.fabric_observer import (
    FabricAtnAgentRecordJson,
    FabricCoordinatorRecordJson,
    FabricFfnAgentRecordJson,
    FabricRecordFieldsJson,
    FabricRecordJson,
    FabricSnapshotJson,
)
from xpool.devkit.transport_observer import (
    TransportRecordJson,
    TransportSnapshotJson,
    transport_phase_summary,
)
from xpool.observability.applicability import fabric_times, transport_times
from xpool.observability.entities import Observation
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.evidence import (
    EndpointReference,
    SnapshotEvidenceReport,
    SnapshotObserver,
    evidence_entry,
    unobserved_entry,
)
from xpool.observability.intervals import fabric_pairs, transport_pairs
from xpool.observability.models import Finding
from xpool.observability.reconstruction import Reconstruction
from xpool.observability.wire import (
    Delivery,
    DpRowLayout,
    EventKind,
    EventRole,
    ForwardMode,
    OutputRequirement,
    ResultCode,
)
from xpool.utils.integer import ensure_uint64

__all__ = ["LegacyProjection", "SnapshotRequest", "project_legacy_snapshots"]


@dataclass(frozen=True)
class SnapshotRequest:
    """Exact lifetime domain and original operation-attempt count supplied by its owner.

    An Endpoint selects Transport. Otherwise Generation selects Fabric. A null
    count leaves the snapshot envelope unavailable while preserving records and
    evidence. Counts are never guessed from the retained event inventory.

    Raises:
        ValueError: If domain components, capacity or attempt counts are not
            uint64 integers, or required positive components are zero.
    """

    run_id: str
    producer: int
    generation: int
    operations_attempted: int | None
    capacity: int = 8192
    endpoint: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        ensure_uint64(self.producer, self.generation, self.capacity)
        if any(value == 0 for value in (self.producer, self.generation, self.capacity)):
            raise ValueError("snapshot domain and capacity require positive uint64 integers")
        if self.operations_attempted is not None:
            ensure_uint64(self.operations_attempted)
        if self.endpoint is not None:
            if len(self.endpoint) != 2:
                raise ValueError("Endpoint requires two positive uint64 components")
            ensure_uint64(*self.endpoint)
            if any(component == 0 for component in self.endpoint):
                raise ValueError("Endpoint requires two positive uint64 components")


@dataclass(frozen=True)
class LegacyProjection:
    """Individual records, optional whole snapshot, and one typed evidence report.

    Unknown totals keep records and report available while snapshot is None.
    Disabled observers return no snapshot/report and an empty record tuple.
    Findings retain diagnostics beyond the report's finite local quality reasons.
    """

    request: SnapshotRequest
    snapshot: TransportSnapshotJson | FabricSnapshotJson | None
    records: tuple[TransportRecordJson | FabricRecordJson, ...]
    evidence_report: SnapshotEvidenceReport | None
    findings: tuple[Finding, ...]


forward_names: dict[int, str] = {value: value.name.lower() for value in ForwardMode}
output_names: dict[int, str] = {value: value.name.lower() for value in OutputRequirement}
layout_names: dict[int, str] = {value: value.name.lower() for value in DpRowLayout}
delivery_names: dict[int, str] = {value: value.name.lower() for value in Delivery}
result_names: dict[int, str] = {value: value.name.lower() for value in ResultCode}


def legacy_durations(observation: Observation, findings: list[Finding]) -> dict[str, int]:
    pairs = (
        transport_pairs if observation.identity.entity.kind == "operation" else fabric_pairs[observation.identity.role]
    )
    durations: dict[str, int] = {}

    for name, start_kind, end_kind in pairs:
        start, end = observation.events.get(start_kind), observation.events.get(end_kind)
        if start is None or end is None:
            continue

        sources = (start.source, end.source)

        if start.record.clock_domain_id != end.record.clock_domain_id:
            findings.append(Finding("clock_domain_mismatch", f"legacy {name} unavailable", sources))
        elif start.record.raw_time_ns == 0 or end.record.raw_time_ns == 0:
            findings.append(Finding("legacy_zero_timestamp", f"legacy {name} unavailable", sources))
        elif end.record.raw_time_ns < start.record.raw_time_ns:
            findings.append(Finding("temporal_conflict", f"legacy {name} reversed", sources))
        else:
            durations[name] = end.record.raw_time_ns - start.record.raw_time_ns

    for event in observation.events.values():
        if event.record.raw_time_ns == 0:
            findings.append(
                Finding("legacy_zero_timestamp", "recorded raw zero projects as legacy zero", (event.source,))
            )

    return durations


def transport_record(observation: Observation, findings: list[Finding]) -> TransportRecordJson | None:
    facts = observation.facts

    required: tuple[str, ...] = (
        "local_trace_id",
        "payload_rows",
        "layer_ordinal",
        "forward_mode",
        "output_requirement",
        "dp_row_layout",
    )

    if any(name not in facts for name in required):
        findings.append(
            Finding("legacy_fact_missing", "Transport scalar facts unavailable", tuple(observation.sources))
        )
        return None

    times = {
        name: observation.events[kind].record.raw_time_ns if kind in observation.events else 0
        for kind, name in enumerate(transport_times, EventKind.REQUEST_STAGING_STARTED)
    }

    if "result_code" not in facts:
        findings.append(Finding("unknown_result", "site-local result unavailable", tuple(observation.sources)))

    return TransportRecordJson(
        trace_id=facts["local_trace_id"],
        payload_rows=facts["payload_rows"],
        layer_ordinal=facts["layer_ordinal"],
        forward_mode=forward_names[facts["forward_mode"]],
        output_requirement=output_names[facts["output_requirement"]],
        dp_row_layout=layout_names[facts["dp_row_layout"]],
        result_code=result_names[facts.get("result_code", ResultCode.PROTOCOL_MISMATCH)],
        request_staging_started=times["request_staging_started"],
        request_staging_completed=times["request_staging_completed"],
        request_published=times["request_published"],
        request_observed=times["request_observed"],
        execution_started=times["execution_started"],
        execution_completed=times["execution_completed"],
        result_published=times["result_published"],
        result_observed=times["result_observed"],
        output_copied=times["output_copied"],
        result_acknowledged=times["result_acknowledged"],
        closed=times["closed"],
        durations_ns=legacy_durations(observation, findings),
    )


def fabric_record(observation: Observation, scheduler: str, findings: list[Finding]) -> FabricRecordJson | None:
    facts = observation.facts
    role = observation.identity.role

    required: tuple[str, ...] = (
        "local_trace_id",
        "instance_index",
        "invocation_sequence",
        "layer_ordinal",
        "payload_rows",
        "output_requirement",
    )

    if role == EventRole.ATNAGENT:
        required += ("dp_rank_payload_rows", "forward_mode", "dp_row_layout")
    elif role == EventRole.COORDINATOR and scheduler == "fifo":
        required += ("ready_ticket",)
    elif role == EventRole.FFNAGENT:
        required += ("lane_index", "lease_sequence", "payload_row_capacity", "delivery")

    if any(name not in facts for name in required):
        findings.append(Finding("legacy_fact_missing", "Fabric scalar facts unavailable", tuple(observation.sources)))
        return None

    fields = FabricRecordFieldsJson(
        local_trace_id=facts["local_trace_id"],
        instance_index=facts["instance_index"],
        invocation_sequence=facts["invocation_sequence"],
        layer_ordinal=facts["layer_ordinal"],
        payload_rows=facts["payload_rows"],
        output_requirement=output_names[facts["output_requirement"]],
    )

    start = {
        EventRole.ATNAGENT: EventKind.SUBMISSION_PREPARED,
        EventRole.COORDINATOR: EventKind.ENQUEUED,
        EventRole.FFNAGENT: EventKind.LANE_EXECUTION_OBSERVED,
    }[role]

    events_ns = {
        name: observation.events[kind].record.raw_time_ns if kind in observation.events else 0
        for kind, name in enumerate(fabric_times[role], start)
    }

    durations_ns = legacy_durations(observation, findings)

    if role == EventRole.ATNAGENT:
        return FabricAtnAgentRecordJson(
            **fields,
            kind="atnagent",
            dp_row_layout=layout_names[facts["dp_row_layout"]],
            facts={
                "dp_rank_payload_rows": facts["dp_rank_payload_rows"],
                "forward_mode": forward_names[facts["forward_mode"]],
                "executor_lane_index": facts.get("lane_index"),
                "executor_lease_sequence": facts.get("lease_sequence"),
            },
            events_ns=events_ns,
            durations_ns=durations_ns,
        )

    if role == EventRole.COORDINATOR:
        return FabricCoordinatorRecordJson(
            **fields,
            kind="coordinator",
            dp_row_layout=None,
            facts={
                "executor_lane_index": facts.get("lane_index"),
                "executor_lease_sequence": facts.get("lease_sequence"),
                "scheduler": {"policy": "fifo", "ready_ticket": facts["ready_ticket"]}
                if scheduler == "fifo"
                else {"policy": "random"},
            },
            events_ns=events_ns,
            durations_ns=durations_ns,
        )

    return FabricFfnAgentRecordJson(
        **fields,
        kind="ffnagent",
        dp_row_layout=None,
        facts={
            "executor_lane_index": facts["lane_index"],
            "executor_lease_sequence": facts["lease_sequence"],
            "payload_row_capacity": facts["payload_row_capacity"],
            "delivery": delivery_names[facts["delivery"]],
        },
        events_ns=events_ns,
        durations_ns=durations_ns,
    )


def project_legacy_snapshots(
    result: Reconstruction, requests: tuple[SnapshotRequest, ...]
) -> tuple[LegacyProjection, ...]:
    """Project earliest-N records and v4 reports from one immutable evidence cutoff.

    Explicit operation totals permit a whole legacy snapshot. Unknown totals
    retain identified records and report entries with null counts. Offline
    reports have no published snapshot hash. Metadata order resolves equivalent
    revision references; field Event sources use the smallest Event identity.

    Raises:
        ObservationError: If a domain is unknown or totals contradict evidence.
    """
    return tuple(project_snapshot(result, request) for request in requests)


def project_snapshot(result: Reconstruction, request: SnapshotRequest) -> LegacyProjection:
    # Resolve the explicit snapshot domain even when no local Events survive.
    findings: list[Finding] = []
    documents = tuple(m for m in result.metadata if m.run_id == request.run_id)
    producer = next((p for m in documents for p in m.producers if p.id == request.producer), None)
    generation = next((g for m in documents for g in m.generations if g.id == request.generation), None)
    if producer is None or generation is None:
        raise ObservationError(ErrorCode.METADATA_UNAVAILABLE, "snapshot Producer/Generation unavailable")
    endpoint = None
    site = None
    if request.endpoint is not None:
        endpoint = next(
            (e for m in documents for e in m.endpoints if (e.owner_producer, e.endpoint_sequence) == request.endpoint),
            None,
        )
        if endpoint is None or endpoint.generation != request.generation:
            raise ObservationError(ErrorCode.METADATA_UNAVAILABLE, "snapshot Endpoint unavailable")
        site = next((s.site for s in endpoint.sites if s.producer == request.producer), None)
        if site is None:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "Producer does not observe snapshot Endpoint")
    observer = SnapshotObserver.TRANSPORT if endpoint is not None else SnapshotObserver.FABRIC
    own_document = next((m for m in documents if m.producer_id == request.producer), documents[0])
    capability = (
        own_document.capabilities.transport
        if observer == SnapshotObserver.TRANSPORT
        else own_document.capabilities.fabric
    )
    if capability in ("disabled", "unavailable", "not_applicable"):
        return LegacyProjection(request, None, (), None, ())

    # Match local IDs only inside the requested protocol lifetime and Producer.
    observations: dict[int, Observation] = {}
    pe = generation.pe_for(request.producer) if endpoint is None else None
    if endpoint is None and pe is None:
        raise ObservationError(
            ErrorCode.INVALID_EVENT,
            "Fabric snapshot Producer is not a Generation member",
            producer=request.producer,
            generation=request.generation,
            metadata_field_path="generations",
        )
    for entity in result.entities.values():
        key = entity.identity
        eligible = (
            key.kind == "operation" and key.components[:2] == request.endpoint
            if endpoint is not None
            else key.kind == "invocation" and key.components[0] == request.generation
        )
        if key.run_id != request.run_id or not eligible:
            continue
        for local_observation in entity.observations.values():
            if local_observation.identity.producer != request.producer:
                continue
            local_id = local_observation.facts.get("local_trace_id")
            if local_id is None:
                continue
            if request.operations_attempted is not None and local_id > request.operations_attempted:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "snapshot attempts precede observed local ID")
            if (
                endpoint is not None
                and local_observation.facts.get("generation", request.generation) != request.generation
            ):
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "snapshot observation Generation mismatch")
            if local_id in observations:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "local snapshot ID reused in its lifetime")
            observations[local_id] = local_observation
    observations = {i: o for i, o in observations.items() if i <= request.capacity}

    # Project records and evidence together; missing scalars omit only that record.
    transport_records: list[TransportRecordJson] = []
    fabric_records: list[FabricRecordJson] = []
    entries = []
    completed = closed = 0
    local_ids = (
        range(1, min(request.capacity, request.operations_attempted) + 1)
        if request.operations_attempted is not None
        else sorted(observations)
    )
    for local_id in local_ids:
        observation = observations.get(local_id)
        if observation is None:
            entries.append(unobserved_entry(local_id, site))
            continue
        retained = False
        if endpoint is not None:
            record = transport_record(observation, findings)
            if record is not None:
                retained = True
                transport_records.append(record)
                terminal = EventKind.RESULT_ACKNOWLEDGED if site == "instance" else EventKind.RESULT_PUBLISHED
                completed += terminal in observation.events
                closed += EventKind.CLOSED in observation.events
                if terminal in observation.events and EventKind.CLOSED in observation.events:
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT, "snapshot has both completed and closed terminals"
                    )
        else:
            fabric_value = fabric_record(observation, generation.scheduler, findings)
            if fabric_value is not None:
                retained = True
                fabric_records.append(fabric_value)
        entries.append(evidence_entry(observation, documents, retained=retained))

    # Exact counts admit the snapshot envelope; reports also describe unknown tails.
    sequence = request.operations_attempted
    dropped = max(0, sequence - request.capacity) if sequence is not None else None
    unobserved = min(request.capacity, sequence) - len(observations) if sequence is not None else None
    report = SnapshotEvidenceReport(
        run_id=request.run_id,
        producer_id=request.producer,
        observer=observer,
        generation=request.generation,
        endpoint=EndpointReference(owner=request.endpoint[0], sequence=request.endpoint[1])
        if request.endpoint
        else None,
        site=site,
        capacity=request.capacity,
        sequence=sequence,
        dropped=dropped,
        unobserved_operations=unobserved,
        entries=entries,
    )
    if sequence is None:
        findings.append(Finding("snapshot_total_unknown", "original operation-attempt count unavailable"))
    if dropped:
        findings.append(Finding("snapshot_operations_dropped", f"{dropped} local IDs exceed retained prefix"))
    if unobserved:
        findings.append(Finding("snapshot_operations_unobserved", f"{unobserved} eligible operations lack evidence"))
    snapshot: TransportSnapshotJson | FabricSnapshotJson | None = None
    if sequence is not None and dropped is not None:
        if endpoint is not None and site is not None:
            snapshot = TransportSnapshotJson(
                pid=producer.process.pid,
                site=site,
                model_id=endpoint.model_id,
                rank=endpoint.instance_rank,
                arena_handle_suffix=endpoint.arena_handle_suffix,
                sequence=sequence,
                dropped=dropped,
                phase_summary=transport_phase_summary(transport_records),
                record_counts={
                    "retained": len(transport_records),
                    "completed": completed,
                    "closed": closed,
                    "incomplete": len(transport_records) - completed - closed,
                },
                records=transport_records,
            )
        elif pe is not None:
            snapshot = FabricSnapshotJson(
                generation=generation.fabric_generation,
                pe=pe,
                atnagent_count=generation.atnagent_count,
                ffnagent_count=generation.ffnagent_count,
                model_topologies=[
                    {"atn_tp_size": t.atn_tp_size, "atn_dp_size": t.atn_dp_size} for t in generation.model_topologies
                ],
                sequence=sequence,
                dropped=dropped,
                records=fabric_records,
            )
        else:
            findings.append(Finding("legacy_fact_missing", "Fabric snapshot PE unavailable"))
    return LegacyProjection(
        request, snapshot, tuple(transport_records) + tuple(fabric_records), report, tuple(findings)
    )
