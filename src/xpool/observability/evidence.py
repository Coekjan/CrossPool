"""Bounded per-snapshot evidence, independent of legacy terminal classification."""

from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from xpool.observability.applicability import boundary_applicability, fabric_times, transport_times
from xpool.observability.codec import kind_names
from xpool.observability.entities import Observation
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.intervals import fabric_pairs, protocol_temporal_conflicts, transport_pairs
from xpool.observability.models import Metadata, StrictModel, positive_uint64, uint64
from xpool.observability.wire import EventKind, EventRole

__all__ = [
    "EndpointReference",
    "EventReference",
    "EvidenceReason",
    "EvidenceStatus",
    "FieldEvidence",
    "InvocationReference",
    "MetadataReference",
    "OperationReference",
    "ProjectionStatus",
    "SnapshotEvidenceEntry",
    "SnapshotEvidenceReport",
    "SnapshotObserver",
    "SnapshotRole",
]


class ProjectionStatus(StrEnum):
    """Membership in the legacy record view."""

    RETAINED = "retained"
    OMITTED = "omitted"
    UNOBSERVED = "unobserved"


class EvidenceStatus(StrEnum):
    """Completeness of the independently retained local evidence."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class EvidenceReason(StrEnum):
    """Contract-defined local reasons, excluding aggregate Producer loss."""

    EVENT_MISSING = "event_missing"
    UNKNOWN_RESULT = "unknown_result"
    LEGACY_FACT_MISSING = "legacy_fact_missing"
    OPERATION_UNOBSERVED = "operation_unobserved"
    APPLICABILITY_UNKNOWN = "applicability_unknown"
    LEGACY_ZERO_TIMESTAMP = "legacy_zero_timestamp"
    TEMPORAL_CONFLICT = "temporal_conflict"


class SnapshotObserver(StrEnum):
    """Legacy observer domain selected by the caller."""

    TRANSPORT = "transport"
    FABRIC = "fabric"


class SnapshotRole(StrEnum):
    """Local role known from the selected site or its own evidence."""

    INSTANCE = "instance"
    ATNAGENT = "atnagent"
    COORDINATOR = "coordinator"
    FFNAGENT = "ffnagent"


class EndpointReference(StrictModel):
    """Run-scoped Transport Endpoint reference."""

    owner: positive_uint64
    sequence: positive_uint64


class OperationReference(StrictModel):
    """Operation identity justified by retained local evidence."""

    endpoint_owner: positive_uint64
    endpoint_sequence: positive_uint64
    operation_sequence: positive_uint64


class InvocationReference(StrictModel):
    """Fabric Invocation identity; independent of the local trace ID."""

    generation: positive_uint64
    instance_index: uint64
    invocation_sequence: positive_uint64


class EventReference(StrictModel):
    """One immutable source Event identity within the report's Run."""

    producer: positive_uint64
    event_sequence: positive_uint64


class MetadataReference(StrictModel):
    """A JSON path within one immutable Producer metadata revision."""

    producer: positive_uint64
    revision: positive_uint64
    path: str


class FieldEvidence(StrictModel):
    """One canonical source for a scalar or applicability decision."""

    field: str
    event: EventReference | None
    metadata: MetadataReference | None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if (self.event is None) == (self.metadata is None):
            raise ValueError("field evidence requires exactly one Event or metadata source")
        return self


class SnapshotEvidenceEntry(StrictModel):
    """Evidence for one eligible local ID, including omitted and unobserved IDs."""

    local_trace_id: positive_uint64
    role: SnapshotRole | None
    operation: OperationReference | None
    invocation: InvocationReference | None
    projection_status: ProjectionStatus
    evidence_status: EvidenceStatus
    unknown_fields: list[str]
    not_applicable_fields: list[str]
    missing_boundaries: list[str]
    unknown_applicability: list[str]
    quality: list[EvidenceReason]
    sources: list[FieldEvidence]

    @model_validator(mode="after")
    def validate_lists(self) -> Self:
        for values in (
            self.unknown_fields,
            self.not_applicable_fields,
            self.missing_boundaries,
            self.unknown_applicability,
            self.quality,
            [source.field for source in self.sources],
        ):
            if values != sorted(set(values)):
                raise ValueError("evidence lists must be unique and lexically sorted")
        if set(self.unknown_fields) & set(self.not_applicable_fields):
            raise ValueError("a field cannot be both unknown and inapplicable")
        return self


class SnapshotEvidenceReport(StrictModel):
    """Contract v4 side report for one read cutoff; offline hashes remain null."""

    contract_version: Literal[4] = 4
    run_id: str
    producer_id: positive_uint64
    observer: SnapshotObserver
    generation: positive_uint64
    endpoint: EndpointReference | None
    site: Literal["instance", "atnagent"] | None
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    capacity: positive_uint64
    sequence: uint64 | None
    dropped: uint64 | None
    unobserved_operations: uint64 | None
    entries: list[SnapshotEvidenceEntry]

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("report Run UUID must be canonical")
        transport = self.observer == SnapshotObserver.TRANSPORT
        if transport != (self.endpoint is not None) or transport != (self.site is not None):
            raise ValueError("snapshot report observer/domain mismatch")
        ids = [entry.local_trace_id for entry in self.entries]
        if ids != sorted(set(ids)) or any(local_id > self.capacity for local_id in ids):
            raise ValueError("report entries must be unique sorted eligible IDs")
        if self.sequence is None:
            if self.dropped is not None or self.unobserved_operations is not None:
                raise ValueError("unknown sequence requires unknown operation counts")
        elif (
            self.dropped != max(self.sequence - self.capacity, 0)
            or ids != list(range(1, min(self.sequence, self.capacity) + 1))
            or self.unobserved_operations
            != sum(e.projection_status == ProjectionStatus.UNOBSERVED for e in self.entries)
        ):
            raise ValueError("report counts do not match eligible entries")
        return self


def metadata_evidence(field: str, document: Metadata, path: str) -> FieldEvidence:
    return FieldEvidence(
        field=field,
        event=None,
        metadata=MetadataReference(producer=document.producer_id, revision=document.revision, path=path),
    )


def unobserved_entry(local_id: int, site: Literal["instance", "atnagent"] | None) -> SnapshotEvidenceEntry:
    return SnapshotEvidenceEntry(
        local_trace_id=local_id,
        role=SnapshotRole(site) if site is not None else None,
        operation=None,
        invocation=None,
        projection_status=ProjectionStatus.UNOBSERVED,
        evidence_status=EvidenceStatus.UNAVAILABLE,
        unknown_fields=[],
        not_applicable_fields=[],
        missing_boundaries=[],
        unknown_applicability=[],
        quality=[EvidenceReason.OPERATION_UNOBSERVED],
        sources=[],
    )


def evidence_entry(
    observation: Observation,
    documents: tuple[Metadata, ...],
    *,
    retained: bool,
) -> SnapshotEvidenceEntry:
    """Project finite fields and their canonical sources without inferring timestamps."""
    role, identity = observation.identity.role, observation.identity.entity
    transport = identity.kind == "operation"
    generation_id = (
        identity.components[0]
        if not transport
        else next(
            (
                e.generation
                for m in documents
                if m.run_id == identity.run_id
                for e in m.endpoints
                if (e.owner_producer, e.endpoint_sequence) == identity.components[:2]
            ),
            None,
        )
    )
    generation_reference = next(
        (
            (m, i, g)
            for m in documents
            if m.run_id == identity.run_id
            for i, g in enumerate(m.generations)
            if g.id == generation_id
        ),
        None,
    )
    if generation_reference is None:
        raise ObservationError(
            ErrorCode.METADATA_UNAVAILABLE,
            "snapshot evidence Generation unavailable",
            producer=observation.identity.producer,
        )
    document, generation_index, generation = generation_reference
    path = boundary_applicability(observation, documents)
    unknown: set[str] = set()
    inapplicable: set[str] = set()
    sources: dict[str, FieldEvidence] = {}
    reasons: set[EvidenceReason] = set()

    def add_event(field: str, event_kind: int | None = None, fact: str | None = None) -> None:
        event = (
            observation.events.get(event_kind)
            if event_kind is not None
            else observation.fact_sources.get(fact or field)
        )
        if event is None:
            unknown.add(field)
        else:
            sources[field] = FieldEvidence(
                field=field,
                event=EventReference(
                    producer=event.source.identity.producer,
                    event_sequence=event.record.event_sequence,
                ),
                metadata=None,
            )

    # Selected-role scalar fields, including nullable assignment facts.
    scalars = {name: name for name in ("layer_ordinal", "payload_rows", "output_requirement")}
    if transport:
        scalars |= {
            "trace_id": "local_trace_id",
            "forward_mode": "forward_mode",
            "dp_row_layout": "dp_row_layout",
            "result_code": "result_code",
        }
        times = dict(enumerate(transport_times, EventKind.REQUEST_STAGING_STARTED))
    else:
        scalars |= {name: name for name in ("local_trace_id", "instance_index", "invocation_sequence")}
        scalars |= {"facts.executor_lane_index": "lane_index", "facts.executor_lease_sequence": "lease_sequence"}
        if role == EventRole.ATNAGENT:
            scalars |= {
                "dp_row_layout": "dp_row_layout",
                "facts.forward_mode": "forward_mode",
                "facts.dp_rank_payload_rows": "dp_rank_payload_rows",
            }
        else:
            inapplicable.add("dp_row_layout")
        if role == EventRole.COORDINATOR:
            sources["facts.scheduler.policy"] = metadata_evidence(
                "facts.scheduler.policy",
                document,
                f"generations[{generation_index}].scheduler",
            )
            if generation.scheduler == "fifo":
                scalars["facts.scheduler.ready_ticket"] = "ready_ticket"
            else:
                inapplicable.add("facts.scheduler.ready_ticket")
        elif role == EventRole.FFNAGENT:
            scalars |= {"facts.payload_row_capacity": "payload_row_capacity", "facts.delivery": "delivery"}
        start = {
            EventRole.ATNAGENT: EventKind.SUBMISSION_PREPARED,
            EventRole.COORDINATOR: EventKind.ENQUEUED,
            EventRole.FFNAGENT: EventKind.LANE_EXECUTION_OBSERVED,
        }[role]
        times = {kind: f"events_ns.{name}" for kind, name in enumerate(fabric_times[role], start)}
    for field, fact in scalars.items():
        add_event(field, fact=fact)

    # Timestamp availability and conditional decisions retain independent sources.
    for kind, field in times.items():
        if kind in path.inapplicable:
            inapplicable.add(field)
        else:
            add_event(field, event_kind=kind)
    for kind, event in path.event_sources.items():
        field = f"boundary.{kind_names[kind]}"
        sources[field] = FieldEvidence(
            field=field,
            event=EventReference(
                producer=event.source.identity.producer,
                event_sequence=event.record.event_sequence,
            ),
            metadata=None,
        )
    for kind, (source_document, metadata_path) in path.metadata_sources.items():
        field = f"boundary.{kind_names[kind]}"
        sources[field] = metadata_evidence(field, source_document, metadata_path)
    missing = path.required - observation.events.keys()
    if missing:
        reasons.add(EvidenceReason.EVENT_MISSING)
    if path.unknown:
        reasons.add(EvidenceReason.APPLICABILITY_UNKNOWN)
    if transport and "result_code" not in observation.facts:
        reasons.add(EvidenceReason.UNKNOWN_RESULT)
    if not retained:
        reasons.add(EvidenceReason.LEGACY_FACT_MISSING)
    if any(e.record.raw_time_ns == 0 for e in observation.events.values()):
        reasons.add(EvidenceReason.LEGACY_ZERO_TIMESTAMP)

    # Raw zero remains observed. Reversed/cross-domain phase endpoints are unusable.
    pairs = transport_pairs if transport else fabric_pairs[role]
    unusable = bool(protocol_temporal_conflicts(observation))
    if unusable:
        reasons.add(EvidenceReason.TEMPORAL_CONFLICT)
    for _, start, end in pairs:
        begin, finish = observation.events.get(start), observation.events.get(end)
        if begin is not None and finish is not None:
            if begin.record.clock_domain_id != finish.record.clock_domain_id:
                unusable = True
            elif finish.record.raw_time_ns < begin.record.raw_time_ns:
                reasons.add(EvidenceReason.TEMPORAL_CONFLICT)
                unusable = True
    complete = not unknown and not path.unknown and not unusable and retained
    components = identity.components
    return SnapshotEvidenceEntry(
        local_trace_id=observation.facts["local_trace_id"],
        role=SnapshotRole(role.name.lower()),
        operation=OperationReference(
            endpoint_owner=components[0], endpoint_sequence=components[1], operation_sequence=components[2]
        )
        if transport
        else None,
        invocation=None
        if transport
        else InvocationReference(
            generation=components[0], instance_index=components[1], invocation_sequence=components[2]
        ),
        projection_status=ProjectionStatus.RETAINED if retained else ProjectionStatus.OMITTED,
        evidence_status=EvidenceStatus.COMPLETE if complete else EvidenceStatus.PARTIAL,
        unknown_fields=sorted(unknown),
        not_applicable_fields=sorted(inapplicable),
        missing_boundaries=sorted(kind_names[k] for k in missing),
        unknown_applicability=sorted(kind_names[k] for k in path.unknown),
        quality=sorted(reasons),
        sources=[sources[field] for field in sorted(sources)],
    )
