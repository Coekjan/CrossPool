"""Identity-based offline reconstruction, independent of calibrated time."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from xpool.observability.applicability import boundary_applicability
from xpool.observability.artifacts import ArtifactSet, AttachmentEvidence, resolve_attachments
from xpool.observability.codec import kind_names, validate_event
from xpool.observability.entities import (
    CausalEdge,
    CausalRelation,
    Entity,
    EntityGraph,
    EntityIdentity,
    EntityKind,
    Observation,
)
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.intervals import (
    Interval,
    fabric_pairs,
    graph_failure_kinds,
    graph_pairs,
    phase_interval,
    protocol_temporal_conflicts,
    transport_pairs,
)
from xpool.observability.models import Event, Finding, Metadata, PairedClockSample, QualityCheckpoint
from xpool.observability.provenance import EventIdentity, Source
from xpool.observability.quality import QualityReport, summarize_quality
from xpool.observability.wire import (
    EventKind,
    EventRole,
    GraphError,
    PresenceBit,
    SampleFailure,
)

__all__ = ["Reconstruction", "reconstruct"]


@dataclass(frozen=True)
class Reconstruction:
    """Evidence inventory consumed by offline export and snapshot projection."""

    events: dict[EventIdentity, Event]
    event_sources: dict[EventIdentity, tuple[Source, ...]]
    entities: dict[EntityIdentity, Entity]
    edges: tuple[CausalEdge, ...]
    intervals: tuple[Interval, ...]
    metadata: tuple[Metadata, ...]
    quality: QualityReport
    calibration_events: tuple[Event, ...]
    attachments: tuple[AttachmentEvidence, ...]
    calibration_samples: tuple[PairedClockSample, ...]

    def of_kind(self, kind: EntityKind) -> tuple[Entity, ...]:
        """Return all retained identities of a contract entity family."""
        return tuple(entity for key, entity in self.entities.items() if key.kind == kind)


def check_metadata_revisions(metadata: Iterable[Metadata]) -> tuple[Metadata, ...]:
    """Reject changed immutable references across revisions, processes and chunks."""
    retained: dict[tuple[str, int, int], Metadata] = {}
    definitions: dict[tuple[str, str, tuple[int, ...]], str] = {}

    for document in metadata:
        identity = (document.run_id, document.producer_id, document.revision)

        if identity in retained and retained[identity] != document:
            raise ObservationError(
                ErrorCode.IDENTITY_CONFLICT, "conflicting metadata revision", producer=document.producer_id
            )

        if identity in retained:
            continue

        retained[identity] = document

        for family, entries in (
            ("producer", [((x.id,), x.model_dump_json()) for x in document.producers]),
            ("generation", [((x.id,), x.model_dump_json()) for x in document.generations]),
            ("clock", [((x.id,), x.model_dump_json()) for x in document.clock_domains]),
            ("endpoint", [((x.owner_producer, x.endpoint_sequence), x.model_dump_json()) for x in document.endpoints]),
            (
                "template",
                [((x.owner_producer, x.template_sequence), x.model_dump_json()) for x in document.graph_templates],
            ),
            ("artifact", [((document.producer_id, x.id), x.model_dump_json()) for x in document.artifacts]),
            (
                "policy",
                [((document.producer_id,), document.model_dump_json(include={"capabilities", "configuration"}))],
            ),
        ):
            for components, value in entries:
                key = (document.run_id, family, components)
                if key in definitions and definitions[key] != value:
                    field_path = family
                    if family == "generation":
                        index = next(i for i, g in enumerate(document.generations) if g.id == components[0])
                        field_path = f"generations[{index}].pe_members"
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT,
                        f"changed immutable {family} {components}",
                        producer=document.producer_id,
                        generation=components[0] if family == "generation" else None,
                        metadata_field_path=field_path,
                    )
                definitions[key] = value

    return tuple(retained.values())


@dataclass
class EventInventory:
    events: dict[EventIdentity, Event]
    locations: dict[EventIdentity, list[Source]]
    documents: tuple[Metadata, ...]


def retain_events(events: Iterable[Event], metadata: Iterable[Metadata], findings: list[Finding]) -> EventInventory:
    """Validate provenance and deduplicate identities while retaining every location."""
    retained: dict[EventIdentity, Event] = {}
    locations: dict[EventIdentity, list[Source]] = defaultdict(list)
    documents = list(metadata)

    for event in events:
        validate_event(event)
        event_identity = event.source.identity

        if (event_identity.run_id, event_identity.producer) != (event.metadata.run_id, event.metadata.producer_id):
            raise event.error(ErrorCode.IDENTITY_CONFLICT, "event provenance disagrees with metadata")

        if (
            event_identity.event_sequence != event.record.event_sequence
            or event.source.process_creation_id != event.metadata.process.creation_id
            or event.source.clock_domain_id != event.record.clock_domain_id
            or event.source.endpoint
            != (
                (event.record.endpoint_owner, event.record.endpoint_sequence)
                if event.record.has(PresenceBit.ENDPOINT)
                else None
            )
            or event.source.instance_rank
            != (event.record.instance_rank if event.record.has(PresenceBit.INSTANCE_RANK) else None)
            or event.source.pe != (event.record.pe if event.record.has(PresenceBit.PE) else None)
        ):
            raise event.error(ErrorCode.IDENTITY_CONFLICT, "source/event sequence mismatch")

        documents.append(event.metadata)
        previous = retained.get(event_identity)

        if previous is not None:
            if previous.record != event.record:
                raise event.error(
                    ErrorCode.EVENT_CONFLICT,
                    "event identity has different bytes",
                    sources=(previous.source, event.source),
                )
            findings.append(Finding("duplicate_event", "repeated event identity", (previous.source, event.source)))
        else:
            retained[event_identity] = event

        locations[event_identity].append(event.source)

    return EventInventory(retained, locations, check_metadata_revisions(documents))


def calibration_samples(samples: Iterable[Event]) -> tuple[PairedClockSample, ...]:
    """Project validated sample Events while retaining raw domains and failed reads."""
    result: list[PairedClockSample] = []
    for event in samples:
        r = event.record
        domain = next(domain for domain in event.metadata.clock_domains if domain.id == r.data3)
        # validate_event established a Device domain with a positive Epoch.
        if domain.raw_epoch is None:
            raise event.error(ErrorCode.INVALID_EVENT, "sample Device Epoch unavailable")
        succeeded = r.kind == EventKind.CALIBRATION_SAMPLE_SUCCEEDED
        result.append(
            PairedClockSample(
                event.source,
                r.data4,
                r.data3,
                domain.raw_epoch,
                r.data0,
                r.data1 if succeeded else None,
                r.data2 if succeeded else None,
                "succeeded" if succeeded else "failed",
                None if succeeded else SampleFailure(r.data2).code,
            )
        )
    return tuple(result)


def reconstruct_entities(events: Iterable[Event], findings: list[Finding]) -> tuple[EntityGraph, tuple[Event, ...]]:
    """Build entity/site facts and identity edges in canonical Event order."""
    graph = EntityGraph()
    samples: list[Event] = []

    for event in sorted(events, key=lambda item: item.source.identity):
        r = event.record
        run_id = event.source.identity.run_id
        keys: dict[str, EntityIdentity] = {}

        identity_groups: tuple[tuple[PresenceBit, EntityKind, tuple[int, ...]], ...] = (
            (PresenceBit.OPERATION, "operation", (r.endpoint_owner, r.endpoint_sequence, r.operation_sequence)),
            (PresenceBit.INVOCATION, "invocation", (r.generation, r.instance_index, r.invocation_sequence)),
            (PresenceBit.GRAPH_TEMPLATE, "graph_template", (r.graph_owner, r.graph_template)),
            (PresenceBit.GRAPH_EXECUTION, "graph_execution", (r.graph_owner, r.graph_execution)),
            (PresenceBit.LEASE, "lease", (r.generation, r.lane_index, r.lease_sequence)),
            (PresenceBit.ATTACHMENT, "attachment", (r.attachment_owner, r.attachment_sequence)),
        )

        for bit, entity_kind, components in identity_groups:
            if r.has(bit):
                identity = EntityIdentity(run_id, entity_kind, components)
                keys[entity_kind] = identity
                entity = graph.entities.setdefault(identity, Entity(identity))
                entity.add(event)

        # Only the owning family mutates a local observation. Association facts
        # remain evidence for edges and cannot manufacture a Fabric boundary.
        family = (
            "operation"
            if EventKind.REQUEST_STAGING_STARTED <= r.kind <= EventKind.CLOSED
            else "invocation"
            if EventKind.SUBMISSION_PREPARED <= r.kind <= EventKind.COMPLETION_PUBLISHED
            else "graph_template"
            if r.kind
            in (
                EventKind.GRAPH_CAPTURE_BEGIN,
                EventKind.GRAPH_CAPTURE_END,
                EventKind.GRAPH_CAPTURE_ERROR,
                EventKind.GRAPH_TEMPLATE_INSTALLED,
                EventKind.GRAPH_TEMPLATE_RETIRED,
                EventKind.GRAPH_INSTALL_BEGIN,
                EventKind.GRAPH_INSTALL_END,
                EventKind.GRAPH_INSTALL_ERROR,
            )
            else "graph_execution"
            if EventKind.GRAPH_HOST_CALL_BEGIN <= r.kind <= EventKind.GRAPH_DEVICE_ACTIVITY_END
            else None
        )

        if family is not None:
            entity = graph.entities[keys[family]]
            entity.observe(event, findings)

        # Auxiliary Graph evidence recovers only explicitly carried local facts.
        # It never enters the protocol boundary map or borrows another site's ID.
        if family == "graph_execution" and r.has(PresenceBit.LOCAL_TRACE_ID):
            auxiliary_family = "invocation" if r.role == EventRole.FFNAGENT else "operation"
            auxiliary_key = keys.get(auxiliary_family)
            if auxiliary_key is not None:
                local_site = auxiliary_family == "invocation" or any(
                    (e.owner_producer, e.endpoint_sequence) == auxiliary_key.components[:2]
                    and any(site.producer == event.source.identity.producer for site in e.sites)
                    for e in event.metadata.endpoints
                )
                if local_site:
                    entity = graph.entities[auxiliary_key]
                    entity.observe(event, findings, boundary=False)

        relations: tuple[tuple[CausalRelation, EntityKind, EntityKind, bool], ...] = (
            ("operation_to_invocation", "operation", "invocation", r.kind == EventKind.TRANSPORT_FABRIC_ASSOCIATED),
            ("graph_to_operation", "graph_execution", "operation", True),
            (
                "invocation_to_graph_activity",
                "invocation",
                "graph_execution",
                r.kind in (EventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, EventKind.GRAPH_DEVICE_ACTIVITY_END),
            ),
            ("execution_of_template", "graph_execution", "graph_template", True),
            ("invocation_lease", "invocation", "lease", True),
            ("invocation_attachment", "invocation", "attachment", r.kind == EventKind.ROUTING_PUBLISHED),
        )
        for relation, left, right, condition in relations:
            if condition and left in keys and right in keys:
                graph.link(relation, keys[left], keys[right], (event.source,))

        if r.kind in (EventKind.CALIBRATION_SAMPLE_SUCCEEDED, EventKind.CALIBRATION_SAMPLE_FAILED):
            samples.append(event)
            if r.kind == EventKind.CALIBRATION_SAMPLE_FAILED:
                findings.append(Finding("sample_failed", "paired sampling attempt failed", (event.source,)))

        if EventKind.GRAPH_CAPTURE_BEGIN <= r.kind <= EventKind.GRAPH_INSTALL_ERROR and r.data4:
            findings.append(
                Finding(GraphError(r.data4).name.lower(), "Graph evidence unavailable or failed", (event.source,))
            )

    graph.validate_associations()
    return graph, tuple(samples)


def observation_intervals(
    observation: Observation, documents: tuple[Metadata, ...], findings: list[Finding]
) -> tuple[Interval, ...]:
    """Resolve local phase applicability and retain explicit incomplete intervals."""
    intervals: list[Interval] = []
    if observation.identity.entity.kind == "operation":
        role = observation.identity.role
        pairs = tuple(
            pair
            for pair in transport_pairs
            if (role == EventRole.INSTANCE and pair[1] not in (EventKind.REQUEST_OBSERVED, EventKind.EXECUTION_STARTED))
            or (role == EventRole.ATNAGENT and pair[1] in (EventKind.REQUEST_OBSERVED, EventKind.EXECUTION_STARTED))
        )
        if "result_code" not in observation.facts:
            findings.append(Finding("unknown_result", "site-local result unavailable", tuple(observation.sources)))
    elif observation.identity.entity.kind == "invocation":
        pairs = fabric_pairs[observation.identity.role]
    else:
        pairs = tuple(
            pair
            for pair in graph_pairs
            if pair[1] in observation.events
            or pair[2] in observation.events
            or ((failure_kind := graph_failure_kinds.get(pair[2])) is not None and failure_kind in observation.events)
        )
    if observation.identity.entity.kind in ("operation", "invocation"):
        findings.extend(protocol_temporal_conflicts(observation))
        applicability = boundary_applicability(observation, documents)
        missing = applicability.required - observation.events.keys()
        if missing:
            findings.append(
                Finding(
                    "event_missing",
                    "missing boundaries: " + ", ".join(kind_names[x] for x in sorted(missing)),
                    tuple(observation.sources),
                )
            )
        if applicability.unknown:
            findings.append(
                Finding("applicability_unknown", "conditional protocol path unresolved", tuple(observation.sources))
            )
    for name, start, end in pairs:
        # Closed and normal acknowledged totals are alternative paths.
        if (
            observation.identity.entity.kind == "operation"
            and name == "closed_total"
            and EventKind.CLOSED not in observation.events
        ):
            continue
        if observation.identity.entity.kind in ("operation", "invocation") and (
            start in applicability.inapplicable or end in applicability.inapplicable
        ):
            continue
        interval = phase_interval(observation, name, start, end)
        if (
            observation.identity.entity.kind in ("operation", "invocation")
            and interval.reasons == ("event_missing",)
            and (start in applicability.unknown or end in applicability.unknown)
        ):
            interval = Interval(
                interval.observation,
                name,
                interval.begin,
                interval.end,
                "partial",
                None,
                ("applicability_unknown",),
            )
        intervals.append(interval)
        for reason in interval.reasons:
            findings.append(Finding(reason, f"{name}: incomplete or invalid boundaries", interval.sources))

    return tuple(intervals)


def reconstruct_intervals(
    graph: EntityGraph, documents: tuple[Metadata, ...], findings: list[Finding]
) -> tuple[Interval, ...]:
    """Retain local boundaries and applicability gaps without cross-domain inference."""
    intervals: list[Interval] = []
    associated_operations = {edge.origin for edge in graph.edges.values() if edge.relation == "operation_to_invocation"}
    for identity, entity in graph.entities.items():
        if identity.kind == "operation":
            endpoint = next(
                e
                for m in documents
                if m.run_id == identity.run_id
                for e in m.endpoints
                if (e.owner_producer, e.endpoint_sequence) == identity.components[:2]
            )
            observed_producers = {observation.identity.producer for observation in entity.observations.values()}
            if any(site.producer not in observed_producers for site in endpoint.sites):
                findings.append(
                    Finding("event_missing", "Endpoint site observation unavailable", tuple(entity.sources))
                )
        if identity.kind == "graph_execution" and not entity.observations:
            findings.append(Finding("event_missing", "Graph activity boundaries unavailable", tuple(entity.sources)))
        if identity.kind == "operation" and identity not in associated_operations:
            # A retained explicit rejection/closure can legitimately precede Fabric.
            rejected = any(EventKind.CLOSED in observation.events for observation in entity.observations.values())
            if not rejected:
                findings.append(
                    Finding("association_missing", "no retained Fabric-entry association", tuple(entity.sources))
                )
        for observation in entity.observations.values():
            intervals.extend(observation_intervals(observation, documents, findings))
    return tuple(intervals)


def link_protocol_boundaries(graph: EntityGraph) -> None:
    """Link actual cross-role endpoints when Invocation and retained Lease agree."""
    # Cross-role protocol edges require the exact Invocation, matching Lease
    # where both sides carry one, and both actual observed boundaries.
    protocol_pairs = (
        (EventKind.SUBMISSION_PUBLISHED, EventKind.ENQUEUED),
        (EventKind.ADMISSION_PUBLISHED, EventKind.ADMISSION_OBSERVED),
        (EventKind.LANE_EXECUTION_PUBLISHED, EventKind.LANE_EXECUTION_OBSERVED),
        (EventKind.INPUT_READY_PUBLISHED, EventKind.INPUT_READY_OBSERVED),
        (EventKind.COMPLETION_PUBLISHED, EventKind.FFNAGENT_COMPLETIONS_OBSERVED),
        (EventKind.OUTPUT_COMMIT_PUBLISHED, EventKind.OUTPUT_COMMIT_OBSERVED),
        (EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED, EventKind.OUTPUT_ACKNOWLEDGEMENTS_OBSERVED),
    )
    for entity in graph.entities.values():
        if entity.identity.kind != "invocation":
            continue
        by_kind: dict[int, list[Event]] = defaultdict(list)
        for observation in entity.observations.values():
            for kind, event in observation.events.items():
                by_kind[kind].append(event)
        for published_kind, observed_kind in protocol_pairs:
            for before in by_kind[published_kind]:
                for after in by_kind[observed_kind]:
                    a, b = before.record, after.record
                    if (
                        a.has(PresenceBit.LEASE)
                        and b.has(PresenceBit.LEASE)
                        and (a.lane_index, a.lease_sequence) != (b.lane_index, b.lease_sequence)
                    ):
                        continue
                    graph.link("protocol", before.source.identity, after.source.identity, (before.source, after.source))


def reconstruct(
    events: Iterable[Event] | ArtifactSet,
    *,
    metadata: Iterable[Metadata] = (),
    checkpoints: Iterable[QualityCheckpoint] = (),
) -> Reconstruction:
    """Reconstruct entities and causality exclusively from protocol identities.

    Inputs are consumed once. Retained identities and evidence grow with input
    size; runtime collection budgets do not bound this offline index. Metadata
    order selects equivalent provenance revisions. Raw clocks remain unfitted.

    Returns:
        Independently retained local observations, explicitly evidenced edges,
        raw intervals, cumulative quality and calibration samples.

    Raises:
        ObservationError: For malformed events, conflicting identities or facts.
    """
    artifact_set = events if isinstance(events, ArtifactSet) else None

    event_input: Iterable[Event]
    if isinstance(events, ArtifactSet):
        metadata = (*metadata, *events.metadata)
        checkpoints = (*checkpoints, *events.checkpoints)
        event_input = events.events
    else:
        event_input = events

    findings: list[Finding] = list(artifact_set.findings) if artifact_set else []
    attachments = artifact_set.attachments if artifact_set else ()
    # Establish input identities before constructing cross-event relationships.
    inventory = retain_events(event_input, metadata, findings)
    retained, locations, documents_tuple = inventory.events, inventory.locations, inventory.documents

    if artifact_set is None:
        attachments, attachment_findings = resolve_attachments((), None, tuple(retained.values()), documents_tuple)
        findings.extend(attachment_findings)

    # Resolve local evidence first; protocol links require retained endpoints.
    graph, samples = reconstruct_entities(retained.values(), findings)
    intervals = reconstruct_intervals(graph, documents_tuple, findings)
    link_protocol_boundaries(graph)
    return Reconstruction(
        retained,
        {key: tuple(value) for key, value in locations.items()},
        graph.entities,
        tuple(graph.edges.values()),
        intervals,
        documents_tuple,
        summarize_quality(retained.values(), documents_tuple, checkpoints, findings),
        samples,
        attachments,
        calibration_samples(samples),
    )
