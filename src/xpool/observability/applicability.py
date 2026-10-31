"""Conservative local protocol paths shared by reconstruction and projections."""

from dataclasses import dataclass, field

from xpool.observability.entities import Observation
from xpool.observability.models import Event, Metadata
from xpool.observability.wire import Delivery, EventKind, EventRole, ResultCode


@dataclass
class BoundaryApplicability:
    """Applicable, excluded and unresolved boundaries with decision evidence."""

    required: set[int] = field(default_factory=set)
    inapplicable: set[int] = field(default_factory=set)
    unknown: set[int] = field(default_factory=set)
    event_sources: dict[int, Event] = field(default_factory=dict)
    metadata_sources: dict[int, tuple[Metadata, str]] = field(default_factory=dict)


def boundary_applicability(observation: Observation, documents: tuple[Metadata, ...]) -> BoundaryApplicability:
    """Resolve only branches justified by this local observation and metadata.

    Missing topology, layer structure or failure-path facts retain unknown
    applicability. Observer disablement never supplies production path facts.
    """
    role = observation.identity.role
    events, facts = observation.events, observation.facts
    path = BoundaryApplicability()

    # Transport terminals exclude their alternative, never restore a lost Hook.
    if observation.identity.entity.kind == "operation":
        instance = set(range(EventKind.REQUEST_STAGING_STARTED, EventKind.REQUEST_PUBLISHED + 1)) | {
            EventKind.RESULT_OBSERVED,
            EventKind.OUTPUT_COPIED,
            EventKind.RESULT_ACKNOWLEDGED,
            EventKind.CLOSED,
        }
        agent = set(range(EventKind.REQUEST_OBSERVED, EventKind.RESULT_PUBLISHED + 1))
        own = instance if role == EventRole.INSTANCE else agent
        path.inapplicable = (instance | agent) - own
        if role == EventRole.INSTANCE:
            ack, closed = events.get(EventKind.RESULT_ACKNOWLEDGED), events.get(EventKind.CLOSED)
            if ack is not None:
                path.required = instance - {EventKind.CLOSED}
                path.inapplicable.add(EventKind.CLOSED)
                path.event_sources[EventKind.CLOSED] = ack
            elif closed is not None or facts.get("result_code") not in (None, ResultCode.OK):
                path.unknown = instance - events.keys() - {EventKind.RESULT_ACKNOWLEDGED}
                path.inapplicable.add(EventKind.RESULT_ACKNOWLEDGED)
                source = closed or observation.fact_sources["result_code"]
                path.event_sources[EventKind.RESULT_ACKNOWLEDGED] = source
            else:
                path.required = instance - {EventKind.CLOSED, EventKind.OUTPUT_COPIED, EventKind.RESULT_ACKNOWLEDGED}
                if facts.get("result_code") == ResultCode.OK:
                    path.required |= {EventKind.OUTPUT_COPIED, EventKind.RESULT_ACKNOWLEDGED}
                    path.inapplicable.add(EventKind.CLOSED)
                    path.event_sources[EventKind.CLOSED] = observation.fact_sources["result_code"]
                else:
                    path.unknown = {EventKind.CLOSED, EventKind.OUTPUT_COPIED, EventKind.RESULT_ACKNOWLEDGED}
        else:
            path.required = agent.copy()
            if facts.get("result_code") not in (None, ResultCode.OK):
                execution_boundaries: set[int] = {EventKind.EXECUTION_STARTED, EventKind.EXECUTION_COMPLETED}
                path.unknown = execution_boundaries - events.keys()
                path.required -= path.unknown
    elif role == EventRole.ATNAGENT:
        path.required = set(range(EventKind.SUBMISSION_PREPARED, EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED + 1))
    elif role == EventRole.COORDINATOR:
        path.required = set(range(EventKind.ENQUEUED, EventKind.LANE_RELEASED + 1))
    else:
        path.required = {
            EventKind.LANE_EXECUTION_OBSERVED,
            EventKind.INPUT_READY_OBSERVED,
            EventKind.COMPUTE_STARTED,
            EventKind.COMPUTE_COMPLETED,
            EventKind.COMPLETION_PUBLISHED,
        }
        routing: set[int] = {EventKind.ROUTING_METADATA_PUBLISHED, EventKind.ROUTING_METADATA_OBSERVED}
        partials: set[int] = {EventKind.PARTIAL_READY_PUBLISHED, EventKind.PEER_PARTIALS_READY_OBSERVED}
        path.unknown = routing | partials
        document = next(
            (
                m
                for m in documents
                if m.run_id == observation.identity.entity.run_id
                and any(g.id == observation.identity.entity.components[0] for g in m.generations)
            ),
            None,
        )
        if document is not None and document.capabilities.routing == "not_applicable":
            path.inapplicable |= routing
            for kind in routing:
                path.metadata_sources[kind] = (document, "capabilities.routing")
        if facts.get("delivery") == Delivery.DIRECT_PARTIAL:
            path.inapplicable |= partials
            for kind in partials:
                path.event_sources[kind] = observation.fact_sources["delivery"]
        elif facts.get("delivery") in (Delivery.SINGLE_COMPLETE, Delivery.REPLICATED_COMPLETE):
            generation_entry = (
                next(((i, g) for i, g in enumerate(document.generations) if g.id == facts.get("generation")), None)
                if document is not None
                else None
            )
            if facts.get("payload_rows") == 0:
                path.inapplicable |= partials
                for kind in partials:
                    path.event_sources[kind] = observation.fact_sources["payload_rows"]
            elif document is not None and generation_entry is not None and generation_entry[1].ffnagent_count == 1:
                # A sole FfnAgent cannot have either a peer or another reader.
                path.inapplicable |= partials
                for kind in partials:
                    path.metadata_sources[kind] = (document, f"generations[{generation_entry[0]}].ffnagent_count")

    # A retained Hook always proves its own applicability, including raw time zero.
    path.required |= events.keys()
    path.inapplicable -= events.keys()
    path.unknown -= path.required | path.inapplicable
    for kind in events:
        path.event_sources.pop(kind, None)
        path.metadata_sources.pop(kind, None)
    return path


transport_times: tuple[str, ...] = tuple(
    EventKind(kind).name.lower() for kind in range(EventKind.REQUEST_STAGING_STARTED, EventKind.CLOSED + 1)
)
fabric_times: dict[EventRole, tuple[str, ...]] = {
    role: tuple(EventKind(kind).name.lower() for kind in range(begin, end + 1))
    for role, begin, end in (
        (EventRole.ATNAGENT, EventKind.SUBMISSION_PREPARED, EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED),
        (EventRole.COORDINATOR, EventKind.ENQUEUED, EventKind.LANE_RELEASED),
        (EventRole.FFNAGENT, EventKind.LANE_EXECUTION_OBSERVED, EventKind.COMPLETION_PUBLISHED),
    )
}
