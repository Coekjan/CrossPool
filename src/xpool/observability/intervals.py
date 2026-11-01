"""Observed same-domain intervals and local protocol precedence."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from xpool.observability import phases
from xpool.observability.entities import Observation, ObservationIdentity
from xpool.observability.errors import ErrorCode
from xpool.observability.models import Event, Finding
from xpool.observability.provenance import Source
from xpool.observability.wire import (
    EventKind,
    EventRole,
    FabricEventKind,
    GraphError,
    GraphEventKind,
    TransportEventKind,
)

__all__ = ["Interval"]


@dataclass(frozen=True)
class Interval:
    """A same-domain observed interval, or an explicit unavailable boundary pair."""

    observation: ObservationIdentity
    name: str
    begin: Event | None
    end: Event | None
    status: Literal["complete", "partial", "conflict", "unavailable"]
    duration_ns: int | None
    reasons: tuple[str, ...]

    @property
    def sources(self) -> tuple[Source, ...]:
        """Return the concrete observed endpoints, including partial evidence."""
        return tuple(event.source for event in (self.begin, self.end) if event is not None)


# Phase entries are (phase name, begin Event kind, end Event kind); fabric_pairs groups them by local role.
# Translate the shared timestamp field names into raw wire boundaries.
transport_pairs = tuple(
    (name, TransportEventKind[start.upper()], TransportEventKind[end.upper()])
    for name, start, end in phases.transport_phases
)
fabric_pairs = {
    role: tuple(
        (name, FabricEventKind[start.upper()], FabricEventKind[end.upper()]) for name, start, end in definitions
    )
    for role, definitions in (
        (EventRole.ATNAGENT, phases.fabric_atnagent_phases),
        (EventRole.COORDINATOR, phases.fabric_coordinator_phases),
        (EventRole.FFNAGENT, phases.fabric_ffnagent_phases),
    )
}
graph_pairs = (
    ("capture", GraphEventKind.GRAPH_CAPTURE_BEGIN, GraphEventKind.GRAPH_CAPTURE_END),
    ("installation", GraphEventKind.GRAPH_INSTALL_BEGIN, GraphEventKind.GRAPH_INSTALL_END),
    ("host_call", GraphEventKind.GRAPH_HOST_CALL_BEGIN, GraphEventKind.GRAPH_HOST_CALL_END),
    ("device_activity", GraphEventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, GraphEventKind.GRAPH_DEVICE_ACTIVITY_END),
)
graph_failure_kinds: dict[int, EventKind] = {
    GraphEventKind.GRAPH_CAPTURE_END: GraphEventKind.GRAPH_CAPTURE_ERROR,
    GraphEventKind.GRAPH_INSTALL_END: GraphEventKind.GRAPH_INSTALL_ERROR,
    GraphEventKind.GRAPH_HOST_CALL_END: GraphEventKind.GRAPH_HOST_CALL_ERROR,
}


def phase_interval(observation: Observation, name: str, begin_kind: int, end_kind: int) -> Interval:
    begin = observation.events.get(begin_kind)
    end = observation.events.get(end_kind)

    failure_kind = graph_failure_kinds.get(end_kind)
    failure = observation.events.get(failure_kind) if failure_kind is not None else None

    if failure is not None:
        if end is not None:
            raise failure.error(ErrorCode.EVENT_CONFLICT, "Graph has both success and failure boundaries")

        return Interval(observation.identity, name, begin, failure, "unavailable", None, ("graph_error",))

    if observation.identity.entity.kind in ("graph_template", "graph_execution") and any(
        event is not None and event.record.data4 != GraphError.NONE for event in (begin, end)
    ):
        return Interval(observation.identity, name, begin, end, "unavailable", None, ("graph_error",))

    if begin is None or end is None:
        return Interval(observation.identity, name, begin, end, "partial", None, ("event_missing",))

    if begin.record.clock_domain_id != end.record.clock_domain_id:
        return Interval(observation.identity, name, begin, end, "unavailable", None, ("clock_domain_mismatch",))

    if end.record.raw_time_ns < begin.record.raw_time_ns:
        return Interval(observation.identity, name, begin, end, "conflict", None, ("temporal_conflict",))

    return Interval(
        observation.identity, name, begin, end, "complete", end.record.raw_time_ns - begin.record.raw_time_ns, ()
    )


def protocol_temporal_conflicts(observation: Observation) -> tuple[Finding, ...]:
    """Check local Hook precedence even when no derived phase spans two Hooks."""
    role = observation.identity.role
    order: tuple[int, ...]
    if observation.identity.entity.kind == "operation":
        order = (
            (
                TransportEventKind.REQUEST_STAGING_STARTED,
                TransportEventKind.REQUEST_STAGING_COMPLETED,
                TransportEventKind.REQUEST_PUBLISHED,
                TransportEventKind.RESULT_OBSERVED,
                TransportEventKind.OUTPUT_COPIED,
                TransportEventKind.RESULT_ACKNOWLEDGED,
                TransportEventKind.CLOSED,
            )
            if role == EventRole.INSTANCE
            else (
                TransportEventKind.REQUEST_OBSERVED,
                TransportEventKind.EXECUTION_STARTED,
                TransportEventKind.EXECUTION_COMPLETED,
                TransportEventKind.RESULT_PUBLISHED,
            )
        )
    elif role == EventRole.ATNAGENT:
        order = tuple(range(FabricEventKind.SUBMISSION_PREPARED, FabricEventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED + 1))
    elif role == EventRole.COORDINATOR:
        order = tuple(range(FabricEventKind.ENQUEUED, FabricEventKind.LANE_RELEASED + 1))
    else:
        # The publisher releases routing from inside compute; peers observe it
        # before compute. Enum ordinals alone do not express this partial order.
        order = (
            FabricEventKind.LANE_EXECUTION_OBSERVED,
            FabricEventKind.INPUT_READY_OBSERVED,
            FabricEventKind.ROUTING_METADATA_OBSERVED,
            FabricEventKind.COMPUTE_STARTED,
            FabricEventKind.ROUTING_METADATA_PUBLISHED,
            FabricEventKind.COMPUTE_COMPLETED,
            FabricEventKind.PARTIAL_READY_PUBLISHED,
            FabricEventKind.PEER_PARTIALS_READY_OBSERVED,
            FabricEventKind.COMPLETION_PUBLISHED,
        )
    retained = [observation.events[k] for k in order if k in observation.events]
    return tuple(
        Finding("temporal_conflict", "local protocol Hook order reversed", (begin.source, end.source))
        for begin, end in pairwise(retained)
        if begin.record.clock_domain_id == end.record.clock_domain_id
        and begin.record.raw_time_ns > end.record.raw_time_ns
    )
