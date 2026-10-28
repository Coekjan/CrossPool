"""Protocol entities and their independently retained local observations."""

from dataclasses import dataclass, field
from typing import Literal

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import Event, Finding
from xpool.observability.provenance import EventIdentity, Source
from xpool.observability.wire import EventRole

__all__ = [
    "CausalEdge",
    "CausalRelation",
    "Entity",
    "EntityIdentity",
    "EntityKind",
    "Observation",
    "ObservationIdentity",
]


type CausalRelation = Literal[
    "operation_to_invocation",
    "graph_to_operation",
    "invocation_to_graph_activity",
    "execution_of_template",
    "invocation_lease",
    "invocation_attachment",
    "protocol",
]


type EntityKind = Literal["operation", "invocation", "graph_template", "graph_execution", "lease", "attachment"]


@dataclass(frozen=True, order=True)
class EntityIdentity:
    """Run-scoped contract identity; components never contain local trace IDs."""

    run_id: str
    kind: EntityKind
    components: tuple[int, ...]


@dataclass(frozen=True, order=True)
class ObservationIdentity:
    """A local site/role's observation, isolated from other participants."""

    entity: EntityIdentity
    producer: int
    role: EventRole


@dataclass
class Observation:
    """Retained boundaries and repeated facts at one actual observing site."""

    identity: ObservationIdentity
    facts: dict[str, int] = field(default_factory=dict)
    events: dict[int, Event] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    fact_sources: dict[str, Event] = field(default_factory=dict)

    def add(self, event: Event, findings: list[Finding], *, boundary: bool = True) -> None:
        """Merge equal repeated facts and retain distinct boundary evidence."""
        record = event.record
        previous = self.events.get(record.kind)

        if previous is not None:
            # event_sequence is transport identity, not a semantic fact.
            before = previous.record.model_dump(exclude={"event_sequence"})
            after = record.model_dump(exclude={"event_sequence"})

            if before != after:
                raise event.error(
                    ErrorCode.EVENT_CONFLICT,
                    "conflicting boundary at the same observation",
                    sources=(previous.source, event.source),
                )

            findings.append(Finding("duplicate_event", "identical semantic boundary", (previous.source, event.source)))
        elif boundary:
            self.events[record.kind] = event

        for name, value in record.present_facts().items():
            if name in self.facts and self.facts[name] != value:
                raise event.error(
                    ErrorCode.IDENTITY_CONFLICT,
                    f"conflicting fact {name}",
                    sources=(self.fact_sources[name].source, event.source),
                )
            self.facts[name] = value
            previous_source = self.fact_sources.get(name)
            if previous_source is None or event.source.identity < previous_source.source.identity:
                self.fact_sources[name] = event

        self.sources.append(event.source)


@dataclass
class Entity:
    """One protocol or Graph identity with independently retained site observations."""

    identity: EntityIdentity
    observations: dict[ObservationIdentity, Observation] = field(default_factory=dict)
    facts: dict[str, int] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    fact_sources: dict[str, Event] = field(default_factory=dict)

    def add(self, event: Event) -> None:
        """Merge cross-site stable facts; report conflicts against local evidence first."""
        self.sources.append(event.source)
        match self.identity.kind:
            case "invocation":
                stable = ("layer_ordinal", "payload_rows", "output_requirement")
            case "operation":
                stable = (
                    "generation",
                    "instance_rank",
                    "layer_ordinal",
                    "payload_rows",
                    "forward_mode",
                    "output_requirement",
                    "dp_row_layout",
                )
            case _:
                stable = ()

        facts = event.record.present_facts()
        for name in stable:
            if name not in facts:
                continue
            value = facts[name]
            if name in self.facts and self.facts[name] != value:
                local = self.observations.get(
                    ObservationIdentity(self.identity, event.source.identity.producer, EventRole(event.record.role))
                )
                previous = (
                    local.fact_sources.get(name, self.fact_sources[name])
                    if local is not None
                    else self.fact_sources[name]
                )
                raise event.error(
                    ErrorCode.IDENTITY_CONFLICT, f"conflicting fact {name}", sources=(previous.source, event.source)
                )
            self.facts[name] = value
            self.fact_sources.setdefault(name, event)

    def observe(self, event: Event, findings: list[Finding], *, boundary: bool = True) -> None:
        """Merge local evidence, keeping auxiliary facts out of the boundary map."""
        identity = ObservationIdentity(self.identity, event.source.identity.producer, EventRole(event.record.role))
        observation = self.observations.setdefault(identity, Observation(identity))
        observation.add(event, findings, boundary=boundary)


@dataclass(frozen=True)
class CausalEdge:
    """An explicitly evidenced relation; source events explain why it exists."""

    relation: CausalRelation
    origin: EntityIdentity | EventIdentity
    destination: EntityIdentity | EventIdentity
    sources: tuple[Source, ...]


@dataclass
class EntityGraph:
    """Mutable reconstruction inventory; edge keys retain all distinct source locations."""

    entities: dict[EntityIdentity, Entity] = field(default_factory=dict)
    edges: dict[tuple[CausalRelation, EntityIdentity | EventIdentity, EntityIdentity | EventIdentity], CausalEdge] = (
        field(default_factory=dict)
    )

    def link(
        self,
        relation: CausalRelation,
        origin: EntityIdentity | EventIdentity,
        destination: EntityIdentity | EventIdentity,
        sources: tuple[Source, ...],
    ) -> None:
        """Merge one identity edge while retaining source encounter order.

        Repeated links append sources absent from the retained edge. Association
        cardinality is checked by validate_associations() after all edges exist.
        """
        key = (relation, origin, destination)
        previous = self.edges.get(key)
        combined = (
            previous.sources + tuple(source for source in sources if source not in previous.sources)
            if previous
            else sources
        )
        self.edges[key] = CausalEdge(relation, origin, destination, combined)

    def validate_associations(self) -> None:
        """Reject remapped operations, executions and reused association destinations."""
        # Repeated association evidence may add sources while keeping one target.
        targets: dict[tuple[CausalRelation, EntityIdentity | EventIdentity], EntityIdentity | EventIdentity] = {}

        for edge in self.edges.values():
            # An Invocation can have several Routing publishers and Graph activities.
            if edge.relation in ("operation_to_invocation", "execution_of_template", "invocation_lease"):
                relation_key = (edge.relation, edge.origin)
                if relation_key in targets and targets[relation_key] != edge.destination:
                    source = edge.sources[0]
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT,
                        f"conflicting {edge.relation}",
                        path=source.path,
                        offset=source.offset,
                        producer=source.identity.producer,
                        event_sequence=source.identity.event_sequence,
                    )
                targets[relation_key] = edge.destination

        inverse_targets: dict[
            tuple[CausalRelation, EntityIdentity | EventIdentity], EntityIdentity | EventIdentity
        ] = {}
        for edge in self.edges.values():
            if edge.relation in ("graph_to_operation", "invocation_lease", "invocation_attachment"):
                relation_key = (edge.relation, edge.destination)
                if relation_key in inverse_targets and inverse_targets[relation_key] != edge.origin:
                    source = edge.sources[0]
                    raise ObservationError(
                        ErrorCode.IDENTITY_CONFLICT,
                        f"reused {edge.relation} destination",
                        path=source.path,
                        offset=source.offset,
                        producer=source.identity.producer,
                        event_sequence=source.identity.event_sequence,
                    )
                inverse_targets[relation_key] = edge.origin
