"""Strict version-four artifact declarations and provenance values."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NoReturn, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.provenance import Source
from xpool.observability.wire import PresenceBit

__all__ = [
    "CalibrationSample",
    "Counters",
    "Event",
    "EventRecord",
    "Finding",
    "Metadata",
    "PeMember",
    "QualityCheckpoint",
    "RoutingAttachment",
]

presence_groups = {
    PresenceBit.GENERATION: ("generation",),
    PresenceBit.ENDPOINT: ("endpoint_owner", "endpoint_sequence"),
    PresenceBit.OPERATION: ("operation_sequence",),
    PresenceBit.GRAPH_TEMPLATE: ("graph_owner", "graph_template"),
    PresenceBit.GRAPH_EXECUTION: ("graph_owner", "graph_execution"),
    PresenceBit.INVOCATION: ("instance_index", "invocation_sequence"),
    PresenceBit.LEASE: ("lane_index", "lease_sequence"),
    PresenceBit.ATTACHMENT: ("attachment_owner", "attachment_sequence"),
    PresenceBit.LOCAL_TRACE_ID: ("local_trace_id",),
    PresenceBit.REQUEST_FACTS: ("layer_ordinal", "payload_rows", "forward_mode", "output_requirement", "dp_row_layout"),
    PresenceBit.INVOCATION_FACTS: ("layer_ordinal", "payload_rows", "output_requirement"),
    PresenceBit.DP_RANK_PAYLOAD_ROWS: ("dp_rank_payload_rows",),
    PresenceBit.PAYLOAD_ROW_CAPACITY: ("payload_row_capacity",),
    PresenceBit.READY_TICKET: ("ready_ticket",),
    PresenceBit.DELIVERY: ("delivery",),
    PresenceBit.RESULT_CODE: ("result_code",),
    PresenceBit.INSTANCE_RANK: ("instance_rank",),
    PresenceBit.PE: ("pe",),
}
type uint64 = Annotated[int, Field(ge=0, le=2**64 - 1)]
type positive_uint64 = Annotated[int, Field(gt=0, le=2**64 - 1)]
type uint32 = Annotated[int, Field(ge=0, le=2**32 - 1)]
type sint32 = Annotated[int, Field(ge=-(2**31), lt=2**31)]
type ProcessRole = Literal["instance", "atnagent", "ffnagent"]
type Capability = Literal["enabled", "disabled", "not_applicable", "unavailable"]
type AttachmentStatus = Literal[
    "published", "not_collected", "capacity_exhausted", "payload_oversize", "publish_failed", "missing"
]


class StrictModel(BaseModel):
    """An owned input rejects coercion and undeclared fields."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    @classmethod
    def parse_artifact(
        cls,
        data: bytes,
        path: Path,
        code: ErrorCode,
        *,
        publication_identity: tuple[str, int, int] | None = None,
    ) -> Self:
        """Decode exact JSON without duplicate keys, coercion or non-finite values.

        A supplied Run/Producer/revision identity is checked before schema and
        membership semantics, after lexical and contract-version validation.
        """

        def unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
            result: dict[str, JsonValue] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field {key}")
                result[key] = value
            return result

        def invalid_constant(value: str) -> JsonValue:
            raise ValueError(f"invalid JSON constant {value}")

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("JSON float exceeds finite representation")
            return parsed

        try:
            # JSON is a dynamic boundary; Pydantic establishes the concrete
            # schema immediately after duplicate and lexical validation.
            value = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=unique_fields,
                parse_constant=invalid_constant,
                parse_float=finite_float,
            )

            if isinstance(value, dict) and type(value.get("contract_version")) is not int:
                raise ValueError("contract_version must be an integer")

            if (
                isinstance(value, dict)
                and type(value.get("contract_version")) is int
                and value["contract_version"] != 4
            ):
                raise ObservationError(ErrorCode.UNSUPPORTED_VERSION, "unsupported artifact version", path=path)

            if publication_identity is not None and (
                not isinstance(value, dict)
                or (value.get("run_id"), value.get("producer_id"), value.get("revision")) != publication_identity
            ):
                raise ObservationError(
                    code, "metadata publication identity mismatch", path=path, producer=publication_identity[1]
                )

            return cls.model_validate(value)
        except ValidationError as exc:
            raise cls.artifact_error(exc, path, code, value) from exc
        except ObservationError:
            raise
        except (ValueError, UnicodeDecodeError) as exc:
            raise ObservationError(code, "invalid JSON artifact", path=path) from exc

    @classmethod
    def artifact_error(cls, exc: ValidationError, path: Path, code: ErrorCode, value: JsonValue) -> ObservationError:
        """Translate schema errors at the published artifact boundary."""
        return ObservationError(code, "invalid JSON artifact", path=path)


class Process(StrictModel):
    creation_id: str
    pid: positive_uint64
    runtime_role: ProcessRole

    @model_validator(mode="after")
    def validate_uuid(self) -> Self:
        if str(UUID(self.creation_id)) != self.creation_id:
            raise ValueError("process creation UUID must be canonical")
        return self


class Producer(StrictModel):
    id: positive_uint64
    source: Literal["host", "device"]
    process: Process
    process_creation_id: str
    clock_domain_id: positive_uint64

    @model_validator(mode="after")
    def validate_process(self) -> Self:
        if self.process_creation_id != self.process.creation_id:
            raise ValueError("Producer process creation identity mismatch")
        return self


class ClockDomain(StrictModel):
    id: positive_uint64
    kind: Literal["host_monotonic", "device_globaltimer"]
    unit: Literal["ns"]
    host_creation_id: str | None = None
    process_creation_id: str | None = None
    physical_device_uuid: str | None = None
    logical_device: uint32 | None = None
    raw_epoch: positive_uint64 | None = None

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if self.kind == "host_monotonic":
            if self.host_creation_id is None or any(
                value is not None
                for value in (self.process_creation_id, self.physical_device_uuid, self.logical_device, self.raw_epoch)
            ):
                raise ValueError("invalid Host domain identity")
            UUID(self.host_creation_id)
        elif (
            any(
                value is None
                for value in (self.process_creation_id, self.physical_device_uuid, self.logical_device, self.raw_epoch)
            )
            or self.host_creation_id is not None
        ):
            raise ValueError("invalid Device domain identity")
        return self


class ModelTopology(StrictModel):
    atn_tp_size: positive_uint64
    atn_dp_size: positive_uint64
    ffn_tp_size: positive_uint64 | None = None


class GenerationModel(StrictModel):
    instance_index: uint64
    model_id: str


class PeMember(StrictModel):
    """One Generation's paired Host/Device recording sources at a Fabric PE."""

    pe: Annotated[int, Field(ge=0, le=2**31 - 1)]
    host_producer: positive_uint64
    device_producer: positive_uint64


class Generation(StrictModel):
    id: positive_uint64
    fabric_generation: str
    atnagent_count: positive_uint64
    ffnagent_count: positive_uint64
    model_topologies: list[ModelTopology]
    scheduler: Literal["fifo", "random"]
    models: list[GenerationModel]
    pe_members: list[PeMember]

    def pe_for(self, producer: int) -> int | None:
        """Resolve a recording source within this Generation; nonmembers return None."""
        return next(
            (member.pe for member in self.pe_members if producer in (member.host_producer, member.device_producer)),
            None,
        )

    def validate_membership(self, producers: dict[int, Producer], domains: dict[int, ClockDomain], index: int) -> None:
        """Validate ordered coverage, then process and clock ownership.

        Raises:
            ObservationError: For malformed membership or duplicate bindings,
                including the metadata field path and available identities.
        """
        field_path = f"generations[{index}].pe_members"

        def fail(
            detail: str,
            *,
            code: ErrorCode = ErrorCode.INVALID_EVENT,
            member: PeMember | None = None,
            producer: int | None = None,
            suffix: str = "",
        ) -> NoReturn:
            raise ObservationError(
                code,
                detail,
                metadata_field_path=field_path + suffix,
                generation=self.id,
                producer=producer,
                pe=member.pe if member is not None else None,
            )

        seen_pes: set[int] = set()
        seen_producers: set[int] = set()
        for member_index, member in enumerate(self.pe_members):
            if member.pe in seen_pes:
                fail(
                    "duplicate PE binding",
                    code=ErrorCode.IDENTITY_CONFLICT,
                    member=member,
                    suffix=f"[{member_index}].pe",
                )
            seen_pes.add(member.pe)
            for producer_id in (member.host_producer, member.device_producer):
                if producer_id in seen_producers:
                    fail(
                        "duplicate Producer binding",
                        code=ErrorCode.IDENTITY_CONFLICT,
                        member=member,
                        producer=producer_id,
                        suffix=f"[{member_index}]",
                    )
                seen_producers.add(producer_id)
        if len(self.pe_members) != self.atnagent_count + self.ffnagent_count or any(
            member.pe != member_index for member_index, member in enumerate(self.pe_members)
        ):
            fail("membership must cover every PE in ascending order")
        for member_index, member in enumerate(self.pe_members):
            host = producers.get(member.host_producer)
            device = producers.get(member.device_producer)
            suffix = f"[{member_index}]"
            if host is None or host.source != "host":
                fail(
                    "member requires a registered Host source",
                    member=member,
                    producer=member.host_producer,
                    suffix=suffix + ".host_producer",
                )
            if device is None or device.source != "device":
                fail(
                    "member requires a registered Device source",
                    member=member,
                    producer=member.device_producer,
                    suffix=suffix + ".device_producer",
                )
            expected_role = "atnagent" if member.pe < self.atnagent_count else "ffnagent"
            domain = domains.get(device.clock_domain_id)
            host_domain = domains.get(host.clock_domain_id)
            if (
                host.process != device.process
                or device.process.runtime_role != expected_role
                or domain is None
                or domain.kind != "device_globaltimer"
                or domain.process_creation_id != device.process.creation_id
                or host_domain is None
                or host_domain.kind != "host_monotonic"
            ):
                fail(
                    "member process, role or clock ownership mismatch",
                    member=member,
                    producer=device.id,
                    suffix=suffix,
                )


class EndpointSite(StrictModel):
    producer: positive_uint64
    site: Literal["instance", "atnagent"]


class Endpoint(StrictModel):
    owner_producer: positive_uint64
    endpoint_sequence: positive_uint64
    generation: positive_uint64
    instance_index: uint64
    instance_rank: uint32
    model_id: str
    arena_handle_suffix: str
    sites: list[EndpointSite]
    supersedes_endpoint_sequence: positive_uint64 | None = None


class GraphTemplate(StrictModel):
    owner_producer: positive_uint64
    template_sequence: positive_uint64
    kind: str
    backend_class: str | None = None
    signature_index: uint64 | None = None
    lane_index: uint64 | None = None
    compute_templates: list[positive_uint64] = Field(default_factory=list)


class Artifact(StrictModel):
    id: positive_uint64
    kind: str
    status: Literal["published", "unavailable", "missing", "publish_failed"]
    path: str | None
    sha256: str | None


class Capabilities(StrictModel):
    transport: Capability
    fabric: Capability
    graph: Capability
    routing: Capability
    calibration: Capability


class Metadata(StrictModel):
    """One immutable, hash-addressed Producer metadata revision."""

    contract_version: Literal[4]
    run_id: str
    producer_id: positive_uint64
    revision: positive_uint64
    process: Process
    clock_domain_id: positive_uint64
    producers: list[Producer]
    generations: list[Generation]
    clock_domains: list[ClockDomain]
    endpoints: list[Endpoint]
    graph_templates: list[GraphTemplate]
    artifacts: list[Artifact]
    capabilities: Capabilities
    # Configuration is owned by xpool.config, not by the offline reader.
    configuration: dict[str, JsonValue]

    @classmethod
    def artifact_error(cls, exc: ValidationError, path: Path, code: ErrorCode, value: JsonValue) -> ObservationError:
        """Keep membership failures distinct from unavailable metadata."""
        for error in exc.errors():
            location = error["loc"]
            context = error.get("ctx", {})
            original = context.get("error")
            if isinstance(original, ObservationError):
                return ObservationError(
                    original.code,
                    original.detail,
                    path=path,
                    metadata_field_path=original.metadata_field_path,
                    generation=original.generation,
                    producer=original.producer,
                    pe=original.pe,
                )
            if "pe_members" in location or "runtime_role" in location:
                field_path = "".join(
                    f"[{part}]" if isinstance(part, int) else ("." if index else "") + part
                    for index, part in enumerate(location)
                )
                generation = producer = pe = None
                # Pydantic locations address the original JSON; retain identities
                # even when a nested member cannot be constructed.
                if isinstance(value, dict) and location and isinstance(location[0], str):
                    entries = value.get(location[0])
                    if isinstance(entries, list) and len(location) > 1 and isinstance(location[1], int):
                        entry = entries[location[1]]
                        if isinstance(entry, dict):
                            if location[0] == "generations":
                                generation = entry.get("id")
                                members = entry.get("pe_members")
                                if isinstance(members, list) and len(location) > 3 and isinstance(location[3], int):
                                    member = members[location[3]]
                                    if isinstance(member, dict):
                                        pe = member.get("pe")
                                        member_field = location[-1]
                                        if isinstance(member_field, str) and member_field != "pe":
                                            producer = member.get(member_field)
                            elif location[0] == "producers":
                                producer = entry.get("id")
                return ObservationError(
                    ErrorCode.INVALID_EVENT,
                    error["msg"],
                    path=path,
                    metadata_field_path=field_path,
                    generation=generation if type(generation) is int else None,
                    producer=producer if type(producer) is int else None,
                    pe=pe if type(pe) is int else None,
                )
        return super().artifact_error(exc, path, code, value)

    def pe_for(self, generation: int | None, producer: int) -> int | None:
        """Resolve this Run's recording source, without inferring absent Generation evidence.

        Raises:
            ObservationError: If an explicit Generation reference is unresolved.
        """
        if generation is None:
            return None
        entry = next((item for item in self.generations if item.id == generation), None)
        if entry is None:
            raise ObservationError(
                ErrorCode.INVALID_EVENT,
                "unregistered Generation",
                generation=generation,
                producer=producer,
                metadata_field_path="generations",
            )
        return entry.pe_for(producer)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """Establish reference uniqueness and ownership before admitting events."""
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("Run UUID must be canonical")

        producers = {item.id: item for item in self.producers}
        domains = {item.id: item for item in self.clock_domains}
        generations = {item.id: item for item in self.generations}

        for entries, keys in (
            (self.producers, [item.id for item in self.producers]),
            (self.clock_domains, [item.id for item in self.clock_domains]),
            (self.generations, [item.id for item in self.generations]),
            (self.endpoints, [(item.owner_producer, item.endpoint_sequence) for item in self.endpoints]),
            (self.graph_templates, [(item.owner_producer, item.template_sequence) for item in self.graph_templates]),
            (self.artifacts, [item.id for item in self.artifacts]),
        ):
            if len(entries) != len(set(keys)):
                raise ValueError("duplicate metadata identity")

        own = producers.get(self.producer_id)

        if own is None or own.process != self.process or own.clock_domain_id != self.clock_domain_id:
            raise ValueError("metadata owner mismatch")

        for index, generation in enumerate(self.generations):
            generation.validate_membership(producers, domains, index)

        for producer in self.producers:
            domain = domains.get(producer.clock_domain_id)
            if domain is None or (producer.source == "host") != (domain.kind == "host_monotonic"):
                raise ValueError("Producer clock domain mismatch")
            if producer.source == "device" and domain.process_creation_id != producer.process_creation_id:
                raise ValueError("Device domain process mismatch")

        for endpoint in self.endpoints:
            owner = producers.get(endpoint.owner_producer)
            if owner is None or owner.source != "device" or owner.process.runtime_role != "atnagent":
                raise ValueError("Endpoint owner is not an AtnAgent Device Producer")
            if endpoint.generation not in generations:
                raise ValueError("Endpoint Generation unavailable")
            for site in endpoint.sites:
                if site.producer not in producers or producers[site.producer].process.runtime_role != site.site:
                    raise ValueError("Endpoint site mismatch")

        templates = {(item.owner_producer, item.template_sequence) for item in self.graph_templates}
        for template in self.graph_templates:
            if template.owner_producer not in producers:
                raise ValueError("Graph owner unavailable")
            if any((template.owner_producer, child) not in templates for child in template.compute_templates):
                raise ValueError("Compute template unavailable")

        return self


class EventRecord(StrictModel):
    """Complete canonical wire record; presence controls semantic availability."""

    event_sequence: positive_uint64
    raw_time_ns: uint64
    clock_domain_id: positive_uint64
    kind: uint32
    presence: uint32
    generation: uint64
    endpoint_owner: uint64
    endpoint_sequence: uint64
    operation_sequence: uint64
    graph_owner: uint64
    graph_template: uint64
    graph_execution: uint64
    instance_index: uint64
    invocation_sequence: uint64
    lane_index: uint64
    lease_sequence: uint64
    attachment_owner: uint64
    attachment_sequence: uint64
    local_trace_id: uint64
    layer_ordinal: uint64
    payload_rows: uint64
    dp_rank_payload_rows: uint64
    payload_row_capacity: uint64
    ready_ticket: uint64
    forward_mode: uint32
    output_requirement: uint32
    dp_row_layout: uint32
    delivery: uint32
    result_code: uint32
    role: uint32
    instance_rank: uint32
    pe: sint32
    data0: uint64
    data1: uint64
    data2: uint64
    data3: uint64
    data4: uint64

    def has(self, bit: PresenceBit) -> bool:
        """Return whether a named contract presence group was recorded."""
        return bool(self.presence & (1 << bit))

    def present_facts(self) -> dict[str, int]:
        """Return fields whose semantic presence is established by recorded group bits."""
        return {name: getattr(self, name) for bit, names in presence_groups.items() if self.has(bit) for name in names}


class Counters(StrictModel):
    attempted: uint64
    committed: uint64
    dropped_full: uint64
    abandoned: uint64
    stale_token: uint64
    duplicate_commit: uint64
    copied: uint64
    exported: uint64
    post_commit_lost: uint64
    snapshot_operations_attempted: uint64
    snapshot_operations_dropped: uint64
    attachment_attempted: uint64
    attachment_capacity_exhausted: uint64
    attachment_payload_oversize: uint64
    attachment_not_collected: uint64
    attachment_publish_failed: uint64
    sample_attempted: uint64
    sample_failed: uint64
    sample_skipped_busy: uint64
    context_unavailable: uint64


class QualityCheckpoint(StrictModel):
    """Cumulative source counters with explicit finality and unknown-tail state."""

    contract_version: Literal[4]
    run_id: str
    producer_id: positive_uint64
    revision: positive_uint64
    host_time_ns: uint64
    collection_state: Literal["active", "stopped", "closed"]
    export_state: Literal["active", "failed", "closed"]
    stop_reason: (
        Literal["disk_quota", "io_error", "metadata_unavailable", "identity_exhausted", "shutdown_deadline"] | None
    )
    last_attempted_sequence: uint64
    counters: Counters
    pending: uint64 | None
    tail: Literal["complete", "pending", "unknown"]
    counter_exhausted: bool
    exact_totals: bool

    @property
    def has_final_totals(self) -> bool:
        """Whether an exact closing checkpoint proves final Producer counters."""
        return (
            self.tail == "complete"
            and self.exact_totals
            and self.collection_state in ("closed", "stopped")
            and self.export_state in ("closed", "failed")
        )

    @model_validator(mode="after")
    def validate_conservation(self) -> Self:
        """Reject contradictory exact accounting while preserving lower bounds."""
        if str(UUID(self.run_id)) != self.run_id:
            raise ValueError("Run UUID must be canonical")

        c = self.counters

        if self.exact_totals:
            if self.counter_exhausted:
                raise ValueError("saturated counters cannot be exact")
            if self.pending is not None and c.attempted != c.committed + c.dropped_full + c.abandoned + self.pending:
                raise ValueError("attempt conservation violated")
            if not c.exported <= c.copied <= c.committed or c.exported + c.post_commit_lost > c.committed:
                raise ValueError("export conservation violated")

        return self


class RoutingAttachment(StrictModel):
    """Final Routing index entry keyed by publication identity rather than slot."""

    owner: positive_uint64
    sequence: positive_uint64
    status: AttachmentStatus
    path: str | None = None
    sha256: str | None = None
    tensor_keys: list[str] = Field(default_factory=list)
    generation: positive_uint64
    instance_index: uint64
    invocation_sequence: positive_uint64
    lane_index: uint64
    lease_sequence: positive_uint64
    layer_ordinal: uint64
    publisher: positive_uint64
    pe: uint32
    clock_domain_id: positive_uint64
    raw_publication_time_ns: uint64
    shape: list[uint64]
    dtypes: list[str]

    @model_validator(mode="after")
    def validate_publication(self) -> Self:
        """Require usable paths and hashes exclusively for published attachments."""
        if self.status == "published":
            if self.path is None or self.sha256 is None or len(self.sha256) != 64:
                raise ValueError("published attachment requires path and SHA-256")
        elif self.path is not None or self.sha256 is not None:
            raise ValueError("unpublished attachment cannot have a usable path/hash")
        return self


@dataclass(frozen=True)
class Event:
    """Canonical record paired with its exact metadata and physical source."""

    record: EventRecord
    source: Source
    metadata: Metadata

    def error(self, code: ErrorCode, detail: str, *, sources: tuple[Source, ...] = ()) -> ObservationError:
        """Attach this Event's source identity and any conflicting evidence to an error."""
        source = self.source
        return ObservationError(
            code,
            detail,
            path=source.path,
            offset=source.offset,
            producer=source.identity.producer,
            event_sequence=self.record.event_sequence,
            sources=sources or (source,),
            generation=self.record.generation if self.record.has(PresenceBit.GENERATION) else None,
            pe=self.record.pe if self.record.has(PresenceBit.PE) else None,
        )


@dataclass(frozen=True)
class Finding:
    """A local evidence limitation with optional event or Producer provenance."""

    code: str
    detail: str
    sources: tuple[Source, ...] = ()
    run_id: str | None = None
    producer: int | None = None


@dataclass(frozen=True)
class CalibrationSample:
    """A raw paired attempt for clock calibration; failed Device reads remain absent."""

    source: Source
    attempt_sequence: int
    device_clock_domain_id: int
    raw_epoch: int
    host_before_ns: int
    device_time_ns: int | None
    host_after_ns: int | None
    outcome: Literal["succeeded", "failed"]
    failure_code: Literal["enqueue_failed", "completion_failed", "timed_out", "shutdown"] | None
