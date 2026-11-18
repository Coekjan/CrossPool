"""Independent fixed-width little-endian observation artifact codec."""

import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import Event, EventRecord, Metadata, presence_groups
from xpool.observability.provenance import EventIdentity, Source
from xpool.observability.wire import (
    CHUNK_FLAGS,
    CHUNK_HEADER_BYTES,
    CHUNK_MAGIC,
    CONTRACT_VERSION,
    CRC32_BYTES,
    EVENT_BYTES,
    MAX_EVENTS_PER_CHUNK,
    METADATA_HASH_PREFIX_BYTES,
    MIN_EVENTS_PER_CHUNK,
    AttachmentState,
    CalibrationEventKind,
    ChunkHeaderOffset,
    Delivery,
    DpRowLayout,
    EventRole,
    FabricEventKind,
    ForwardMode,
    GraphError,
    GraphEventKind,
    GraphMode,
    GraphPhase,
    InvalidRequestField,
    OutputRequirement,
    PresenceBit,
    ResultCode,
    RoutingEventKind,
    SampleFailure,
    TransportEventKind,
    device_execution_boundary,
    known_presence_mask,
    parse_event_kind,
)

__all__ = ["Chunk", "ChunkHeader", "encode_chunk", "parse_chunk"]

event_struct = struct.Struct("<3Q2I19Q7Ii5Q")
header_struct = struct.Struct("<8s4H9Q2I16s16s")
event_fields = tuple(EventRecord.model_fields)
kind_names: dict[int, str] = {
    TransportEventKind.REQUEST_STAGING_STARTED: "RequestStagingStarted",
    TransportEventKind.REQUEST_STAGING_COMPLETED: "RequestStagingCompleted",
    TransportEventKind.REQUEST_PUBLISHED: "RequestPublished",
    TransportEventKind.REQUEST_OBSERVED: "RequestObserved",
    TransportEventKind.EXECUTION_STARTED: "ExecutionStarted",
    TransportEventKind.EXECUTION_COMPLETED: "ExecutionCompleted",
    TransportEventKind.RESULT_PUBLISHED: "ResultPublished",
    TransportEventKind.RESULT_OBSERVED: "ResultObserved",
    TransportEventKind.OUTPUT_COPIED: "OutputCopied",
    TransportEventKind.RESULT_ACKNOWLEDGED: "ResultAcknowledged",
    TransportEventKind.CLOSED: "Closed",
    TransportEventKind.TRANSPORT_ANOMALY: "TransportAnomaly",
    FabricEventKind.SUBMISSION_PREPARED: "SubmissionPrepared",
    FabricEventKind.SUBMISSION_PUBLISHED: "SubmissionPublished",
    FabricEventKind.ADMISSION_OBSERVED: "AdmissionObserved",
    FabricEventKind.INPUT_READY_PUBLISHED: "InputReadyPublished",
    FabricEventKind.OUTPUT_COMMIT_OBSERVED: "OutputCommitObserved",
    FabricEventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED: "OutputAcknowledgementPublished",
    FabricEventKind.ENQUEUED: "Enqueued",
    FabricEventKind.SCHEDULED: "Scheduled",
    FabricEventKind.ADMISSION_PUBLISHED: "AdmissionPublished",
    FabricEventKind.LANE_EXECUTION_PUBLISHED: "LaneExecutionPublished",
    FabricEventKind.FFNAGENT_COMPLETIONS_OBSERVED: "FfnAgentCompletionsObserved",
    FabricEventKind.OUTPUT_COMMIT_PUBLISHED: "OutputCommitPublished",
    FabricEventKind.OUTPUT_ACKNOWLEDGEMENTS_OBSERVED: "OutputAcknowledgementsObserved",
    FabricEventKind.LANE_RELEASED: "LaneReleased",
    FabricEventKind.LANE_EXECUTION_OBSERVED: "LaneExecutionObserved",
    FabricEventKind.INPUT_READY_OBSERVED: "InputReadyObserved",
    FabricEventKind.ROUTING_METADATA_PUBLISHED: "RoutingMetadataPublished",
    FabricEventKind.ROUTING_METADATA_OBSERVED: "RoutingMetadataObserved",
    FabricEventKind.COMPUTE_STARTED: "ComputeStarted",
    FabricEventKind.COMPUTE_COMPLETED: "ComputeCompleted",
    FabricEventKind.PARTIAL_READY_PUBLISHED: "PartialReadyPublished",
    FabricEventKind.PEER_PARTIALS_READY_OBSERVED: "PeerPartialsReadyObserved",
    FabricEventKind.COMPLETION_PUBLISHED: "CompletionPublished",
    GraphEventKind.GRAPH_CAPTURE_BEGIN: "GraphCaptureBegin",
    GraphEventKind.GRAPH_CAPTURE_END: "GraphCaptureEnd",
    GraphEventKind.GRAPH_CAPTURE_ERROR: "GraphCaptureError",
    GraphEventKind.GRAPH_TEMPLATE_INSTALLED: "GraphTemplateInstalled",
    GraphEventKind.GRAPH_HOST_CALL_BEGIN: "GraphHostCallBegin",
    GraphEventKind.GRAPH_HOST_CALL_END: "GraphHostCallEnd",
    GraphEventKind.GRAPH_HOST_CALL_ERROR: "GraphHostCallError",
    GraphEventKind.GRAPH_DEVICE_ACTIVITY_BEGIN: "GraphDeviceActivityBegin",
    GraphEventKind.GRAPH_DEVICE_ACTIVITY_END: "GraphDeviceActivityEnd",
    GraphEventKind.GRAPH_TEMPLATE_RETIRED: "GraphTemplateRetired",
    GraphEventKind.GRAPH_INSTALL_BEGIN: "GraphInstallBegin",
    GraphEventKind.GRAPH_INSTALL_END: "GraphInstallEnd",
    GraphEventKind.GRAPH_INSTALL_ERROR: "GraphInstallError",
    FabricEventKind.TRANSPORT_FABRIC_ASSOCIATED: "TransportFabricAssociated",
    RoutingEventKind.ROUTING_PUBLISHED: "RoutingPublished",
    CalibrationEventKind.CALIBRATION_SAMPLE_SUCCEEDED: "CalibrationSampleSucceeded",
    CalibrationEventKind.CALIBRATION_SAMPLE_FAILED: "CalibrationSampleFailed",
}


@dataclass(frozen=True)
class ChunkHeader:
    """Publication identity and collection window, independent of event time."""

    run_id: str
    producer: int
    chunk_sequence: int
    metadata_revision: int
    host_window_begin_ns: int
    host_window_end_ns: int


@dataclass(frozen=True)
class Chunk:
    """A verified complete publication and its exact metadata revision."""

    header: ChunkHeader
    metadata: Metadata
    events: tuple[Event, ...]


def validate_event(event: Event) -> None:
    """Validate semantic groups, role requirements and revision-local references."""
    r, m = event.record, event.metadata
    try:
        kind = parse_event_kind(r.kind)
    except ValueError as exc:
        raise event.error(ErrorCode.UNKNOWN_KIND, f"unknown event kind {r.kind}") from exc

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise event.error(ErrorCode.INVALID_EVENT, detail)

    require(not r.presence & ~known_presence_mask, "unknown presence bits")
    require(r.reserved_zero == 0, "reserved bytes must be zero")
    require(
        not r.has(PresenceBit.RAW_REQUEST) or r.kind == TransportEventKind.TRANSPORT_ANOMALY,
        "RawRequest is only valid on TransportAnomaly",
    )
    require(not r.has(PresenceBit.OPERATION) or r.has(PresenceBit.ENDPOINT), "Operation requires Endpoint")
    require(
        not (r.has(PresenceBit.INVOCATION) or r.has(PresenceBit.LEASE)) or r.has(PresenceBit.GENERATION),
        "Invocation/Lease requires Generation",
    )
    require(not r.has(PresenceBit.PE) or r.has(PresenceBit.GENERATION), "PE requires Generation")
    require(
        not r.has(PresenceBit.REQUEST_FACTS) or r.has(PresenceBit.INVOCATION_FACTS),
        "request facts require invocation facts presence",
    )

    # Presence groups share fields; a field is absent only when all its groups are absent.
    present = {field for bit, fields in presence_groups.items() if r.has(bit) for field in fields}

    for field in {field for fields in presence_groups.values() for field in fields} - present:
        require(getattr(r, field) == 0, f"absent field {field} is nonzero")

    for field in (
        "generation",
        "endpoint_owner",
        "endpoint_sequence",
        "operation_sequence",
        "graph_owner",
        "graph_template",
        "graph_execution",
        "invocation_sequence",
        "lease_sequence",
        "attachment_owner",
        "attachment_sequence",
    ):
        if field in present:
            require(getattr(r, field) > 0, f"{field} must be positive")

    require(r.role in EventRole, "unknown role")
    require(not r.has(PresenceBit.PE) or r.pe >= 0, "negative PE")

    if r.has(PresenceBit.REQUEST_FACTS):
        require(r.forward_mode in ForwardMode and r.dp_row_layout in DpRowLayout, "invalid request enum")

    if r.has(PresenceBit.INVOCATION_FACTS):
        require(r.output_requirement in OutputRequirement, "invalid output requirement")

    require(not r.has(PresenceBit.DELIVERY) or r.delivery in Delivery, "invalid delivery")
    require(not r.has(PresenceBit.RESULT_CODE) or r.result_code in ResultCode, "invalid result")

    # Resolve identities against the event's exact metadata revision.
    producers = {item.id: item for item in m.producers}
    producer = producers.get(event.source.identity.producer)
    if producer is None:
        raise event.error(ErrorCode.INVALID_EVENT, "unregistered Producer")

    require(event.source.process_creation_id == producer.process_creation_id, "event process provenance mismatch")

    require(r.clock_domain_id == producer.clock_domain_id, "Producer/domain mismatch")

    roles = {role.name.lower(): role for role in EventRole}
    require(
        r.role == roles[producer.process.runtime_role]
        or (producer.process.runtime_role == "ffnagent" and r.role == EventRole.COORDINATOR)
        or (producer.source == "host" and r.role == EventRole.COLLECTOR),
        "Producer/role mismatch",
    )

    generations = {item.id: item for item in m.generations}
    require(not r.has(PresenceBit.GENERATION) or r.generation in generations, "unregistered Generation")

    if r.has(PresenceBit.ENDPOINT):
        endpoint = next(
            (
                item
                for item in m.endpoints
                if (item.owner_producer, item.endpoint_sequence) == (r.endpoint_owner, r.endpoint_sequence)
            ),
            None,
        )
        require(endpoint is not None, "unregistered Endpoint")

        if endpoint is not None:
            require(
                not r.has(PresenceBit.GENERATION) or endpoint.generation == r.generation, "Endpoint Generation mismatch"
            )
            require(
                not r.has(PresenceBit.INSTANCE_RANK) or endpoint.instance_rank == r.instance_rank,
                "Endpoint rank mismatch",
            )
            require(
                not r.has(PresenceBit.INVOCATION) or endpoint.instance_index == r.instance_index,
                "Endpoint Instance mismatch",
            )

            if isinstance(kind, TransportEventKind) or kind == FabricEventKind.TRANSPORT_FABRIC_ASSOCIATED:
                require(
                    any(site.producer == producer.id and roles[site.site] == r.role for site in endpoint.sites),
                    "event is not from this Endpoint site",
                )

    if r.has(PresenceBit.GRAPH_TEMPLATE):
        require(
            any(
                (item.owner_producer, item.template_sequence) == (r.graph_owner, r.graph_template)
                for item in m.graph_templates
            ),
            "unregistered Graph template",
        )

    require(not r.has(PresenceBit.GRAPH_EXECUTION) or r.graph_owner in producers, "unregistered Graph owner")

    require(not r.has(PresenceBit.ATTACHMENT) or r.attachment_owner in producers, "unregistered attachment owner")

    if r.has(PresenceBit.INVOCATION):
        require(
            any(item.instance_index == r.instance_index for item in generations[r.generation].models),
            "unregistered Instance",
        )

    # Enforce the facts and role owned by each event family.
    required: tuple[PresenceBit, ...] = ()

    if isinstance(kind, TransportEventKind) and kind != TransportEventKind.TRANSPORT_ANOMALY:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.ENDPOINT,
            PresenceBit.OPERATION,
            PresenceBit.REQUEST_FACTS,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.INSTANCE_RANK,
        )
        require(r.role in (EventRole.INSTANCE, EventRole.ATNAGENT), "Transport role mismatch")
    elif r.kind == TransportEventKind.TRANSPORT_ANOMALY:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.ENDPOINT,
            PresenceBit.OPERATION,
            PresenceBit.INSTANCE_RANK,
            PresenceBit.RAW_REQUEST,
        )
        allowed = sum(1 << bit for bit in (*required, PresenceBit.RESULT_CODE))
        require(not r.presence & ~allowed, "unexpected anomaly presence group")
        require(r.role in (EventRole.INSTANCE, EventRole.ATNAGENT), "Transport role mismatch")
    elif isinstance(kind, FabricEventKind) and kind != FabricEventKind.TRANSPORT_FABRIC_ASSOCIATED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.INVOCATION,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.PE,
        )
        expected_role = (
            EventRole.ATNAGENT
            if r.kind < FabricEventKind.ENQUEUED
            else EventRole.COORDINATOR
            if r.kind < FabricEventKind.LANE_EXECUTION_OBSERVED
            else EventRole.FFNAGENT
        )
        require(r.role == expected_role, "Fabric role mismatch")

        if r.role == EventRole.ATNAGENT:
            required += (PresenceBit.REQUEST_FACTS, PresenceBit.DP_RANK_PAYLOAD_ROWS)
            if r.kind >= FabricEventKind.ADMISSION_OBSERVED:
                required += (PresenceBit.LEASE,)
        elif r.role == EventRole.COORDINATOR and r.kind >= FabricEventKind.SCHEDULED:
            required += (PresenceBit.LEASE,)
        elif r.role == EventRole.FFNAGENT:
            required += (PresenceBit.LEASE, PresenceBit.PAYLOAD_ROW_CAPACITY, PresenceBit.DELIVERY)
    elif r.kind == FabricEventKind.TRANSPORT_FABRIC_ASSOCIATED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.ENDPOINT,
            PresenceBit.OPERATION,
            PresenceBit.INVOCATION,
            PresenceBit.REQUEST_FACTS,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.DP_RANK_PAYLOAD_ROWS,
            PresenceBit.INSTANCE_RANK,
            PresenceBit.PE,
        )
        require(r.role == EventRole.ATNAGENT, "association role mismatch")
    elif r.kind == RoutingEventKind.ROUTING_PUBLISHED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.INVOCATION,
            PresenceBit.LEASE,
            PresenceBit.ATTACHMENT,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.PE,
        )
        require(r.role == EventRole.FFNAGENT and r.attachment_owner == producer.id, "Routing publisher mismatch")

    require(all(r.has(bit) for bit in required), "missing required presence group")

    if r.has(PresenceBit.GENERATION):
        generation = generations[r.generation]
        member_pe = m.pe_for(r.generation, producer.id)
        require(not r.has(PresenceBit.PE) or member_pe is not None, "recording source is not a Fabric member")
        if r.role == EventRole.COORDINATOR:
            require(member_pe == generation.atnagent_count, "Coordinator requires the first FfnAgent member")
        if r.has(PresenceBit.PE) and r.pe != member_pe:
            error = event.error(ErrorCode.IDENTITY_CONFLICT, "Event PE conflicts with membership")
            error.metadata_field_path = f"generations[{m.generations.index(generation)}].pe_members"
            raise error

    # Interpret family-specific payload words after common references are validated.
    if r.kind == TransportEventKind.TRANSPORT_ANOMALY:
        raw_forward = r.data1 & 0xFFFFFFFF
        raw_output = r.data1 >> 32
        raw_layout = r.data2 & 0xFFFFFFFF
        invalid_mask = (
            (InvalidRequestField.FORWARD_MODE if raw_forward not in ForwardMode else 0)
            | (InvalidRequestField.OUTPUT_REQUIREMENT if raw_output not in OutputRequirement else 0)
            | (InvalidRequestField.DP_ROW_LAYOUT if raw_layout not in DpRowLayout else 0)
        )
        require(invalid_mask != 0 and r.data2 >> 32 == invalid_mask, "invalid anomaly field mask")
        allowed_boundaries = (
            {
                TransportEventKind.REQUEST_STAGING_STARTED,
                TransportEventKind.REQUEST_STAGING_COMPLETED,
                TransportEventKind.REQUEST_PUBLISHED,
                TransportEventKind.RESULT_OBSERVED,
                TransportEventKind.OUTPUT_COPIED,
                TransportEventKind.RESULT_ACKNOWLEDGED,
                TransportEventKind.CLOSED,
            }
            if r.role == EventRole.INSTANCE
            else {
                TransportEventKind.REQUEST_OBSERVED,
                TransportEventKind.EXECUTION_STARTED,
                TransportEventKind.EXECUTION_COMPLETED,
                TransportEventKind.RESULT_PUBLISHED,
            }
        )
        require(r.data0 in allowed_boundaries, "invalid anomaly site/boundary")
        require(
            r.has(PresenceBit.RESULT_CODE) == (r.data0 >= TransportEventKind.RESULT_PUBLISHED),
            "invalid anomaly result presence",
        )
    elif isinstance(kind, (TransportEventKind, FabricEventKind)):
        require(not any((r.data0, r.data1, r.data2, r.data3, r.data4)), "unexpected protocol payload")
    elif isinstance(kind, GraphEventKind):
        owner = producers.get(r.graph_owner)
        require(
            owner is not None and owner.source == "host" and owner.process_creation_id == producer.process_creation_id,
            "Graph owner/source mismatch",
        )

        require(r.data0 in GraphMode and r.data1 in GraphPhase and r.data4 in GraphError, "invalid Graph payload")

        if r.kind in (
            GraphEventKind.GRAPH_CAPTURE_BEGIN,
            GraphEventKind.GRAPH_CAPTURE_END,
            GraphEventKind.GRAPH_CAPTURE_ERROR,
            GraphEventKind.GRAPH_TEMPLATE_INSTALLED,
            GraphEventKind.GRAPH_TEMPLATE_RETIRED,
            GraphEventKind.GRAPH_INSTALL_BEGIN,
            GraphEventKind.GRAPH_INSTALL_END,
            GraphEventKind.GRAPH_INSTALL_ERROR,
        ):
            require(
                r.has(PresenceBit.GRAPH_TEMPLATE) and not r.has(PresenceBit.GRAPH_EXECUTION),
                "template event requires template, not execution",
            )
        else:
            require(r.has(PresenceBit.GRAPH_EXECUTION), "Graph activity requires execution")
            require(
                r.data0 != GraphMode.EAGER or not r.has(PresenceBit.GRAPH_TEMPLATE), "Eager cannot reference a template"
            )
        if r.kind in (GraphEventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, GraphEventKind.GRAPH_DEVICE_ACTIVITY_END):
            require(
                producer.source == "device" and r.graph_execution > device_execution_boundary,
                "Device execution range/source",
            )
        else:
            require(producer.source == "host" and r.data3 == 0, "Host Graph source/Context mismatch")
            require(
                not r.has(PresenceBit.GRAPH_EXECUTION) or r.graph_execution < device_execution_boundary,
                "Host execution range",
            )
        require(r.data2 == 0 or any(item.id == r.data2 for item in m.artifacts), "unknown Graph artifact")
    elif r.kind == RoutingEventKind.ROUTING_PUBLISHED:
        require(r.data0 in AttachmentState and r.data1 > 0 and r.data3 == r.data4 == 0, "invalid Routing payload")
        if r.data2 != 0:
            artifact = next((item for item in m.artifacts if item.id == r.data2), None)
            require(artifact is not None, "unknown Routing artifact")
            if artifact is not None and (
                artifact.kind != "routing" or artifact.status != "published" or not artifact.path or not artifact.sha256
            ):
                raise event.error(
                    ErrorCode.IDENTITY_CONFLICT, "Routing reference requires a published Routing artifact"
                )
    else:
        require(
            r.presence == 0 and r.role == EventRole.COLLECTOR and producer.source == "host", "invalid sampling source"
        )
        require(
            r.data4 > 0
            and any(
                item.id == r.data3
                and item.kind == "device_globaltimer"
                and item.process_creation_id == producer.process_creation_id
                for item in m.clock_domains
            ),
            "invalid sample identity/domain",
        )
        if r.kind == CalibrationEventKind.CALIBRATION_SAMPLE_SUCCEEDED:
            require(r.data0 <= r.data2 <= r.raw_time_ns, "invalid sampling bracket")
        else:
            require(r.data1 == 0 and r.data2 in SampleFailure, "invalid sample failure")


def parse_chunk(data: bytes, metadata_bytes: bytes | None, *, path: Path = Path("<memory>")) -> Chunk:
    """Decode one complete chunk in the artifact contract's validation order.

    Raises:
        ObservationError: With a structured artifact error code and source location.
    """

    producer: int | None = None

    def fail(code: ErrorCode, detail: str, offset: int = 0) -> ObservationError:
        return ObservationError(code, detail, path=path, offset=int(offset), producer=producer)

    if len(data) < CHUNK_HEADER_BYTES:
        raise fail(ErrorCode.HEADER_TRUNCATED, f"header requires {CHUNK_HEADER_BYTES} bytes")
    (
        magic,
        version,
        header_bytes,
        event_bytes,
        flags,
        producer,
        sequence,
        revision,
        count,
        first,
        last,
        begin,
        end,
        payload_bytes,
        payload_crc,
        header_crc,
        run_bytes,
        metadata_hash,
    ) = header_struct.unpack_from(data)

    if magic != CHUNK_MAGIC:
        raise fail(ErrorCode.BAD_MAGIC, "invalid chunk magic")

    if version != CONTRACT_VERSION:
        raise fail(ErrorCode.UNSUPPORTED_VERSION, f"unsupported chunk version {version}", ChunkHeaderOffset.VERSION)

    if (header_bytes, event_bytes, flags) != (CHUNK_HEADER_BYTES, EVENT_BYTES, CHUNK_FLAGS):
        raise fail(ErrorCode.INVALID_LAYOUT, "invalid header/event layout or flags", ChunkHeaderOffset.HEADER_BYTES)

    if (
        zlib.crc32(
            data[: ChunkHeaderOffset.HEADER_CRC32]
            + bytes(CRC32_BYTES)
            + data[ChunkHeaderOffset.HEADER_CRC32 + CRC32_BYTES : CHUNK_HEADER_BYTES]
        )
        != header_crc
    ):
        raise fail(ErrorCode.HEADER_CHECKSUM, "header CRC mismatch", ChunkHeaderOffset.HEADER_CRC32)

    if (
        not MIN_EVENTS_PER_CHUNK <= count <= MAX_EVENTS_PER_CHUNK
        or payload_bytes != count * EVENT_BYTES
        or len(data) != CHUNK_HEADER_BYTES + payload_bytes
        or not producer
        or not sequence
        or not revision
        or begin > end
    ):
        raise fail(ErrorCode.INVALID_LENGTH, "invalid declared count, size, sequence or window")

    if metadata_bytes is None:
        raise fail(ErrorCode.METADATA_UNAVAILABLE, "required metadata revision unavailable")

    if hashlib.sha256(metadata_bytes).digest()[:METADATA_HASH_PREFIX_BYTES] != metadata_hash:
        raise fail(ErrorCode.METADATA_CHECKSUM, "metadata SHA-256 mismatch", ChunkHeaderOffset.METADATA_HASH)

    run_id = str(UUID(bytes=run_bytes))
    metadata = Metadata.parse_artifact(
        metadata_bytes,
        path,
        ErrorCode.METADATA_UNAVAILABLE,
        publication_identity=(run_id, producer, revision),
    )

    if zlib.crc32(data[CHUNK_HEADER_BYTES:]) != payload_crc:
        raise fail(ErrorCode.PAYLOAD_CHECKSUM, "payload CRC mismatch", ChunkHeaderOffset.PAYLOAD_CRC32)

    events: list[Event] = []

    previous = 0

    for offset in range(CHUNK_HEADER_BYTES, len(data), EVENT_BYTES):
        try:
            record = EventRecord.model_validate(
                dict(zip(event_fields, event_struct.unpack_from(data, offset), strict=True))
            )
        except ValidationError as exc:
            raise fail(ErrorCode.INVALID_EVENT, "invalid record widths or identities", offset) from exc

        source = Source(
            EventIdentity(run_id, producer, record.event_sequence),
            path,
            offset,
            metadata.process.creation_id,
            record.clock_domain_id,
            (record.endpoint_owner, record.endpoint_sequence) if record.has(PresenceBit.ENDPOINT) else None,
            record.instance_rank if record.has(PresenceBit.INSTANCE_RANK) else None,
            record.pe if record.has(PresenceBit.PE) else None,
        )

        event = Event(record, source, metadata)

        validate_event(event)

        if record.event_sequence <= previous:
            raise event.error(ErrorCode.INVALID_EVENT, "non-increasing sequence within chunk")

        previous = record.event_sequence
        events.append(event)

    if events[0].record.event_sequence != first or events[-1].record.event_sequence != last:
        raise fail(ErrorCode.INVALID_EVENT, "header sequence bounds disagree with payload")

    return Chunk(ChunkHeader(run_id, producer, sequence, revision, begin, end), metadata, tuple(events))


def encode_chunk(header: ChunkHeader, records: tuple[EventRecord, ...], metadata_bytes: bytes) -> bytes:
    """Encode canonical fields independently of Native layouts and validate the result.

    Raises:
        ObservationError: If the requested chunk violates the v7 contract.
    """
    try:
        payload = b"".join(event_struct.pack(*(getattr(record, name) for name in event_fields)) for record in records)
        packed = bytearray(
            header_struct.pack(
                CHUNK_MAGIC,
                CONTRACT_VERSION,
                CHUNK_HEADER_BYTES,
                EVENT_BYTES,
                CHUNK_FLAGS,
                header.producer,
                header.chunk_sequence,
                header.metadata_revision,
                len(records),
                records[0].event_sequence if records else 0,
                records[-1].event_sequence if records else 0,
                header.host_window_begin_ns,
                header.host_window_end_ns,
                len(payload),
                zlib.crc32(payload),
                0,
                UUID(header.run_id).bytes,
                hashlib.sha256(metadata_bytes).digest()[:METADATA_HASH_PREFIX_BYTES],
            )
        )
    except (struct.error, ValueError) as exc:
        raise ObservationError(ErrorCode.INVALID_EVENT, "chunk fields exceed wire representation") from exc

    struct.pack_into("<I", packed, ChunkHeaderOffset.HEADER_CRC32, zlib.crc32(packed))
    data = bytes(packed) + payload
    parse_chunk(data, metadata_bytes)
    return data
