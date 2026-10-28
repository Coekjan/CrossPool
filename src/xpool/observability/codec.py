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
    AttachmentState,
    Delivery,
    DpRowLayout,
    EventKind,
    EventRole,
    ForwardMode,
    GraphError,
    GraphMode,
    GraphPhase,
    OutputRequirement,
    PresenceBit,
    ResultCode,
    SampleFailure,
    device_execution_boundary,
    known_presence_mask,
)

__all__ = ["Chunk", "ChunkHeader", "encode_chunk", "parse_chunk"]

event_struct = struct.Struct("<3Q2I19Q7Ii5Q")
header_struct = struct.Struct("<8s4H9Q2I16s16s")
event_fields = tuple(EventRecord.model_fields)
kind_names: dict[int, str] = {
    EventKind.REQUEST_STAGING_STARTED: "RequestStagingStarted",
    EventKind.REQUEST_STAGING_COMPLETED: "RequestStagingCompleted",
    EventKind.REQUEST_PUBLISHED: "RequestPublished",
    EventKind.REQUEST_OBSERVED: "RequestObserved",
    EventKind.EXECUTION_STARTED: "ExecutionStarted",
    EventKind.EXECUTION_COMPLETED: "ExecutionCompleted",
    EventKind.RESULT_PUBLISHED: "ResultPublished",
    EventKind.RESULT_OBSERVED: "ResultObserved",
    EventKind.OUTPUT_COPIED: "OutputCopied",
    EventKind.RESULT_ACKNOWLEDGED: "ResultAcknowledged",
    EventKind.CLOSED: "Closed",
    EventKind.SUBMISSION_PREPARED: "SubmissionPrepared",
    EventKind.SUBMISSION_PUBLISHED: "SubmissionPublished",
    EventKind.ADMISSION_OBSERVED: "AdmissionObserved",
    EventKind.INPUT_READY_PUBLISHED: "InputReadyPublished",
    EventKind.OUTPUT_COMMIT_OBSERVED: "OutputCommitObserved",
    EventKind.OUTPUT_ACKNOWLEDGEMENT_PUBLISHED: "OutputAcknowledgementPublished",
    EventKind.ENQUEUED: "Enqueued",
    EventKind.SCHEDULED: "Scheduled",
    EventKind.ADMISSION_PUBLISHED: "AdmissionPublished",
    EventKind.LANE_EXECUTION_PUBLISHED: "LaneExecutionPublished",
    EventKind.FFNAGENT_COMPLETIONS_OBSERVED: "FfnAgentCompletionsObserved",
    EventKind.OUTPUT_COMMIT_PUBLISHED: "OutputCommitPublished",
    EventKind.OUTPUT_ACKNOWLEDGEMENTS_OBSERVED: "OutputAcknowledgementsObserved",
    EventKind.LANE_RELEASED: "LaneReleased",
    EventKind.LANE_EXECUTION_OBSERVED: "LaneExecutionObserved",
    EventKind.INPUT_READY_OBSERVED: "InputReadyObserved",
    EventKind.ROUTING_METADATA_PUBLISHED: "RoutingMetadataPublished",
    EventKind.ROUTING_METADATA_OBSERVED: "RoutingMetadataObserved",
    EventKind.COMPUTE_STARTED: "ComputeStarted",
    EventKind.COMPUTE_COMPLETED: "ComputeCompleted",
    EventKind.PARTIAL_READY_PUBLISHED: "PartialReadyPublished",
    EventKind.PEER_PARTIALS_READY_OBSERVED: "PeerPartialsReadyObserved",
    EventKind.COMPLETION_PUBLISHED: "CompletionPublished",
    EventKind.GRAPH_CAPTURE_BEGIN: "GraphCaptureBegin",
    EventKind.GRAPH_CAPTURE_END: "GraphCaptureEnd",
    EventKind.GRAPH_CAPTURE_ERROR: "GraphCaptureError",
    EventKind.GRAPH_TEMPLATE_INSTALLED: "GraphTemplateInstalled",
    EventKind.GRAPH_HOST_CALL_BEGIN: "GraphHostCallBegin",
    EventKind.GRAPH_HOST_CALL_END: "GraphHostCallEnd",
    EventKind.GRAPH_HOST_CALL_ERROR: "GraphHostCallError",
    EventKind.GRAPH_DEVICE_ACTIVITY_BEGIN: "GraphDeviceActivityBegin",
    EventKind.GRAPH_DEVICE_ACTIVITY_END: "GraphDeviceActivityEnd",
    EventKind.GRAPH_TEMPLATE_RETIRED: "GraphTemplateRetired",
    EventKind.GRAPH_INSTALL_BEGIN: "GraphInstallBegin",
    EventKind.GRAPH_INSTALL_END: "GraphInstallEnd",
    EventKind.GRAPH_INSTALL_ERROR: "GraphInstallError",
    EventKind.TRANSPORT_FABRIC_ASSOCIATED: "TransportFabricAssociated",
    EventKind.ROUTING_PUBLISHED: "RoutingPublished",
    EventKind.CALIBRATION_SAMPLE_SUCCEEDED: "CalibrationSampleSucceeded",
    EventKind.CALIBRATION_SAMPLE_FAILED: "CalibrationSampleFailed",
}

CHUNK_MAGIC = b"XPOOLC04"


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
    if r.kind not in kind_names:
        raise event.error(ErrorCode.UNKNOWN_KIND, f"unknown event kind {r.kind}")

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise event.error(ErrorCode.INVALID_EVENT, detail)

    require(not r.presence & ~known_presence_mask, "unknown presence bits")
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
        "local_trace_id",
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

            if (
                EventKind.REQUEST_STAGING_STARTED <= r.kind <= EventKind.CLOSED
                or r.kind == EventKind.TRANSPORT_FABRIC_ASSOCIATED
            ):
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

    if EventKind.REQUEST_STAGING_STARTED <= r.kind <= EventKind.CLOSED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.ENDPOINT,
            PresenceBit.OPERATION,
            PresenceBit.LOCAL_TRACE_ID,
            PresenceBit.REQUEST_FACTS,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.INSTANCE_RANK,
        )
        require(r.role in (EventRole.INSTANCE, EventRole.ATNAGENT), "Transport role mismatch")
    elif EventKind.SUBMISSION_PREPARED <= r.kind <= EventKind.COMPLETION_PUBLISHED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.INVOCATION,
            PresenceBit.LOCAL_TRACE_ID,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.PE,
        )
        expected_role = (
            EventRole.ATNAGENT
            if r.kind < EventKind.ENQUEUED
            else EventRole.COORDINATOR
            if r.kind < EventKind.LANE_EXECUTION_OBSERVED
            else EventRole.FFNAGENT
        )
        require(r.role == expected_role, "Fabric role mismatch")

        if r.role == EventRole.ATNAGENT:
            required += (PresenceBit.REQUEST_FACTS, PresenceBit.DP_RANK_PAYLOAD_ROWS)
            if r.kind >= EventKind.ADMISSION_OBSERVED:
                required += (PresenceBit.LEASE,)
        elif r.role == EventRole.COORDINATOR and r.kind >= EventKind.SCHEDULED:
            required += (PresenceBit.LEASE,)
        elif r.role == EventRole.FFNAGENT:
            required += (PresenceBit.LEASE, PresenceBit.PAYLOAD_ROW_CAPACITY, PresenceBit.DELIVERY)
    elif r.kind == EventKind.TRANSPORT_FABRIC_ASSOCIATED:
        required = (
            PresenceBit.GENERATION,
            PresenceBit.ENDPOINT,
            PresenceBit.OPERATION,
            PresenceBit.INVOCATION,
            PresenceBit.LOCAL_TRACE_ID,
            PresenceBit.REQUEST_FACTS,
            PresenceBit.INVOCATION_FACTS,
            PresenceBit.DP_RANK_PAYLOAD_ROWS,
            PresenceBit.INSTANCE_RANK,
            PresenceBit.PE,
        )
        require(r.role == EventRole.ATNAGENT, "association role mismatch")
    elif r.kind == EventKind.ROUTING_PUBLISHED:
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
    if r.kind < EventKind.GRAPH_CAPTURE_BEGIN:
        require(not any((r.data0, r.data1, r.data2, r.data3, r.data4)), "unexpected protocol payload")
    elif EventKind.GRAPH_CAPTURE_BEGIN <= r.kind <= EventKind.GRAPH_INSTALL_ERROR:
        owner = producers.get(r.graph_owner)
        require(
            owner is not None and owner.source == "host" and owner.process_creation_id == producer.process_creation_id,
            "Graph owner/source mismatch",
        )

        require(r.data0 in GraphMode and r.data1 in GraphPhase and r.data4 in GraphError, "invalid Graph payload")

        if r.kind in (
            EventKind.GRAPH_CAPTURE_BEGIN,
            EventKind.GRAPH_CAPTURE_END,
            EventKind.GRAPH_CAPTURE_ERROR,
            EventKind.GRAPH_TEMPLATE_INSTALLED,
            EventKind.GRAPH_TEMPLATE_RETIRED,
            EventKind.GRAPH_INSTALL_BEGIN,
            EventKind.GRAPH_INSTALL_END,
            EventKind.GRAPH_INSTALL_ERROR,
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
        if r.kind in (EventKind.GRAPH_DEVICE_ACTIVITY_BEGIN, EventKind.GRAPH_DEVICE_ACTIVITY_END):
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
    elif r.kind == EventKind.ROUTING_PUBLISHED:
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
        if r.kind == EventKind.CALIBRATION_SAMPLE_SUCCEEDED:
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
        return ObservationError(code, detail, path=path, offset=offset, producer=producer)

    if len(data) < 128:
        raise fail(ErrorCode.HEADER_TRUNCATED, "header requires 128 bytes")
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

    if version != 4:
        raise fail(ErrorCode.UNSUPPORTED_VERSION, f"unsupported chunk version {version}", 8)

    if (header_bytes, event_bytes, flags) != (128, 256, 0):
        raise fail(ErrorCode.INVALID_LAYOUT, "invalid header/event layout or flags", 10)

    if zlib.crc32(data[:92] + bytes(4) + data[96:128]) != header_crc:
        raise fail(ErrorCode.HEADER_CHECKSUM, "header CRC mismatch", 92)

    if (
        not 1 <= count <= 4095
        or payload_bytes != count * 256
        or len(data) != 128 + payload_bytes
        or not producer
        or not sequence
        or not revision
        or begin > end
    ):
        raise fail(ErrorCode.INVALID_LENGTH, "invalid declared count, size, sequence or window")

    if metadata_bytes is None:
        raise fail(ErrorCode.METADATA_UNAVAILABLE, "required metadata revision unavailable")

    if hashlib.sha256(metadata_bytes).digest()[:16] != metadata_hash:
        raise fail(ErrorCode.METADATA_CHECKSUM, "metadata SHA-256 mismatch", 112)

    run_id = str(UUID(bytes=run_bytes))
    metadata = Metadata.parse_artifact(
        metadata_bytes,
        path,
        ErrorCode.METADATA_UNAVAILABLE,
        publication_identity=(run_id, producer, revision),
    )

    if zlib.crc32(data[128:]) != payload_crc:
        raise fail(ErrorCode.PAYLOAD_CHECKSUM, "payload CRC mismatch", 88)

    events: list[Event] = []

    previous = 0

    for offset in range(128, len(data), 256):
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
        ObservationError: If the requested chunk violates the v4 contract.
    """
    try:
        payload = b"".join(event_struct.pack(*(getattr(record, name) for name in event_fields)) for record in records)
        packed = bytearray(
            header_struct.pack(
                CHUNK_MAGIC,
                4,
                128,
                256,
                0,
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
                hashlib.sha256(metadata_bytes).digest()[:16],
            )
        )
    except (struct.error, ValueError) as exc:
        raise ObservationError(ErrorCode.INVALID_EVENT, "chunk fields exceed wire representation") from exc

    struct.pack_into("<I", packed, 92, zlib.crc32(packed))
    data = bytes(packed) + payload
    parse_chunk(data, metadata_bytes)
    return data
