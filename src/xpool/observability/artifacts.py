"""Explicit artifact inventory loading and Routing tensor integrity checks."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import TypeAdapter
from safetensors import SafetensorError, safe_open

from xpool.observability.codec import Chunk, parse_chunk
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.models import (
    AttachmentStatus,
    Event,
    Finding,
    Metadata,
    QualityCheckpoint,
    RoutingAttachment,
    StrictModel,
    uint64,
)
from xpool.observability.provenance import Source
from xpool.observability.wire import (
    AttachmentState,
    EventKind,
)

__all__ = ["ArtifactInputs", "ArtifactSet", "AttachmentEvidence", "AttachmentIndex", "read_artifacts"]

attachment_outcome_names: dict[int, AttachmentStatus] = {
    AttachmentState.NOT_COLLECTED: "not_collected",
    AttachmentState.CAPACITY_EXHAUSTED: "capacity_exhausted",
    AttachmentState.PAYLOAD_OVERSIZE: "payload_oversize",
}


@dataclass(frozen=True)
class AttachmentIndex:
    """Caller-supplied index; completion applies only to this Run/owner."""

    run_id: str
    owner: int
    entries: tuple[RoutingAttachment, ...]
    closing_complete: bool = False


@dataclass(frozen=True)
class ArtifactInputs:
    """Published files and final attachment indices supplied by the caller.

    Paths identify completed publications; readers perform no directory discovery.
    The attachment root confines relative tensor paths. Each index declares its
    own Run/owner completion scope independently of other Producers.
    """

    chunks: tuple[Path, ...]
    metadata: tuple[Path, ...]
    quality: tuple[Path, ...] = ()
    attachment_indices: tuple[AttachmentIndex, ...] = ()
    attachment_root: Path | None = None


@dataclass(frozen=True)
class AttachmentEvidence:
    """Independent tensor and event availability for an exact attachment identity."""

    run_id: str
    owner: int
    sequence: int
    entry: RoutingAttachment | None
    status: AttachmentStatus | Literal["unknown"]
    available: bool
    path: Path | None
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class ArtifactSet:
    """Verified inventory; pass its evidence directly to reconstruct()."""

    chunks: tuple[Chunk, ...]
    metadata: tuple[Metadata, ...]
    checkpoints: tuple[QualityCheckpoint, ...]
    attachments: tuple[AttachmentEvidence, ...]
    findings: tuple[Finding, ...]

    @property
    def events(self) -> tuple[Event, ...]:
        """Flatten publication order without treating it as causal order."""
        return tuple(event for chunk in self.chunks for event in chunk.events)


class TensorRecord(StrictModel):
    owner: uint64
    sequence: uint64
    status: Literal["published"]
    generation: uint64
    instance_index: uint64
    invocation_sequence: uint64
    lane_index: uint64
    lease_sequence: uint64
    layer_ordinal: uint64
    publisher: uint64
    pe: uint64
    clock_domain_id: uint64
    raw_publication_time_ns: uint64
    shape: list[uint64]


def read_bytes(path: Path, code: ErrorCode) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ObservationError(code, "could not read artifact", path=path) from exc


def artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ObservationError(ErrorCode.INVALID_EVENT, "attachment path escapes artifact root", path=path)
    return path


def verify_tensors(path: Path, entry: RoutingAttachment, metadata: Metadata) -> None:
    """Verify tensor bytes, Safetensors metadata and the exact record association."""
    data = read_bytes(path, ErrorCode.INVALID_EVENT)
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ObservationError(ErrorCode.PAYLOAD_CHECKSUM, "Routing attachment SHA-256 mismatch", path=path)
    generation = next(item for item in metadata.generations if item.id == entry.generation)
    try:
        with safe_open(path, framework="numpy") as handle:
            tensor_metadata = handle.metadata()
            if tensor_metadata is None:
                raise ValueError("Routing attachment has no metadata")

            expected = {
                "contract_version": "4",
                "run_id": metadata.run_id,
                "producer_id": str(entry.publisher),
                "generation": generation.fabric_generation,
                "pe": str(entry.pe),
            }
            if any(tensor_metadata.get(key) != value for key, value in expected.items()):
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT, "Routing tensor metadata identity mismatch", path=path
                )
            if int(tensor_metadata["sequence"]) < 1 or int(tensor_metadata["dropped"]) < 0:
                raise ValueError("invalid legacy Routing counters")

            records = TypeAdapter(list[TensorRecord]).validate_json(tensor_metadata["records"])

            matches = [
                i
                for i, record in enumerate(records)
                if (record.owner, record.sequence) == (entry.owner, entry.sequence)
            ]

            if len(matches) != 1:
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT, "Routing tensor record identity missing/duplicated", path=path
                )

            index = matches[0]
            record = records[index]
            facts = entry.model_dump(include=set(TensorRecord.model_fields))

            if record.model_dump() != facts:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "Routing tensor record facts mismatch", path=path)

            expected_keys = [f"records.{index}.topk_ids", f"records.{index}.topk_weights"]
            if entry.tensor_keys != expected_keys or not set(expected_keys) <= set(handle.keys()):
                raise ValueError("Routing tensor keys do not match the record identity")

            all_expected = {f"records.{i}.{name}" for i in range(len(records)) for name in ("topk_ids", "topk_weights")}

            if set(handle.keys()) != all_expected:
                raise ValueError("Routing attachment contains undeclared or missing tensor keys")
            if len(entry.shape) != 2 or entry.dtypes != ["int32", "float32"]:
                raise ValueError("Routing shape/dtype declarations are invalid")

            for key, dtype in zip(expected_keys, ("I32", "F32"), strict=True):
                tensor = handle.get_slice(key)
                if tensor.get_shape() != entry.shape or tensor.get_dtype() != dtype:
                    raise ValueError("Routing tensor shape/dtype mismatch")

            # Force payload reads; metadata/hash validation alone does not prove
            # tensors can be materialized by the declared format.
            for key in expected_keys:
                handle.get_tensor(key)
    except (SafetensorError, ValueError, KeyError, OSError) as exc:
        if isinstance(exc, ObservationError):
            raise
        raise ObservationError(ErrorCode.INVALID_EVENT, "invalid Routing tensor artifact", path=path) from exc


def resolve_attachments(
    indices: tuple[AttachmentIndex, ...],
    root: Path | None,
    events: tuple[Event, ...],
    metadata: tuple[Metadata, ...],
) -> tuple[tuple[AttachmentEvidence, ...], tuple[Finding, ...]]:
    entries: dict[tuple[str, int, int], RoutingAttachment] = {}
    closing: dict[tuple[str, int], bool] = {}

    for index in indices:
        if str(UUID(index.run_id)) != index.run_id or index.owner <= 0:
            raise ObservationError(ErrorCode.INVALID_EVENT, "invalid attachment index owner")

        key = (index.run_id, index.owner)

        if key in closing:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "multiple final attachment indices for one owner")

        closing[key] = index.closing_complete

        for indexed_entry in index.entries:
            identity = (*key, indexed_entry.sequence)
            if indexed_entry.owner != index.owner or identity in entries:
                raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "attachment index owner/identity conflict")
            entries[identity] = indexed_entry

    publications: dict[tuple[str, int, int], list[Event]] = {}

    for event in events:
        r = event.record
        if r.kind == EventKind.ROUTING_PUBLISHED:
            identity = (event.source.identity.run_id, r.attachment_owner, r.attachment_sequence)
            publications.setdefault(identity, []).append(event)

    evidence: list[AttachmentEvidence] = []
    findings: list[Finding] = []

    for identity in sorted(entries.keys() | publications.keys()):
        run_id, owner, sequence = identity
        entry = entries.get(identity)

        published_events = publications.get(identity, [])
        sources = tuple(event.source for event in published_events)
        document = next((m for m in metadata if m.run_id == run_id and any(p.id == owner for p in m.producers)), None)

        if document is None:
            raise ObservationError(
                ErrorCode.METADATA_UNAVAILABLE, "attachment owner metadata unavailable", producer=owner
            )

        if entry is not None:
            producer = next((p for p in document.producers if p.id == entry.publisher), None)
            generation = next((g for g in document.generations if g.id == entry.generation), None)

            if (
                producer is None
                or generation is None
                or entry.publisher != owner
                or producer.clock_domain_id != entry.clock_domain_id
                or not any(m.instance_index == entry.instance_index for m in generation.models)
                or not generation.atnagent_count <= entry.pe < generation.atnagent_count + generation.ffnagent_count
                or document.pe_for(entry.generation, entry.publisher) != entry.pe
            ):
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT,
                    "invalid Routing index references",
                    producer=owner,
                    generation=entry.generation,
                    pe=entry.pe,
                    metadata_field_path=(
                        f"generations[{document.generations.index(generation)}].pe_members"
                        if generation is not None
                        else "generations"
                    ),
                )

        for event in published_events:
            r = event.record
            known_status = attachment_outcome_names.get(r.data0)

            # Artifact IDs are local to the referencing event's metadata revision.
            if entry is not None and r.data2 != 0:
                artifact = next(item for item in event.metadata.artifacts if item.id == r.data2)
                if entry.status != "published" or (artifact.path, artifact.sha256) != (entry.path, entry.sha256):
                    raise event.error(
                        ErrorCode.IDENTITY_CONFLICT, "Routing artifact reference disagrees with attachment index"
                    )

            if entry is not None and known_status is not None and entry.status != known_status:
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT,
                    "Routing event/index outcome mismatch",
                    path=event.source.path,
                    offset=event.source.offset,
                    producer=owner,
                    event_sequence=r.event_sequence,
                )

            if entry is not None and (
                (
                    entry.generation,
                    entry.instance_index,
                    entry.invocation_sequence,
                    entry.lane_index,
                    entry.lease_sequence,
                    entry.layer_ordinal,
                    entry.publisher,
                    entry.pe,
                    entry.clock_domain_id,
                    entry.raw_publication_time_ns,
                    entry.shape,
                )
                != (
                    r.generation,
                    r.instance_index,
                    r.invocation_sequence,
                    r.lane_index,
                    r.lease_sequence,
                    r.layer_ordinal,
                    event.source.identity.producer,
                    r.pe,
                    r.clock_domain_id,
                    r.raw_time_ns,
                    [r.payload_rows, r.data1],
                )
            ):
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT,
                    "Routing event/index facts mismatch",
                    path=event.source.path,
                    offset=event.source.offset,
                    producer=owner,
                    event_sequence=r.event_sequence,
                )

        path: Path | None = None
        available = False

        if entry is None:
            # An explicit non-reserved event outcome survives a lost index.
            states = {event.record.data0 for event in published_events}
            if len(states) > 1:
                raise ObservationError(
                    ErrorCode.IDENTITY_CONFLICT, "conflicting Routing reservation outcomes", producer=owner
                )
            state = next(iter(states), AttachmentState.RESERVED)
            status = attachment_outcome_names.get(
                state, "missing" if closing.get((run_id, owner), False) else "unknown"
            )
        else:
            status = entry.status
            if status == "published":
                if root is None or entry.path is None:
                    raise ObservationError(
                        ErrorCode.METADATA_UNAVAILABLE, "attachment root unavailable", producer=owner
                    )
                path = artifact_path(root, entry.path)
                if not path.exists():
                    status = "missing"
                else:
                    verify_tensors(path, entry, document)
                    available = True

        evidence.append(AttachmentEvidence(run_id, owner, sequence, entry, status, available, path, sources))

        if not published_events:
            findings.append(
                Finding("event_missing", "attachment retained without publication event", (), run_id, owner)
            )

        if status != "published":
            code = "tail_unknown" if status == "unknown" else f"attachment_{status}"
            findings.append(Finding(code, "Routing attachment unavailable", sources, run_id, owner))

    return tuple(evidence), tuple(findings)


def read_artifacts(inputs: ArtifactInputs) -> ArtifactSet:
    """Read explicit published inputs, preserving incomplete valid evidence.

    Raises:
        ObservationError: For I/O, unsupported versions, invalid data or conflicts.
    """
    raw_metadata: dict[tuple[str, int, int], bytes] = {}
    metadata: list[Metadata] = []

    for path in inputs.metadata:
        data = read_bytes(path, ErrorCode.METADATA_UNAVAILABLE)

        document = Metadata.parse_artifact(data, path, ErrorCode.METADATA_UNAVAILABLE)

        key = (document.run_id, document.producer_id, document.revision)

        if key in raw_metadata and raw_metadata[key] != data:
            raise ObservationError(ErrorCode.IDENTITY_CONFLICT, "metadata revision has differing bytes", path=path)

        raw_metadata[key] = data
        metadata.append(document)

    chunks: list[Chunk] = []
    chunk_bytes: dict[tuple[str, int, int], bytes] = {}

    for path in inputs.chunks:
        if path.suffix != ".xpool":
            raise ObservationError(ErrorCode.INVALID_LAYOUT, "only published .xpool chunks are accepted", path=path)

        data = read_bytes(path, ErrorCode.INVALID_LENGTH)
        metadata_bytes = None

        if len(data) >= 128:
            producer = struct.unpack_from("<Q", data, 16)[0]
            revision = struct.unpack_from("<Q", data, 32)[0]
            run_id = str(UUID(bytes=data[96:112]))
            metadata_bytes = raw_metadata.get((run_id, producer, revision))

        chunk = parse_chunk(data, metadata_bytes, path=path)
        identity = (chunk.header.run_id, chunk.header.producer, chunk.header.chunk_sequence)

        if identity in chunk_bytes and chunk_bytes[identity] != data:
            raise ObservationError(
                ErrorCode.IDENTITY_CONFLICT, "chunk publication identity has differing bytes", path=path
            )

        chunk_bytes[identity] = data
        chunks.append(chunk)

    quality: list[QualityCheckpoint] = []

    for path in inputs.quality:
        checkpoint = QualityCheckpoint.parse_artifact(
            read_bytes(path, ErrorCode.INVALID_EVENT), path, ErrorCode.INVALID_EVENT
        )

        if not any(
            m.run_id == checkpoint.run_id and any(p.id == checkpoint.producer_id for p in m.producers) for m in metadata
        ):
            raise ObservationError(ErrorCode.METADATA_UNAVAILABLE, "checkpoint Producer unavailable", path=path)

        quality.append(checkpoint)

    events = tuple(event for chunk in chunks for event in chunk.events)

    attachments, findings = resolve_attachments(
        inputs.attachment_indices, inputs.attachment_root, events, tuple(metadata)
    )

    return ArtifactSet(tuple(chunks), tuple(metadata), tuple(quality), attachments, findings)
