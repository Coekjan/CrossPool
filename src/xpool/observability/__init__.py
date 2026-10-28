"""Strict offline observation artifacts, identity-based reconstruction and legacy views."""

from xpool.observability.artifacts import (
    ArtifactInputs,
    ArtifactSet,
    AttachmentEvidence,
    AttachmentIndex,
    read_artifacts,
)
from xpool.observability.codec import Chunk, ChunkHeader, encode_chunk, parse_chunk
from xpool.observability.entities import CausalEdge, Entity, EntityIdentity, Observation
from xpool.observability.errors import ErrorCode, ObservationError
from xpool.observability.evidence import (
    EvidenceReason,
    EvidenceStatus,
    FieldEvidence,
    ProjectionStatus,
    SnapshotEvidenceEntry,
    SnapshotEvidenceReport,
    SnapshotObserver,
    SnapshotRole,
)
from xpool.observability.intervals import Interval
from xpool.observability.models import (
    CalibrationSample,
    Event,
    EventRecord,
    Finding,
    Metadata,
    PeMember,
    QualityCheckpoint,
    RoutingAttachment,
)
from xpool.observability.provenance import EventIdentity, Source
from xpool.observability.quality import QualityReport
from xpool.observability.reconstruction import Reconstruction, reconstruct
from xpool.observability.snapshots import LegacyProjection, SnapshotRequest, project_legacy_snapshots

__all__ = [
    "ArtifactInputs",
    "ArtifactSet",
    "AttachmentEvidence",
    "AttachmentIndex",
    "CalibrationSample",
    "CausalEdge",
    "Chunk",
    "ChunkHeader",
    "Entity",
    "EntityIdentity",
    "ErrorCode",
    "Event",
    "EventIdentity",
    "EventRecord",
    "EvidenceReason",
    "EvidenceStatus",
    "FieldEvidence",
    "Finding",
    "Interval",
    "LegacyProjection",
    "Metadata",
    "Observation",
    "ObservationError",
    "PeMember",
    "ProjectionStatus",
    "QualityCheckpoint",
    "QualityReport",
    "Reconstruction",
    "RoutingAttachment",
    "SnapshotEvidenceEntry",
    "SnapshotEvidenceReport",
    "SnapshotObserver",
    "SnapshotRequest",
    "SnapshotRole",
    "Source",
    "encode_chunk",
    "parse_chunk",
    "project_legacy_snapshots",
    "read_artifacts",
    "reconstruct",
]
