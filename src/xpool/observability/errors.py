"""Structured failures at observation artifact boundaries."""

from enum import StrEnum
from pathlib import Path

from xpool.observability.provenance import Source

__all__ = ["ErrorCode", "ObservationError"]


class ErrorCode(StrEnum):
    """Stable artifact error categories."""

    HEADER_TRUNCATED = "header_truncated"
    BAD_MAGIC = "bad_magic"
    UNSUPPORTED_VERSION = "unsupported_version"
    INVALID_LAYOUT = "invalid_layout"
    HEADER_CHECKSUM = "header_checksum"
    INVALID_LENGTH = "invalid_length"
    METADATA_UNAVAILABLE = "metadata_unavailable"
    METADATA_CHECKSUM = "metadata_checksum"
    PAYLOAD_CHECKSUM = "payload_checksum"
    INVALID_EVENT = "invalid_event"
    UNKNOWN_KIND = "unknown_kind"
    IDENTITY_CONFLICT = "identity_conflict"
    EVENT_CONFLICT = "event_conflict"


class ObservationError(ValueError):
    """An artifact failure with machine-readable category and source context.

    ``offset`` is a byte offset in ``path``. Producer and attempted-event sequence
    identify the recording source when known. Conflicting evidence retains all
    supplied ``sources``; metadata failures expose the offending JSON field path,
    Generation and PE when available. Callers branch on ``code`` and context
    fields rather than diagnostic text. Parsing and I/O owners chain root causes.
    """

    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        *,
        path: Path | None = None,
        offset: int | None = None,
        producer: int | None = None,
        event_sequence: int | None = None,
        sources: tuple[Source, ...] = (),
        metadata_field_path: str | None = None,
        generation: int | None = None,
        pe: int | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.path = path
        self.offset = offset
        self.producer = producer
        self.event_sequence = event_sequence
        self.metadata_field_path = metadata_field_path
        self.generation = generation
        self.pe = pe
        self.sources = sources
        super().__init__(f"{code.value}: {detail}" + (f" ({path}:{offset})" if path else ""))
