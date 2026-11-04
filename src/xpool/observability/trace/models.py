"""Versioned offline trace inputs, publication indices and structured failures."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from xpool.observability.calibration_index import SamplerDeclaration
from xpool.observability.models import RoutingAttachment, StrictModel, positive_uint64
from xpool.observability.trace import versions


class TraceErrorCode(StrEnum):
    """Stable categories for input, publication, processor and SQL failures."""

    INPUT = "input"
    OUTPUT = "output"
    DEPENDENCY_MISSING = "dependency_missing"
    TOOL_VERSION = "tool_version"
    TOOL_SETUP = "tool_setup"
    QUERY = "query"
    EXPORT = "export"


class TraceError(ValueError):
    """Offline failure retaining its category, file and query context.

    I/O and processor owners chain the original exception. Consumers branch on
    code rather than parsing the human-readable diagnostic.
    """

    def __init__(
        self, code: TraceErrorCode, detail: str, *, path: Path | None = None, query: str | None = None
    ) -> None:
        self.code = code
        self.detail = detail
        self.path = path
        self.query = query
        super().__init__(
            f"xpool trace {code.value}: {detail}"
            + (f" [query={query}]" if query else "")
            + (f" ({path})" if path else "")
        )


class AttachmentInput(StrictModel):
    run_id: str
    owner: positive_uint64
    entries: tuple[RoutingAttachment, ...]
    closing_complete: bool = False


class TraceInputs(StrictModel):
    """Explicit inventory relative to the separately supplied input directory.

    Files must resolve within that directory. Sampler declarations are caller
    evidence, with the same completeness semantics as build_calibration().
    """

    version: Literal[1] = 1
    chunks: tuple[str, ...]
    metadata: tuple[str, ...]
    quality: tuple[str, ...] = ()
    attachment_indices: tuple[AttachmentInput, ...] = ()
    attachment_root: str | None = None
    samplers: tuple[SamplerDeclaration, ...] = ()


class TraceFile(StrictModel):
    path: str
    sha256: str
    run_id: str
    kind: Literal["unified", "raw"]
    reference: str
    origin_ns: int


class InputDigest(StrictModel):
    path: str
    sha256: str


class TraceBundle(StrictModel):
    """Completion marker published after every trace and evidence report.

    Trace paths are relative to the bundle directory. Raw input paths remain
    references; their hashes identify the evidence consumed by this export.
    """

    version: Literal[1] = 1
    perfetto_version: str = versions.perfetto_version
    processor_version: str = versions.processor_version
    traces: tuple[TraceFile, ...]
    inputs: tuple[InputDigest, ...] = ()


class QueryName(StrEnum):
    """Supported built-ins over evidence identities, independent of display time."""

    GRAPH = "graph"
    TRANSPORT = "transport"
    INVOCATION = "invocation"
    LANE_LEASE = "lane-lease"
    ROUTING = "routing"
    CALIBRATION = "calibration"
    QUALITY = "quality"


class QueryResult(StrictModel):
    """One query's ordered columns and rows from a specific trace member.

    User SQL preserves independent per-member results. Built-ins remove earlier
    evidence identities, preferring unified members, so some rows may be empty.
    """

    trace: str
    query: str
    columns: tuple[str, ...]
    rows: tuple[tuple[JsonValue, ...], ...]
