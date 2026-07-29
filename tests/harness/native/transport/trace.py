"""Transport observer snapshot persistence for native tests."""

from __future__ import annotations

import json
from pathlib import Path

import xpool.native

TRACE_FIELDS = (
    "trace_id",
    "payload_rows",
    "layer_ordinal",
    "forward_mode",
    "result_handoff",
    "dp_padding_mode",
    "result_code",
    "staging_started",
    "staging_completed",
    "published",
    "published_observed",
    "execution_started",
    "execution_admitted",
    "execution_completed",
    "evaluated",
    "evaluated_observed",
    "output_copied",
    "acknowledged",
    "closed",
)


def write_transport_trace(arena: str, output_path: Path) -> None:
    """Verify stable trace reads and write the raw observer payload."""

    first = xpool.native.transport.read_trace(arena)
    second = xpool.native.transport.read_trace(arena)
    if first is None or second is None:
        raise RuntimeError("enabled transport observer returned no trace snapshot")
    if (
        first.sequence != second.sequence
        or first.dropped != second.dropped
        or len(first.records) != len(second.records)
    ):
        raise RuntimeError("repeated owner transport trace reads differ")
    records = [{field: int(getattr(record, field)) for field in TRACE_FIELDS} for record in first.records]
    output_path.write_text(
        json.dumps({"sequence": first.sequence, "dropped": first.dropped, "records": records}),
        encoding="utf-8",
    )
