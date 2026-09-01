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
    "output_requirement",
    "dp_row_layout",
    "result_code",
    "request_staging_started",
    "request_staging_completed",
    "request_published",
    "request_observed",
    "execution_started",
    "execution_completed",
    "result_published",
    "result_observed",
    "output_copied",
    "result_acknowledged",
    "closed",
)


def write_transport_trace(output_path: Path, *, site: str) -> None:
    """Verify stable trace reads and write the raw observer payload."""

    first = xpool.native.devkit.transport_observer.read()
    second = xpool.native.devkit.transport_observer.read()
    if first is None or second is None:
        raise RuntimeError("enabled transport observer returned no trace snapshot")
    first_endpoint = first.endpoints[0]
    second_endpoint = second.endpoints[0]
    if (
        first_endpoint.sequence != second_endpoint.sequence
        or first_endpoint.dropped != second_endpoint.dropped
        or len(first_endpoint.records) != len(second_endpoint.records)
    ):
        raise RuntimeError("repeated transport observer reads differ")
    records = [{field: int(getattr(record, field)) for field in TRACE_FIELDS} for record in first_endpoint.records]
    output_path.write_text(
        json.dumps(
            {
                "site": site,
                "sequence": first_endpoint.sequence,
                "dropped": first_endpoint.dropped,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
