"""Pinned Trace Processor queries over independently timed trace bundle members."""

import base64
import hashlib
import json
import math
from pathlib import Path

from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
from pydantic import JsonValue, TypeAdapter

from xpool.observability.trace import versions
from xpool.observability.trace.inputs import confined, prepare_output
from xpool.observability.trace.models import QueryName, QueryResult, TraceBundle, TraceError, TraceErrorCode
from xpool.observability.trace.tools import verify_processor

evidence_view = """
CREATE PERFETTO VIEW xpool_evidence AS
SELECT id, ts, dur, track_id, name,
       EXTRACT_ARG(arg_set_id, 'debug.evidence_id') AS evidence_id,
       EXTRACT_ARG(arg_set_id, 'debug.category') AS category,
       EXTRACT_ARG(arg_set_id, 'debug.payload') AS payload
FROM slice WHERE EXTRACT_ARG(arg_set_id, 'debug.evidence_id') IS NOT NULL;
"""

query_categories: dict[QueryName, tuple[str, ...]] = {
    QueryName.GRAPH: (
        "graph",
        "graph_template",
        "graph_execution",
        "graph_to_operation",
        "invocation_to_graph_activity",
        "execution_of_template",
    ),
    QueryName.TRANSPORT: ("transport", "operation", "operation_to_invocation", "graph_to_operation", "protocol"),
    QueryName.INVOCATION: (
        "fabric",
        "invocation",
        "operation_to_invocation",
        "invocation_to_graph_activity",
        "invocation_lease",
        "invocation_attachment",
        "protocol",
    ),
    QueryName.LANE_LEASE: ("lease", "invocation_lease"),
    QueryName.ROUTING: ("routing", "attachment", "invocation_attachment"),
    QueryName.CALIBRATION: ("calibration",),
    QueryName.QUALITY: ("quality",),
}


def builtin_sql(name: QueryName) -> str:
    """Return a standard query over xpool_evidence, available to user SQL too."""
    categories = ",".join(f"'{category}'" for category in query_categories[name])
    return (
        "SELECT evidence_id, category, payload FROM xpool_evidence "
        f"WHERE category IN ({categories}) ORDER BY evidence_id"
    )


def read_bundle(directory: Path) -> TraceBundle:
    """Verify completion and trace digests before invoking any external tool."""
    path = directory / "bundle.json"
    try:
        bundle = TraceBundle.model_validate_json(path.read_bytes())
        if (
            bundle.perfetto_version != versions.perfetto_version
            or bundle.processor_version != versions.processor_version
        ):
            raise TraceError(TraceErrorCode.TOOL_VERSION, "bundle uses an unqualified tool version", path=path)
        names: set[str] = set()
        for entry in bundle.traces:
            trace = confined(directory, entry.path)
            if entry.path in names:
                raise TraceError(TraceErrorCode.INPUT, "duplicate bundle member", path=trace)
            names.add(entry.path)
            with trace.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry.sha256:
                raise TraceError(TraceErrorCode.INPUT, "trace checksum mismatch", path=trace)
    except (OSError, ValueError) as error:
        if isinstance(error, TraceError):
            raise
        raise TraceError(TraceErrorCode.INPUT, "invalid or incomplete trace bundle", path=path) from error
    return bundle


def query_bundle(
    directory: Path, processor: Path, *, names: tuple[QueryName, ...] = tuple(QueryName), sql: str | None = None
) -> tuple[QueryResult, ...]:
    """Execute built-ins or user SQL on each trace, without downloads or updates.

    Built-ins deduplicate raw/unified representations by evidence identity,
    preferring the unified view. User SQL keeps independent result sets and
    local slice IDs. The xpool_evidence view exposes raw provenance and calibrated
    duration bounds in its JSON payload; display dur is not a precision claim.
    SQL errors fail the operation with query and member context. BLOB cells
    use an explicit base64 encoding object; non-finite numbers are rejected.
    """
    binary = verify_processor(processor)
    bundle = read_bundle(directory)
    statements = {"sql": sql} if sql is not None else {name.value: builtin_sql(name) for name in names}
    results: list[QueryResult] = []
    seen: dict[str, set[str]] = {name: set() for name in statements}
    for entry in sorted(bundle.traces, key=lambda item: (item.kind != "unified", item.path)):
        path = confined(directory, entry.path)
        # The upstream startup boundary raises heterogeneous exceptions. Wrap
        # exactly this owned processor lifetime, retaining the concrete cause.
        try:
            with TraceProcessor(trace=str(path), config=TraceProcessorConfig(bin_path=str(binary))) as trace:
                trace.query(evidence_view)
                for name, statement in statements.items():
                    try:
                        result = trace.query(statement)
                        columns = tuple(result.column_names)
                        if len(set(columns)) != len(columns):
                            raise TraceError(
                                TraceErrorCode.QUERY, "SQL result columns must have unique names", path=path, query=name
                            )
                        rows: list[tuple[JsonValue, ...]] = []
                        for row in result:
                            # Perfetto rows expose SQL-selected column names
                            # dynamically. Establish JSON values immediately.
                            values: list[JsonValue] = []
                            for column in columns:
                                cell = getattr(row, column)
                                if isinstance(cell, bytes):
                                    cell = {"encoding": "base64", "data": base64.b64encode(cell).decode("ascii")}
                                if isinstance(cell, float) and not math.isfinite(cell):
                                    raise TraceError(TraceErrorCode.QUERY, "non-finite SQL cell", path=path, query=name)
                                values.append(TypeAdapter(JsonValue).validate_python(cell))
                            if sql is None:
                                identity = str(values[0])
                                if identity in seen[name]:
                                    continue
                                seen[name].add(identity)
                                if isinstance(values[2], str):
                                    values[2] = TypeAdapter(JsonValue).validate_json(values[2])
                            rows.append(tuple(values))
                        results.append(QueryResult(trace=entry.path, query=name, columns=columns, rows=tuple(rows)))
                    except Exception as error:
                        if isinstance(error, TraceError):
                            raise
                        raise TraceError(
                            TraceErrorCode.QUERY, f"SQL execution failed: {error}", path=path, query=name
                        ) from error
        except Exception as error:
            if isinstance(error, TraceError):
                raise
            raise TraceError(TraceErrorCode.QUERY, f"trace processor failed: {error}", path=path) from error
    return tuple(results)


def write_queries(
    directory: Path,
    processor: Path,
    output: Path,
    *,
    names: tuple[QueryName, ...] = tuple(QueryName),
    sql_file: Path | None = None,
) -> tuple[QueryResult, ...]:
    """Publish complete JSON results to a new directory separate from the bundle.

    User SQL files remain unchanged and cannot be overwritten by publication.
    query.json is created only after all member queries have succeeded.
    """
    protected = (directory,) if sql_file is None else (directory, sql_file)
    prepare_output(output, protected)
    sql = sql_file.read_text() if sql_file is not None else None
    result = query_bundle(directory, processor, names=names, sql=sql)
    payload = TypeAdapter(tuple[QueryResult, ...]).dump_python(result, mode="json")
    completion = output / "query.json.tmp"
    completion.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    completion.replace(output / "query.json")
    return result
