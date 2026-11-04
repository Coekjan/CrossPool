# Observation traces

CrossPool validates published observation artifacts, exports native Perfetto
protobuf traces, and queries them with a pinned Trace Processor. These commands
operate on completed offline inputs and require no running deployment or model
configuration. The installed CrossPool toolchain remains required.

## Prepare the processor

The repository pins the official Perfetto Python package through uv. The
[qualified toolchain](../../src/xpool/observability/trace/versions.py) owns the
expected package version, processor release and digest. Explicit setup downloads
the qualified Linux x86-64 Trace Processor,
verifies its SHA-256, and publishes an executable under the configured
`cache_root/perfetto`:

```bash
uv run xpool trace setup
```

Use `--tool-dir tools/perfetto` to select another installation directory. Queries
accept `--processor tools/perfetto/trace_processor_shell`; an explicit binary
must match the same qualified digest. Setup reuses a valid installation and
reports an invalid existing file. Remove or replace an invalid tool deliberately
before running setup again. Query commands never download or upgrade tools.
Setup needs HTTPS access to the official Perfetto artifact host; subsequent
validation, export and queries run offline.

## Declare raw inputs

Create an explicit JSON inventory relative to the input directory. The inventory
itself may live elsewhere. Paths resolve within the input directory, including
symlink targets; the reader does not discover files or infer publication state.

```json
{
  "version": 1,
  "chunks": ["producer-1.1.xpool"],
  "metadata": ["producer-1.1.json"],
  "quality": [],
  "attachment_indices": [],
  "attachment_root": null,
  "samplers": []
}
```

`attachment_indices` entries contain `run_id`, `owner`, `entries` and
`closing_complete`. Each entry reuses the `RoutingAttachment` schema. The
optional `attachment_root` resolves relative to the input directory; its default
is that directory. Graph structures and Routing tensors remain referenced files.

`samplers` reuses the `SamplerDeclaration` schema documented at its declaration
and in [offline calibration](../designs/devkit.md#offline-clock-calibration).
Every retained sampler must be declared. Attempt ranges, measurement conditions
and diagnostics are caller evidence; retained samples cannot establish their
completeness. Unknown attempted ranges produce unavailable calibration models.
Device domains with no samples remain queryable without calibration.

```bash
uv run xpool trace validate \
  --input-dir raw-run --manifest inputs.json --output-dir validation

uv run xpool trace export \
  --input-dir raw-run --manifest inputs.json --output-dir trace-bundle
```

Each output directory must be new and separate from its inputs. Derived outputs
are outside runtime raw-file quota accounting. Validation writes calibration and
quality reports. Export writes those reports, `.pftrace` files, and `bundle.json`
as the final completion marker. A failed export may leave partial files; a
directory without the completion marker is not a queryable bundle. Retry with a
new output directory after resolving the failure.

## Inspect and query

Open individual `.pftrace` files with [Perfetto UI](https://ui.perfetto.dev).
The bundle index identifies each file's Run, reference, display origin and hash.
Unified files share one evidenced Host reference. Raw files preserve each
domain independently, including full intervals crossing calibration boundaries.
Display timestamps are relative to the recorded origin. Raw integer timestamps
and exact calibration estimates remain in event arguments.

```bash
# All seven built-in queries; results go to query.json.
uv run xpool trace query --input-dir trace-bundle --output-dir trace-results

# Select one or more built-in queries.
uv run xpool trace query --input-dir trace-bundle --output-dir graph-results \
  --name graph --name calibration

# Run the same SQL independently against every trace in the bundle.
uv run xpool trace query --input-dir trace-bundle --output-dir sql-results \
  --sql docs/examples/observability/identities.sql
```

Built-in names are `graph`, `transport`, `invocation`, `lane-lease`, `routing`,
`calibration`, and `quality`. They query Trace Processor's `xpool_evidence` view
and deduplicate raw/unified representations by evidence identity, preferring
the unified representation. Results retain the trace that supplied each row;
an empty per-trace result after deduplication is valid.

User SQL runs separately on each member. Slice IDs and timestamps stay local to
that file. Each JSON result includes its trace, query, columns and rows. SQL
columns must have unique names. Integers and nulls remain JSON integers and
nulls; BLOBs use `{ "encoding": "base64", "data": "..." }`. Non-finite
numbers and SQL failures are explicit errors, and no complete result is
published if any query fails.

The `xpool_evidence` view exposes `id`, `ts`, `dur`, `track_id`, `name`,
`evidence_id`, `category`, and `payload`. `payload` is JSON containing original
identities, source endpoints, raw records, attachment references, calibration
evidence, or derived interval and relation evidence. Built-in queries decode
that payload into structured JSON. User SQL can use `json_extract` to select
identity fields. Standard Perfetto `slice`, `flow`, `counter` and related tables
are also available.

## Interpret timing and data quality

Normal slices require complete observed raw boundaries. Missing, conflicting or
failed boundaries remain instants with status and provenance. Semantic tracks
reuse overlap slots rather than retaining one track per Invocation. Graph
Capture, Host call and Device activity remain separate phases.

The exporter computes unified timestamps with CrossPool calibration, retaining
exact estimates and both uncertainty bounds. Evidence-backed ClockSnapshots
record correspondences; they do not establish calibration or fit a clock rate.
Unknown mappings remain in raw traces and explicit gap events. A slice whose
endpoints map but whose interior is not covered by one unambiguous model remains
visible with `calibration_coverage: incomplete`. It does not prove continuous
calibration coverage.

Interval and protocol-relation payloads contain conditional duration estimates,
uncertainty, ordering, or an explicit unavailable result. Entity association
events prove identities, not arbitrary start/end times. Visual flows require
concrete protocol endpoints in the same file. Relations spanning independent
files remain queryable; reversed display estimates retain the relation with
`flow_status: display_time_reversed` rather than inventing a visual ordering.
Perfetto scalar `dur` is a display value, not a fixed-precision timing promise.

Cumulative counters are placed at their observed Host checkpoint time. Their
payload retains scope, finality and saturation evidence. Values exceeding the
exact integer range of Perfetto's floating-point counter table remain exact in
checkpoint JSON and produce an explicit counter-display limitation event.
Retained-event counts never substitute for unknown runtime totals.

The [synthetic example bundle](../examples/observability/README.md) illustrates
all four event families, mapped and unmapped domains, and unavailable Routing
tensors. It is not measured serving or hardware precision evidence.

## Failures and acceptance

Library callers receive structured `ObservationError` or `TraceError` categories
with file/query context and chained causes. Missing or mismatched processor
binaries require explicit setup. Invalid inputs, overlapping output paths,
existing output directories, corrupted bundles and SQL failures return CLI
status 2. Missing dependencies are reported; they are not silently bypassed.

After setup, run the CPU-only acceptance suites through the repository toolchain:

```bash
uv run pytest tests/suites/unit/observability tests/suites/integration/observability
```

Trace Processor integration tests skip with setup instructions when its binary
is absent. A qualification run must prepare the tool and execute these tests.
Synthetic acceptance establishes offline export/query consistency. Real loaded,
MPS, multiprocess and multi-device sampling precision needs separate evidence.
