# Devkit

Devkit observes real production execution through typed Hook Points. Transport,
Fabric, and FFN computation remain production responsibilities.

Debug configuration is process-global and immutable after native
initialization. Native owners read the installed value directly; Debug policy
stays out of runtime state, adapters, and Kernel arguments. The configuration
contains Transport, Fabric, Graph, and FFN Routing Observer options.

Each Observer records evidence from the real execution path and owns
process-local storage. Observer state is never stored in a Transport or Fabric
arena. Fabric and FFN Routing Observer allocations are included in FFN memory
admission through their native sizing queries. Transport Observer storage is a
normal attention-side allocation. The AtnAgent post-capture device-memory
observation therefore already reflects its bytes; it needs no duplicate
planning field.

Every Observer owns its record and snapshot values and exposes reads through
its own `xpool.native.devkit.<observer>` submodule. An enabled Observer that
cannot allocate or install fails startup. Exhausted Device record capacity
increments the Observer's dropped count; an Observer destination-bounds failure
omits that record rather than corrupting storage or changing execution.
Snapshot read failures are reported to the Host reader and do not alter an
already completed invocation.

## Hook contract

Host and Device call sites use the neutral typed surface:

```cpp
Point::hooks(Point::Context{...});
```

A production call site names only its owning Point. Host and Device adapters
form an explicit compile-visible catalog. Hook Points provide observation; the
production path owns control flow and execution selection.

Observe calls are additive, so the surrounding production code remains readable
and complete without them. Collective Device Points are reached by every
participating CTA thread; adapters that need only one observation select thread
zero themselves, while an Observer that copies a collective payload may use the
complete CTA.

## Protocol and Graph evidence

Protocol events correspond to real publication or observation boundaries.
Events are not inserted merely to separate adjacent calls or to expose a
debug-only algorithm.

Capacity evidence is recorded by the FfnAgent that selected the installed
Capacity and contains both live rows and selected row capacity. Delivery
evidence is recorded from the production delivery derivation. Coordinator
records do not infer either fact.

Routing evidence follows the collective `RoutingMetadataPublished` event after
the production payload and identity are published. Graph evidence records
Primary captures and installed Lane Graphs at their owning lifecycle Points.
Observers record evidence; runtime lifecycle and Graph topology remain owned by
production execution.
Native Graph snapshots expose CUDA node kinds as underlying integers; the
Python Graph Observer resolves official cuda-python enum names only when
constructing its JSON presentation.

## Serialized observations

Python Fabric and Transport writers own concrete JSON declarations shared by
readers and fixtures. These projections differ from native snapshots: they
flatten role facts, serialize enums, derive durations and add process or model
context. Matching-role native getters determine field types and nullability.
Shared declarations in `xpool.devkit.phases` own the Transport and role-specific
Fabric phase names and timestamp pairs. Native snapshot writers consume those
field names directly; offline reconstruction resolves them to wire Event kinds.
Raw intervals retain recorded zero timestamps, while legacy durations keep their
existing zero-as-missing projection rule.
Incomplete records remain diagnostic input: AtnAgent lane and lease facts are
nullable until AdmissionObserved, and Coordinator lane and lease facts until
Scheduled.

Transport snapshot filenames combine PID, endpoint site, the Model ID's
[URI component representation](control-plane.md#configuration-and-integration)
and Instance Rank. Each model retains its own process-local snapshot file;
the JSON payload carries its canonical scalar Model ID for readers.

The production SGLang Devkit package owns its lightweight Graph-event
declaration, shared by the event writer and readers independently of Graph
runner implementations. Native Graph snapshots retain their existing production
model. [Tooling](tooling.md#shared-serving-lifecycle) owns file-input validation
and the separate semantic qualification assertions.

## Offline observation artifacts

`xpool.observability` reads the version-four raw observation contract through an
explicit inventory of published chunk, metadata and quality files. The caller
supplies typed Routing attachment indices, their Run/owner completion scopes,
and an attachment root. It does not discover directories, parse temporary tails,
or write conversion files. It uses the complete CrossPool installation because
legacy projection reuses the existing Observer JSON declarations.

The independent codec uses declared little-endian widths, never Native Buffer
layouts. It validates header layout/CRC, declared lengths, exact metadata hashes
and identities with metadata membership semantics, payload CRC, and event semantics
in that order. JSON inputs
reject duplicate keys, coercion and non-finite constants. Errors retain a stable
`ErrorCode`, path, offset and available Producer/event identity; underlying I/O
and parsing failures remain attached as causes. Repeated scalar conflicts and
conflicting Event boundaries also retain both source references; a local scalar
conflict prefers the earlier local observation over an equivalent remote fact.
Chunks use `XPOOLC04` and version 4; metadata, quality checkpoints and Routing
tensors carry the same contract version. Other versions fail explicitly.
Generation topology preserves optional FFN TP width when supplied in metadata.

Every Generation owns an ordered `pe_members` table covering its complete Fabric
PE range. Each member pairs registered Host and Device Producers with identical
process identity and the member's Device clock ownership. Instance processes are
nonmembers. `Metadata.pe_for()` resolves a Producer within an explicit Generation;
absent Generation evidence and nonmembers return `None`, while an unresolved
explicit Generation reference fails. The same Producer may move to another PE in
a new Generation; an existing Generation's definitions remain immutable across
metadata revisions. Coordinator events belong to the first FfnAgent PE and share
that process's Producer and local Fabric trace-ID space. Graph events can resolve
only their own Generation and recording Producer's membership.

Membership validation checks duplicate bindings before ordered coverage and
process/clock ownership. Malformed members and role placement fail with
`invalid_event`; duplicate bindings, immutable redefinition and Event PE disagreement
fail with `identity_conflict`. Errors expose `metadata_field_path`, `generation`
and `pe` alongside existing artifact and Producer/Event context when available.
Process roles admit only Instance Rank, AtnAgent and FfnAgent sources. Coordinator
and Collector remain Event roles. The Daemon is excluded from registration and
missing-artifact accounting; runtime launch and collection remain separate owners.

Reconstruction keys every identity by Run and registered Producer or protocol
owner. Local trace IDs are only legacy projection facts. Each entity retains
independent site/role observations, source process creation, Endpoint, rank/PE,
raw domain, source file and byte offset. Duplicate events count once while all
locations remain available. Conflicting bytes, repeated facts or identity
associations fail explicitly. Graph templates, executions, Host calls and Device
activity remain distinct. Graph-to-operation one-to-many and TP/DP
operation-to-Invocation many-to-one edges require retained identity evidence.
Protocol edges retain their actual publication/observation endpoints; aggregate
conditions never create per-participant events. No edge depends on time proximity.
Graph owners are Host Producers in the actual executing process; Device Graph
events resolve their owner within that process creation. One Invocation can
reference distinct Graph activities in several owning processes. A retained
Transport/Fabric entry association proves Submission preparation independently
of publication or execution return. Failed or non-returning execution retains
that entry evidence while missing completion boundaries remain unknown.

Intervals require observed endpoints in one raw domain. Missing boundaries,
Graph errors and reversed time remain explicit partial/unavailable/conflict
results, rather than normal slices. There is no cross-domain duration inference.
Calibration events expose raw paired samples, Epoch and source; failed Device
reads and Host-after values stay absent. The offline calibration consumer below
fits these samples independently of causal reconstruction. Perfetto export is
a separate consumer of the resulting mappings.

Routing resolution preserves event and tensor availability independently.
Published tensors require a matching SHA-256, Safetensors record identity,
Invocation, Lease, publisher/domain/time, keys, shape and dtypes. Missing files,
uncollected tensors, pool exhaustion, payload overflow, publication failure and
unknown tails remain different findings. A retained tensor never manufactures
a lost publication event. Relative artifact paths must remain inside their root.
An event's nonzero Routing artifact ID resolves in its own metadata revision
to a published artifact of kind `routing` with a nonempty path and SHA-256.
When a final attachment entry exists, its published status, path and SHA-256
must agree with that reference; conflicts are structured identity errors.
A zero artifact ID continues to resolve through the attachment identity.
Without a final index, even a nonzero reference leaves attachment availability
unknown or missing according to the owner's closing state. Artifact declarations
remain immutable across metadata revisions; later revisions may add new IDs.
Graph structure artifacts remain metadata references rather than execution slices.
Routing indices must also match their publisher's Generation-scoped PE membership,
including when no publication Event survives.

Quality checkpoints are cumulative per Producer and revision; consumers select
the latest rather than summing revisions. Conservation is checked when exact,
regression and conflicts fail, and final totals require an exact closing
checkpoint. Sequence gaps do not identify a loss cause without counters. An
absent source or closing checkpoint remains an evidence gap or unknown tail.

Legacy projection takes explicit per-lifetime operation-attempt totals and
capacities. Request constructors reject Boolean and noninteger numeric values,
require positive domain components and capacity, and admit zero or unknown
operation-attempt totals. Transport domains name their exact Endpoint; Fabric domains name
their Generation/Producer. The earliest local IDs remain eligible even when
opening events are lost, and the full offline inventory remains available.
Fabric snapshot Generation, PE and topology come from metadata. A domain with known
attempt totals can retain its envelope even when all local Events are absent;
unobserved entries keep their role and protocol identities unknown.
Unknown attempt totals leave the whole snapshot envelope unavailable while
preserving individually projected records and their evidence report. Missing
required scalar facts omit only the affected record; unknown site-local results
preserve the legacy sentinel with separate quality. Auxiliary Graph evidence
with an actual local identity can recover only explicitly present facts and
never supplies protocol timestamps or another participant's scalar facts.

Each `LegacyProjection` contains `request`, `snapshot`, `records`,
`evidence_report`, and `findings`. `SnapshotEvidenceReport` owns per-field
provenance, exact or unknown counts, and one entry per eligible local ID.
Entries distinguish retained, omitted, and entirely unobserved operations;
unobserved Fabric IDs retain an unknown role and unknown protocol identities.
Unknown counts expose only positively identified eligible IDs. Disabled or
unavailable observers return no separate report. Offline reports carry no
published snapshot hash; this library does not publish paired snapshot files.

Evidence completeness is independent of legacy terminal classification. A
retained acknowledgement can classify a record as completed while missing
intermediate Hooks leave partial evidence. Repeated scalar facts never restore
missing timestamps. Fields distinguish unknown values, definitely missing
applicable boundaries, unresolved conditional applicability, and inapplicable
paths. Canonical field sources use the smallest Event identity; metadata
references use the first supplied revision containing the needed fact and the
lexically smallest equivalent JSON path. Supply authoritative metadata before
Event-derived revisions when a particular equivalent revision should own report
provenance. Repeated Events retain all offline raw locations but do not expand
per-field report sources.

DirectPartial excludes partial publication and peer waits. Complete delivery
also excludes these Hooks when retained zero-row facts or a sole FfnAgent prove
there can be no destination or active reader with peers. Otherwise missing
participant selection stays unknown. An explicit not-applicable Routing
capability can justify the corresponding exclusion; disabled, unavailable or
enabled observation alone does not establish a layer's Dense/MoE path or its
publisher/waiter selection. Opaque Graph structure references are not parsed to
invent this missing evidence. Closed and non-Ok results do not establish the
exact earlier stopping point.

Raw zero timestamps remain observed evidence but cannot provide legacy
durations. Reversed local Hook order marks partial evidence and preserves raw
values; only invalid phase samples are suppressed. Existing percentile indices,
phase-specific sample counts and terminal classifications are preserved.

Chunks are validated separately; retained offline identities and evidence grow
with input size. Runtime collection memory budgets do not bound this offline
index. The library does not install runtime collectors or alter Observer getters.
Synthetic contract agreement establishes CPU implementation behavior, not real
CUDA concurrency, calibration accuracy or serving acceptance.

### Offline library ownership

The package facade exposes supported artifact inputs, reconstructed values and
snapshot projections. Source declarations own exact signatures and field types.
Immutable Event identities and physical locations live in `provenance`; artifact
schemas, presence semantics and Event error context live in `models`. Entity-local
fact merging and association invariants belong to `entities`. Same-domain interval
rules belong to `intervals`, and cumulative Producer accounting belongs to
`quality`. Reconstruction orchestrates these owners before snapshot and evidence
projection. These dependencies use runtime imports and form an acyclic graph.

The Event record declaration owns serialized field order. The codec reuses that
order with explicit wire widths; independent fixed vectors validate the resulting
bytes. Readers retain all duplicate locations, while field evidence selects its
canonical source independently.

### Library use

```python
from pathlib import Path

from xpool.observability import ArtifactInputs, SnapshotRequest, project_legacy_snapshots, read_artifacts, reconstruct

inputs = ArtifactInputs(
    chunks=(Path("raw/chunks/3/1.xpool"),),
    metadata=(Path("raw/metadata.3.1.json"),),
    quality=(Path("raw/quality.3.1.json"),),
)
view = reconstruct(read_artifacts(inputs))
# These identity values and original attempt count come from the caller's
# inventory, not from event count or the largest observed local ID.
(projection,) = project_legacy_snapshots(
    view,
    (SnapshotRequest(view.metadata[0].run_id, producer=3, generation=1,
                     operations_attempted=10, endpoint=(2, 1)),),
)
records = projection.records
report = projection.evidence_report
# snapshot is None when totals or required envelope metadata are unknown.
snapshot = projection.snapshot
```

Pass `AttachmentIndex` values and `attachment_root` when resolving Routing
tensors. `parse_chunk`/`encode_chunk` also expose the independent in-memory codec.
`reconstruct` accepts either a verified `ArtifactSet` or already decoded `Event`
values with optional metadata and checkpoints. With no final attachment index,
reserved Routing outcomes remain unknown. Supported signatures and concrete
models are declared in `src/xpool/observability/`.

## Offline clock calibration

`xpool.observability` implements single-Host calibration over the version-four
paired samples using deterministic calibration algorithm version 1. This section
owns the implemented numerical contract; the independent
[reference vectors](../plans/unified-timeline-observability/fixtures/README.md) freeze its
expected results. Calibration
does not install a sampler, synchronize CUDA, modify artifacts or infer protocol
relationships. Runtime sample visibility and Native/Python CLOCK_MONOTONIC
consistency require separate measurement evidence.

### Declarations and model ownership

`build_calibration(view, declarations)` consumes a verified `Reconstruction` and
explicit `SamplerDeclaration` values. A sampler identifies its Run, Host Producer,
Device clock domain and raw Epoch. Registration resolves process creation,
physical Device UUID and the common Host reference. Processes on the same
physical Device remain separate raw domains. Retained samplers require
declarations; Device event domains without samples remain uncalibrated.

Declarations carry the attempted range, optional Device observation interval,
load-phase attempt ranges, provenance and optional measured sampler diagnostics.
Missing declarations of attempt ranges produce `attempt_range_unknown`, never a
guessed closing sequence. An explicit complete range with absent numbered
attempts produces `sampling_gap`; an incomplete fixed block produces
`incomplete_block`. Process-wide quality counters retain their original scope
and never become inferred per-Epoch totals. Busy scheduling skips are distinct
from attempted samples. Constructors validate field types, unsigned integer
bounds and closed values through `Pydantic ValidationError`. Declaration
constructors retain cross-field inputs until `build_calibration` checks them with
sampler context: reversed attempt/observation/phase ranges, overlapping phases,
phases outside declared attempts, and failed counts above attempted counts
raise `CalibrationError` with code `invalid_sample` and any retained sample sources.
Retained attempts and failures exceeding known diagnostic totals also fail.
This check runs before registration resolution or model construction.

`CalibrationIndex` retains immutable model/declaration/domain inventories and
domain lookup tables. `SampleBlock` reuses `PairedClockSample`, retaining all
known source locations and the original declared scope. Duplicate attempts count
once in distributions while their source evidence remains available. Invalid
samples, contradictory attempts and conflicting declarations raise
`CalibrationError`/`ObservationError` with stable codes and source context;
batch calls fail immediately. Normal evidence limitations return unavailable
models rather than exceptions.

`build_calibration` partitions attempted sequences into fixed blocks; `calibrate`
fits one `SampleBlock`. A `PairedClockSample` retains its Run, Host Producer,
Device domain, Epoch, attempt number, outcome and source. Raw time values and
counts use unsigned 64-bit bounds; IDs and attempts are positive. Versions,
failure codes, mapping assumptions and reason fields have closed public types.
Mathematical coefficients, mapped bounds and signed duration differences
retain arbitrary precision.

Successful samples require `H_before <= H_after` and present Device/Host-after
readouts. Failed samples retain null Device/Host-after readouts and an explicit
`SampleFailureCode`. Invalid outcome-dependent fields raise `invalid_sample`
before fitting; they are never converted into unavailable model evidence.

### Blocks and feasible region

Block `k` contains attempt sequences `10*k+1` through `10*k+10`, for k >= 0.
Positions 4 and 9 are validation; positions 1,2,3,5,6,7,8,10 fit the model.
Blocks are fixed by attempt number, not by arrival order or only successful
samples. A partial final block is `incomplete_block`. No sliding windows,
outlier deletion, automatic refitting, or fallback midpoint regression exists
in v4. The default interval is 100 ms; actual sample spacing remains evidence.

Use local coordinates x = G - G0, with G0 the first fitting sample's G, and
h = H_before(first fitting sample). Parameterize:

```text
H(G) = a * (G - G0) + c + h
H_before_i - h <= a * x_i + c <= H_after_i - h
a >= 0
```

The closed region uses a >= 0 to compute conservative extrema; admissible
models require a > 0. Require at least two distinct fitting G values, then
enumerate intersections of all pairs of boundary lines. Retain intersections
that satisfy every half-plane exactly, deduplicate exact rational vertices,
and sort lexicographically by (a,c). This defines P, including line/point
degeneracies. With finite windows and distinct fitting G values, a nonempty P
is bounded. An empty P or maximum feasible a <= 0 is `model_conflict`.
Distinct fitting G values must span the validation G values; otherwise mark
`validation_outside_support` and provide no model.

Arithmetic is exact rational arithmetic on original integer observations.
No epsilon, floating-point least squares, or magic slope threshold is used.
For rational p/q, q is positive, gcd(abs(p),q)=1, and zero is 0/1. JSON stores
numerator and denominator as decimal strings so rational parameters have no
fixed integer-width truncation. Public shape is
`{"numerator":"...","denominator":"..."}`.

Let V be the distinct vertices. Select a_star = mean(a over V) and
c_star = mean(c over V). A point is its own mean; a line uses its endpoints.
The mean has positive slope whenever P admits a positive slope. Report the
equivalent intercept b_star = h + c_star - a_star*G0. Keep the full vertices,
origins, source fitting sequences, and algorithm_version=1.

### Held-out validation and validity

For each held-out sample, test exactly:

```text
H_before <= a_star * (G - G0) + c_star + h <= H_after
```

Both must pass; otherwise the whole block is `validation_failed`. Validation
does not tighten P, choose a different vertex, or split/refit the failed block.
The model remains conditional on an affine clock relationship over its segment;
this check is model consistency, not independent hardware timing truth.

The inclusive validity interval is [min(G_fit), max(G_fit)]. Raw events outside
it are `outside_support`, even if close. Preserve the uncovered interval
between adjacent blocks. For overlapping valid intervals, model ID is
`(sampler, block_index)`; require exactly one applicable block or return
`ambiguous_model` rather than silently selecting a model. No averaging or
interpolation across models/Epochs is allowed.

Unavailable-block precedence is: invalid input/sample conflict (error),
attempt_range_unknown, incomplete_block, sampling_gap, sample_failed, clock_jump, insufficient_span,
model_conflict, validation_outside_support, validation_failed. A complete block
with a missing numbered attempt is sampling_gap, even when its records arrive
unordered. Capture all contributing reasons as diagnostics; precedence selects
the single public status. Incomplete means the declared attempt range starts
after the block's first position or ends before its tenth position; it is
distinct from missing records in a closed full attempt range.

### Mapping and duration outputs

For a supported event G:

```text
estimate_exact = a_star*(G-G0) + c_star + h
lower_exact = min(a*(G-G0) + c + h for (a,c) in V)
upper_exact = max(a*(G-G0) + c + h for (a,c) in V)
estimate_ns = round_to_nearest_integer_ties_to_even(estimate_exact)
lower_ns = floor(lower_exact)
upper_ns = ceil(upper_exact)
```

Return status, model ID, Epoch, original G/domain, exact estimate/lower/upper,
integer estimate/lower/upper, and `assumption=segment_affine_bracketing`.
Do not serialize non-finite numbers. Missing calibration returns null for all
mapped values and a structured reason. Host timestamps have a degenerate
identity mapping to the common Host reference; Host readout overhead is already
part of sampling windows, not a hidden correction.

For end minus start, propagate without assuming statistical independence:

```text
estimate_exact = estimate_end - estimate_start
lower_exact = lower_end - upper_start
upper_exact = upper_end - lower_start
```

Round the exact duration estimate to the nearest integer with ties to even;
floor its lower bound and ceil its upper bound after the subtraction. This conservative rule
also applies to two endpoints of the same model; it intentionally does not
exploit parameter covariance. If either mapping is unavailable, duration is
unavailable. Ordering is `after` only when lower_exact > 0, `before` only when
upper_exact < 0, `simultaneous` when both are zero, and `indeterminate` otherwise.
Never clamp a negative estimate or bound to zero. A protocol causal edge is
retained independently of this timing result.

Rate reporting uses `(a_star - 1) * 10^6` ppm with corresponding min/max feasible
slopes. Differences between block slopes or Device mappings use interval
subtraction; if zero lies in the difference interval, drift is unresolved.
Sample window widths, validation results, coverage intervals/gaps, attempted
and failed samples, skipped scheduling opportunities, source load phase, and
sampling overhead enter reports. Residuals are not true error; midpoint window
half-width is not an all-event error guarantee. No fixed accuracy or performance
threshold is introduced.

A decreasing Device sample invalidates its block and subsequent blocks of that
Epoch; recovery requires an explicitly registered new Epoch/domain.
map_time queries one model; index.map_raw and map_event require exactly one
supported model. map_event verifies clock/source registration and retains the
queried Event's source. Host timestamps map identically to their common
CLOCK_MONOTONIC reference with assumption=host_identity. Runs and known Host
references must match for duration and drift queries.

### Reports and coverage

`build_calibration_report(index, view)` produces an immutable `CalibrationReport`.
`to_json()` serializes deterministically with exact decimal-string rational
numerators/denominators. Reports retain original samples, model vertices,
parameters, support, validation and provenance; they stream retained-event
statistics without storing every event mapping again. There is no calibration
CLI or implicit output-directory discovery.

Reports expose Host window and sampling-interval distributions, load phases,
caller-measured overhead, mapping uncertainty and unmodified quality evidence.
Missing overhead and scoped counters remain null. Drift summaries compare
adjacent valid sampler blocks and cross-domain blocks with overlapping
conservative Host support envelopes; arbitrary pairs use `estimate_drift`.

Each Device domain has two separate coverage denominators:

- Time coverage measures the continuous duration with exactly one valid mapping
  inside the caller's explicit raw-Device observation interval. Missing or
  zero-duration denominators yield null ratios. Support and gap spans remain
  visible; endpoint-only overlap is represented explicitly and has zero duration.
  For observation [L,U], divide the total lengths of intersections with
  exactly-one-model spans by U-L; inclusive endpoints do not add duration.
- Retained-event coverage counts uniquely mapped canonical Device events from
  the supplied reconstruction. Zero retained events yield a null ratio. This
  metric makes no statement about lost events or unobserved runtime. The ratio
  is uniquely mapped canonical Device events divided by all retained canonical
  Device events in that domain, including unavailable mappings in the denominator.

Window consistency and conditional affine bounds are not independent hardware
ground truth. Synthetic vectors establish implementation agreement; actual idle,
serving, stress, MPS and multiprocess precision, coverage, drift and sampling
overhead require returned real execution evidence. Reports introduce no fixed
precision or overhead acceptance threshold.

### Calibration example

The IDs and measured ranges below must come from the caller's verified inventory.
They are not inferred from the largest retained event or sample sequence.

```python
from xpool.observability import (
    LoadPhase,
    LoadPhaseRange,
    Sampler,
    SamplerDeclaration,
    build_calibration,
    build_calibration_report,
    map_event,
)

declaration = SamplerDeclaration(
    sampler=Sampler(
        run_id=run_id,
        host_producer=host_producer,
        device_clock_domain_id=device_domain,
        raw_epoch=epoch,
    ),
    declared_attempt_range=(1, 20),
    observation_interval=(device_observation_start, device_observation_end),
    load_phases=(LoadPhaseRange(first=1, last=20, phase=LoadPhase.SERVING),),
    provenance=("caller-owned run inventory",),
)
index = build_calibration(view, (declaration,))
mapped = map_event(index, event)
report_json = build_calibration_report(index, view).to_json()
# The caller explicitly selects where to write report_json; raw files stay intact.
```
