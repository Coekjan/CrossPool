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
reads and Host-after values stay absent. Clock fitting and Perfetto export are
separate consumers of this evidence.

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
