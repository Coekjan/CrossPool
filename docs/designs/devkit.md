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
