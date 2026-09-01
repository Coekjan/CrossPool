# Devkit

Devkit observes real production execution through typed Hook Points. It does
not replace Transport, Fabric, or FFN computation.

Debug configuration is process-global and immutable after native
initialization. Native owners read the installed value directly rather than
copying Debug policy into runtime state, adapters, or Kernel arguments. The
configuration contains only Transport, Fabric, Graph, and FFN Routing Observer
options.

Each Observer records evidence from the real execution path and owns
process-local storage. Observer state is never stored in a Transport or Fabric
arena. Fabric and FFN Routing Observer allocations are included in FFN memory
admission through their native sizing queries; Transport Observer storage
remains outside FFN admission until attention-side memory planning exists.

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

A production call site names only its owning Point and does not branch on a
concrete Observer. Host and Device adapters are explicit compile-visible
catalogs. There is no runtime registry, priority, dynamic plugin ABI, handler
storage, interception, mutation, fallback, or alternate execution.

Removing an Observe call leaves the surrounding execution readable and
complete. Collective Device Points are reached by every participating CTA
thread; adapters that need only one observation select thread zero themselves,
while an Observer that copies a collective payload may use the complete CTA.

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
Observers never replace execution, provide readiness, or mutate Graph topology.
