# Control Plane

This document defines CrossPool configuration, integration, generation planning,
placement, memory admission, and participant lifecycle. See the
[system overview](overview.md) for process roles and supported deployment
boundaries.

## Configuration and integration

Runtime configuration flows through `xpool.config`. Entry points install one
process-global configuration and business logic reads that configuration rather
than caching selected values elsewhere. CLI values override allowlisted
environment variables, which override TOML, which overrides registry defaults.
Each setting declares which of these sources it accepts.
Model paths are resolved from the configured model entries and vendor model
base.

Required settings without defaults fail fast. Deployment settings live in TOML
with selected CLI and environment overrides; `.env` supplies process environment
settings such as `XPOOL_CONFIG` and `SGLANG_PLUGINS` and optional diagnostics.
Every accepted `XPOOL_*` variable is declared in the config registry, and unknown
names produce a warning. Debug settings have nested in-process names such as
`debug.graph_observer.enable`; their registered environment variables and
defaults are the accepted sources. For example, Graph Observer uses
`XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE` and
`XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR`.

`ModelConfig.path` is the schema's explicit path override. Machine-local paths
belong in ignored `*.local.toml` files.

Every process receives the same effective configuration. The daemon compares
declared process identity and topology against that configuration during
registration; processes do not negotiate independent settings.

Role-owned placement is declared through the required `atn.devices` and
`ffn.devices` lists. Each list is nonempty, nonnegative, unique, and ascending;
its order determines the corresponding Agent ranks, and one CUDA device cannot
belong to both roles. FFN also owns its optional device-memory calibration,
explicit operator margin, checkpoint-loader policy, and placement-solver
policy. Loader parallelism lives under `ffn.loader`; solver parallelism and its
whole-solve deadline live directly under `ffn.placement`.

`atn.device_memory_utilization` defines the maximum share of each attention
GPU that the post-capture Elastic KV Capacity Pool may retain. The daemon
applies it to the AtnAgent's observed total memory after accounting for
non-KV allocations and already mapped bootstrap backing.

`scheduler.slo` supplies required positive, finite `ttft_ms` and `tbt_ms`
targets for Elastic KV arbitration. A model may replace both targets with a
complete `models[].slo` value; overrides must be complete and request-level
SLOs are outside this interface. These targets use scheduler-local Prefill and
Decode timing rather than client-observed HTTP latency. The pinned SGLang
scheduler provides timing observations and request priority, but no typed
per-request TTFT/TBT objective. Priority remains an ordering hint. See
[Elastic KV Cache Pooling](elastic-kv-cache.md) for the demand and deadline
contract.

`scheduler.atn_concurrency` is retained as an explicitly reserved attention-side
compute-admission budget. The current runtime ignores it; future attention
admission will define its owner and resource unit before making it operational.

The core runtime contains engine-neutral model, topology, transport, execution,
and failure values. Serving-engine runtime imports, hooks, objects, and
compatibility behavior remain under `xpool.integrations.sglang`. The integration
translates SGLang state into CrossPool-owned values before crossing the core seam.
One explicitly bounded implementation exception permits the FfnAgent operator
module to use the pinned SGLang distribution's low-level Expert kernel and
configuration-selection modules. Core APIs, Plans, Registries, GraphTemplates,
and native Projections retain CrossPool-owned values only.

The SGLang adapter replaces supported decoder FFN modules with a shim module,
filters their FFN tensors from attention-side loading, and preserves the model's
attention-side behavior. Model adapters are discovered automatically and are
split by model family. A model adapter receives a strict `FfnSourceConfig` view
of the parsed `config.json` object and owns architecture extraction, checkpoint
key mapping, activation selection, routing function selection, and reference
binding for that family. The resulting `FfnModelSpec.digest()` identifies the
compiled FFN semantics; the source JSON bytes are not carried as a separate
contract field.

Outer graph mode is an attention-side concept. Eager, Decode Full, and Prefill
Breakable graph modes all invoke the same FFN data-plane protocol.
FfnAgent execution is always graph-backed; SGLang owns the outer graph mode.

Fabric Executable is the earlier data-plane barrier that permits Instance
Ranks to attach Transport while their serving schedulers are still starting.
An Instance Rank publishes initialization from SGLang's scheduler handshake
after scheduler construction completes. That publication includes an immutable
Serving Listener shared by every rank of the Instance; a conflicting rank
publication is rejected atomically. Daemon System Ready requires every expected
publication in addition to executable Fabric, healthy processes, Transport,
MPS, and failure-free generation state.

After System Ready, the daemon concurrently probes the public HTTP `/health`
endpoint of every configured Instance. Successful listeners remain satisfied
while their peers finish starting. The daemon confirms and logs Serving Healthy
once only when the generation and complete ordered listener snapshot still
match. Wildcard bind hosts are normalized to loopback only for local probes;
published and logged listener values remain unchanged. Serving Healthy is a
one-time startup observation rather than a readiness phase or continuous
availability monitor. The integration supports this boundary for plain HTTP
serving and rejects gRPC-only and TLS serving modes.

## Operational logging

Every process configures the process-local `xpool` logger after resolving the
global configuration. Runtime records use the configured level and
terminal-aware color policy, write to stderr, and do not replace the root,
Uvicorn, or SGLang logging policy. CLI result data remains on stdout.

Runtime logs describe low-frequency lifecycle transitions, completed slow
startup phases, and recoverable communication state edges. Observer records
remain authoritative for per-request, protocol, routing, and Graph evidence.
The layer that terminates an operation owns its failure log. The daemon logs the
first entry and final clearance of each global warning aggregated by
`(kind, device)`; heartbeat clients do not duplicate unchanged warning state.

## Generation planning

One daemon-authored `FabricPlan` describes a stopped-world Fabric generation.
It contains global participant counts, executor lane count, scheduler policy,
ordered model plans, and ordered Instance plans. Native processes create live
CUDA objects and process-local addresses after retaining the plan.

`FfnModelSpec` is the model-source contract. It contains ordered gated Dense or
MoE layer semantics, intrinsic dimensions, activation and routing behavior,
and checkpoint identities. Generation planning resolves placement and device
state separately.

`InstanceFfnProfile` is the serving Instance declaration. It contains the
runtime payload dtype, hidden size, ordered layer projection, and decode and
prefill row capacities observed at the shim boundary. Python represents the
payload element type with `torch.dtype`. Its control-plane JSON field uses the
canonical unqualified Torch dtype name and restores any dtype exposed by the
installed Torch build. `InstanceFfnProfile` alone owns this lossless wire
mapping. Serving integration preparation and native execution boundaries
decide whether the current FFN implementation can execute the dtype. In-process
consumers see only `torch.dtype`; dtype-name conversion remains local to the
profile boundary.

`FfnModelPlan` is the generation-static FFN realization. It assigns one
execution group to every layer and retains only semantics needed to materialize
that layer. Every selected layer group has true TP width equal to its number of
FfnAgent members.

`FabricInstancePlan` pairs one Instance profile with its attention topology and
result-delivery requirements. Model and Instance plans are co-indexed by
`instance_index`; request lookup uses `(instance_index, layer_ordinal)`.

Elastic KV memory has a separate control seam. Instance registrations carry
immutable, model-derived partition geometry. Live capacity is coordinated by
the separate Elastic KV control seam.
One Generation-scoped daemon policy freezes each attention GPU's physical pool
after Graph capture and coordinates persistent quantified demand, immutable
group capacity operations, TP readiness votes, and terminal partition completions
through a host-local native channel. SGLang retains logical allocation and
prefix-cache ownership. See [Elastic KV Cache Pooling](elastic-kv-cache.md).

`xpool::fabric::ArenaProjection` is the minimal native join projection derived
from the plan. Native layout code derives byte geometry, offsets, and local
views from that projection and the ABI. Debug capacities belong to the owning
Devkit adapter, which allocates its process-local storage directly.

## Placement

Placement is computed once before the generation starts. Every layer receives
one FfnAgent execution group; its TP weights stay within that group.
The optimizer respects device memory admission, TP width, group topology, and
model-layer order.

The primary objective minimizes the number of distinct device groups used by
one model so layers can reuse GraphTemplates and reduce startup cost. Secondary
objectives balance admitted bytes and avoid unnecessary fragmentation. The
solver is exact for the accepted formulation rather than a beam-search
heuristic. Its parallelism and timeout are configuration values.

After the optimization objectives are fixed, equal-optimum placements use one
deterministic lowest-index-first representative. The final fixed search visits
`assignment_count` and `primary_member` in coordinate order and selects their
maximum feasible values, so earlier FfnAgent indices win only when all admitted
objectives are already equal. This is a canonical tie-break, not a performance
cost model; it adds no runtime configuration or topology heuristic.

Only cross-model concurrency is admitted in the current target. Requests for
one Instance are serialized across its layers. Executor lanes allow independent
Instances to overlap without creating weight replicas. `ffn_concurrency` is the
generation's structural Executor Lane count: it bounds simultaneous Invocations
and determines preallocated GraphExec, stream, workspace, payload, and protocol
state. It is not an active-row or compute-load budget. Any workload-aware
admission limit would be an independent scheduler dimension and requires a new
accepted design. A controlled throwaway Capacity sweep rejected
`payload_row_capacity` sum as that dimension: slowdown was nonmonotonic in the
sum, equal sums behaved differently across model compositions, and the smallest
paired Capacity produced the largest short-MoE slowdown. Capacity remains a
GraphTemplate shape boundary, not a resource-consumption estimate or admission
charge. The production scheduler therefore admits by available Executor Lane
only.

## Memory admission

Admission estimates every startup stage as the Exact Resource Ledger plus the
Analytic Allocator Allowance and, when a compatible Profile is selected, the
Calibrated Overhead Envelope. The peak across those stages is the predicted
peak. An explicit operator device-memory margin is added once after that peak;
it is not part of the estimator and cannot repair estimator underprediction.

The analytic estimator works with no prior profiling. An optional calibration
file supplies a device-local correction learned from a synthetic, model-neutral
corpus. With no `ffn.device_memory_calibration` path configured, admission uses
analytic estimation alone. An explicitly configured profile must be readable,
valid, and compatible with the deployment; startup reports a profile error
when those conditions fail. The compatibility checks compare recorded
software, configuration, MPS, and per-FfnAgent GPU evidence. They constrain
profile reuse, not the set of GPU models on which CrossPool may run.

The `xpool memory-profile` command produces calibration evidence. A profile
records local GPU and software identity, fitted coefficients, observed and
predicted allocation values, and the Calibrated Overhead Envelope. The fitter
adds one empirically derived Device Observation Quantum to each grouped
residual target and absorbs that correction into the fitted coefficients. The
quantum belongs to the fitting procedure; it is separate from Profile fields
and the operator margin. A Profile records allocation behavior and remains
model-neutral.

Checkpoint files may be read concurrently. Host buffers may use pinned memory,
and independent tensor copies may use multiple CUDA streams. Device-side
materialization remains bounded by the admitted plan and must not retain
construction-only buffers after installation.

## Startup and shutdown

Startup is monotonic:

1. every process resolves configuration and registers its role capabilities;
2. the daemon admits and retains the Fabric Plan;
3. during `PREPARING_JOIN`, AtnAgents publish Transport arenas while FfnAgents
   perform memory admission and materialize selected weight shards;
4. participants join the admitted Fabric generation collectively;
5. during `PREPARING_EXECUTION`, FfnAgents capture and install execution;
6. activation makes the Fabric generation executable;
7. Instance ranks observe the executable barrier, attach their Transport
   arenas, and publish initialization readiness; and
8. the daemon reports ready only after all required owners are initialized.

An AtnAgent or FfnAgent may repeat registration only before retaining a Fabric
Plan. Registration loss after Plan acquisition is terminal because participants
cannot recover into a retained Generation. An Instance rank may retry temporary
daemon transport failures within its existing bounded deadline, but a daemon
response that its registration is missing is terminal; it does not re-register
or reacquire its Transport lease.

Readiness never derives from process existence alone. It requires live
registrations, MPS availability, a retained admitted plan, usable Transport
leases, initialized Fabric participants, installed real FFN execution, and no
canonical failure.

Shutdown is coordinated while CUDA and NVSHMEM runtimes remain live. AtnAgent
and FfnAgent participants drain asynchronous work, release rank-local
resources, destroy Transport arenas, and finalize Fabric collectively before
exiting. SGLang owns request draining and Instance process-tree cleanup. The
daemon treats an exited Instance rank as Generation owner loss and drives the
remaining participants through `QUIESCING -> DRAINING -> FINALIZING ->
STOPPED`; it does not retain a Generation after an Instance leaves. Explicit
Instance detach remains the startup-rollback and controlled-cleanup path, while
process loss relies on CUDA process teardown and the existing daemon watchdog.
Destructors are best-effort guards, not distributed recovery.
