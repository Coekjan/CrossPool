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
The registry, source records, path-origin handling and CLI override mechanics
live in [`xpool.utils.config`](../../src/xpool/utils/config.py). Production
`XpoolConfig` owns runtime fields and validation; the private development tools
reuse this mechanism without becoming production dependencies.
[`ModelId`](../../src/xpool/model.py) owns immutable, case-sensitive model
identity in strict `namespace/name` form. Components admit ASCII letters,
digits, underscore, hyphen and dot, excluding `.` and `..` components. Identity
preserves spelling, supports hashing and lexicographic ordering of the full
string, and owns its namespace, name and relative identity path. In-process
configuration, plans and tool declarations use this value; JSON/TOML and native
string boundaries retain scalar strings. The identity also owns complete-ID URI
component encoding, which preserves spelling and escapes the separating slash.
Model and Instance are distinct concepts. The supported deployment has one
configured Instance for each Model ID, so Instance configuration, runtime and
daemon lookups share that logical key. Configuration-ordered native Instance
indices, process-owner identities and Fabric generations retain their separate
meanings.

Instance wire records name the logical key `model_id` and serialize its full
scalar string. Instance operation URLs use `/instance/<namespace/name>/...`;
the route boundary parses a `ModelId` before daemon lookup.

`XpoolConfig.model_path_of()` resolves a Model ID through a matching configured
path override or the vendor model base plus the identity's relative path.
Portable deployments and tool catalogues select identities; the effective
runtime configuration owns the machine-local checkpoint location. Preflight
and launch resolve checkpoints through this owner.

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

`XpoolConfig.to_config_mapping()` serializes effective CONFIG-allowed values for
TOML through registry source permissions, including model wildcard fields.
Optional `None` values, bootstrap inputs and env-only debug settings are omitted;
debug inputs and original source provenance have separate owners. Reloading a
snapshot establishes CONFIG provenance rather than preserving its original
sources. [Tooling launch snapshots](tooling.md#shared-serving-lifecycle) preserve
complete effective configuration while binding owned resources and child inputs.

Every process receives the same effective configuration. Participants compare
configuration and their ordered Deployment Device View at the
[startup check](#managed-mps-and-role-preparation). Registration validates
declared process identity, topology and execution contracts against that
configuration; processes do not negotiate independent settings.

Role-owned placement is declared through the required `atn.devices` and
`ffn.devices` lists. Both are nonempty consecutive blocks: attention starts at
zero and FFN immediately follows attention. For attention width `A` and FFN
width `F`, the lists are `0..A-1` and `A..A+F-1`. These indices select positions
in the Deployment Device View, not host inventory indices. Each role's list
order determines its logical Agent ranks. Runtime configuration and portable
tool deployments share this validation contract.

FFN also owns its optional device-memory calibration, explicit operator margin,
checkpoint-loader policy, and placement-solver policy. Loader parallelism lives
under `ffn.loader`; solver parallelism and its whole-solve deadline live directly
under `ffn.placement`.

Each configured Instance covers the complete Attention World, whose size is
`len(atn.devices)`. A model's `atn_dp_size` defaults to one. An omitted
`atn_tp_size` resolves to World size divided by DP size; division must be exact.
For every model, resolved attention TP times DP must equal the World size.
Explicit TP is validated, never rewritten. A standalone `ModelConfig` retains
omitted TP until the complete `XpoolConfig` supplies the Fleet. Omitted FFN TP
resolves independently to the full FfnAgent Fleet.

SGLang's tensor-parallel launch argument counts the complete attention worker
World, not the per-DP-group attention TP width. DP greater than one requires
DP Attention. The integration validates engine geometry against configuration;
daemon registration and readiness require the complete configured rank set.

`atn.device_memory_utilization` defines the maximum share of each attention
device that the post-capture Elastic KV Capacity Pool may retain. Available KV
bytes are the AtnAgent's observed free bytes plus already mapped bootstrap
backing. The daemon subtracts the larger of the configured unused-memory
margin and the summed Attention Runtime Headroom declared by co-located
Instance Ranks; it does not add those reserves together.

When an SGLang Instance has no resolved `max_running_requests` value and Decode
Graph is enabled, the integration defaults request concurrency to the resolved
Decode Graph `max_bs`. Explicit and model-derived values remain authoritative;
eager Decode receives no graph-derived default. This keeps ordinary Decode
within captured coverage and avoids reserving eager-Decode activation memory
for an otherwise unused upstream concurrency default.

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

The supported SGLang path is non-expert-parallel: `ep_size` remains one and
`moe_a2a_backend` remains `none`. CrossPool rejects DeepEP and other expert-
parallel settings at this integration boundary. The uv-managed environment
therefore excludes `sgl-deep-ep` from the pinned SGLang dependency graph; this
is a resolver policy for the supported path, not support for arbitrary SGLang
expert-parallel configurations or non-uv installers.

The SGLang adapter replaces supported decoder FFN modules with a shim module,
filters their FFN tensors from attention-side loading, and preserves the model's
attention-side behavior. Model adapters are discovered automatically and are
split by model family. A model adapter receives a strict `FfnSourceConfig` view
of the parsed `config.json` object and owns architecture extraction, checkpoint
key mapping, activation selection, routing function selection, and reference
binding for that family. The resulting `FfnModelSpec.digest()` identifies the
compiled FFN semantics; the source JSON bytes are not carried as a separate
contract field.

Outer graph mode is an attention-side concept. Eager, Decode Full, and combined
Decode Full plus Prefill Breakable modes all invoke the same FFN data-plane
protocol.
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
Daemon startup records identify the Fabric Generation through the transition
into `executable`. Later serving, capacity, quiesce, drain, finalize, and stop
records omit that established context. Instance and Agent startup records name
completed arena attachment, GraphTemplate capture, and native installation
phases at their owning layer.

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
One Generation-scoped daemon policy freezes each attention device's physical pool
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
software, configuration and per-FfnAgent device evidence from direct FFN
execution. They constrain profile reuse, not the set of device models on which
CrossPool may run.

The `xpool memory-profile` command produces calibration evidence. A profile
records local device and software identity, fitted coefficients, observed and
predicted allocation values, and the Calibrated Overhead Envelope. The fitter
adds one empirically derived Device Observation Quantum to each grouped
residual target and absorbs that correction into the fitted coefficients. The
quantum belongs to the fitting procedure; it is separate from Profile fields
and the operator margin. A Profile records allocation behavior and remains
model-neutral. The profiler owns its local attention MPS scope and participants;
FFN allocation measurements use the same direct execution environment as serving.

Checkpoint files may be read concurrently. Host buffers may use pinned memory,
and independent tensor copies may use multiple CUDA streams. Device-side
materialization remains bounded by the admitted plan and must not retain
construction-only buffers after installation.

## Startup and shutdown

Startup is monotonic:

1. the daemon resolves physical placement and starts its attention MPS scope;
2. Agents normalize deployment visibility, check configuration and obtain
   startup admission before preparing MPS or direct execution. Serving owners
   launch commands with normalized visibility and the attention MPS endpoint;
   the plugin checks configuration before installing its hooks. Participants
   initialize their devices and register their runtime capabilities;
3. the daemon admits and retains the Fabric Plan;
4. during `PREPARING_JOIN`, AtnAgents publish Transport arenas while FfnAgents
   perform memory admission and materialize selected weight shards;
5. participants join the admitted Fabric generation collectively;
6. during `PREPARING_EXECUTION`, FfnAgents capture and install execution;
7. activation makes the Fabric generation executable;
8. Instance ranks observe the executable barrier, attach their Transport
   arenas, and publish initialization readiness; and
9. the daemon reports ready only after all required owners are initialized.

An AtnAgent or FfnAgent may automatically recover a missing registration only
before retaining a Fabric Plan. Registration loss after Plan acquisition is
terminal because participants cannot recover into a retained Generation.
At the daemon boundary, an identical repeated registration from the same
admitted live owner is idempotent, including while a Plan is retained, and
preserves its Transport resources. Replacement admission waits for actual
retirement of the old generation owners.

An Instance rank may retry temporary daemon transport failures within its
existing bounded deadline, but a daemon response that its registration is
missing is terminal; it does not re-register or reacquire its Transport lease.

Readiness never derives from process existence alone. It requires live
registrations, MPS availability, a retained admitted plan, usable Transport
leases, initialized Fabric participants, installed real FFN execution, and no
canonical failure.

### Managed MPS and role preparation

[`xpool.utils.mps.MpsEndpoint`](../../src/xpool/utils/mps.py) is an immutable
value for the ordered attention MPS UUID subset. It owns address derivation,
pipe/log environment preparation checks, management queries and actual client
inspection. Construction validates physical identities and creates no resources.
The value remains independent of configuration and controller ownership.

`MpsScope` composes that endpoint and owns one foreground
`nvidia-cuda-mps-control -f` process, its exact identity, observed server identities,
logs and private endpoint. The daemon retains that scope before starting it.
Server creation may be lazy; startup requires the controller, not a server PID.
The foreground mode is provided by the
[NVIDIA control interface](https://docs.nvidia.com/deploy/mps/595/appendix-tools-and-interface-reference.html#nvidia-cuda-mps-control).

The address is `/tmp/xpool-mps-<uid>/<key>/{pipe,log}`. The key is the first 32
hexadecimal characters of SHA-256 over newline-joined, sorted full attention
UUIDs. Sorting affects the address only; visibility preserves rank order.
Each user has an independent root directly under `/tmp`. Root, endpoint, pipe
and log directories are created with mode `0700`.
Exclusive directory creation establishes ownership. An existing scope is neither
adopted nor deleted. Cleanup removes only the directory whose identity the owner
retained, after its controller/server domain exits. This endpoint claim is not
a physical-device lock; operators coordinate independent deployments. Runtime
code changes neither compute mode nor privileged host policy.

[`xpool.utils.device`](../../src/xpool/utils/device.py) queries physical
index-to-UUID inventory directly through a bounded `nvidia-smi` command without
initializing the CUDA Driver or creating a context. Numeric selectors name the
returned `nvidia-smi` indices; explicit selection order is preserved. Unset
visibility selects all physical devices in ascending inventory-index order;
an explicitly empty value selects none. Full physical UUIDs are accepted;
unknown selectors, duplicate physical selections and MIG selectors are rejected.
The read-only visibility accessor caches successful resolutions by the raw
environment value, assuming stable inventory during one process lifetime.
Changing the selection chooses a different entry. Physical inventory and actual
MPS availability or membership observations remain uncached.

The daemon, both Agent entries and `xpool exec` normalize the complete ordered
Deployment Device View into `CUDA_VISIBLE_DEVICES`. Full UUIDs avoid
[MPS's server-visible numeric remapping](https://docs.nvidia.com/deploy/mps/595/appendix-tools-and-interface-reference.html#cuda-visible-devices).
Daemon and Agent startup require the configured layout to fit that view. The
daemon creates no device context; only its controller child receives
attention-only visibility.
`MpsEndpoint.environment()` prepares pipe/log addresses without changing
visibility. Attention clients retain the complete UUID environment, while MPS
exposes the zero-based attention prefix to the CUDA Driver. FFN retains the
complete view with `CUDA_MPS_PIPE_DIRECTORY=""` for direct execution and selects
its configured deployment device index. Its logical Fleet rank remains distinct
from that execution index.

`xpool exec -- PROGRAM ARG...` normalizes complete visibility and prepares the
attention MPS endpoint before the target imports its runtime, then replaces its
process image. Arguments, PID, inherited process group and standard streams are
preserved. Configuration comes from normal TOML/environment resolution; the
wrapper accepts target arguments rather than CrossPool CLI configuration flags.
It performs no daemon RPC or participant configuration check and leaves plugin
selection and engine-specific settings to their owners. Agent commands prepare
their own execution environment without this wrapper.

Manual SGLang launch selects `xpool` through `SGLANG_PLUGINS` and uses
`xpool exec -- sglang serve ...`. Tool-owned launch selects the plugin in its
child environment. During installation in the serving parent and workers, the
plugin checks pipe/log addresses and effective configuration plus the complete
ordered UUID view through `/config/check`. Agent startup performs the same check
before admission and native initialization. Matching requests return `204`,
disagreement returns `409`, and malformed requests return `422`. These startup
owners perform the complete check once; registration owns identity, topology and
execution contracts.

The plugin sets `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` to true and
`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` and
`SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION` to false through SGLang's environment API
before installing hooks. Launch preparation must precede engine import because
the pinned CLI can enumerate devices before plugin installation. After context
initialization, the ModelRunner load hook requires actual current-process MPS
membership before loading weights. An excluded plugin cannot verify its absence.

Agent startup admission retains exact identities under the daemon's user and PID
namespace and checks configured role/device placement. Instance registration
retains identity, ABI, topology and runtime contracts. Ordered physical placement
agreement belongs to the startup configuration check; address equality alone
does not prove rank order because address derivation sorts UUIDs. Environment
preparation, actual membership and process ownership remain separate facts.
Serving owners retain their actual workers and helpers.

### Ordered retirement

The serving owner initiates SGLang exit and retires workers and helpers. Tool-owned
execution first seals and joins startup, then closes every server process group
and signals the exact daemon PID. Manual deployments finish or cancel startup and
retire their clients before stopping the daemon, with new client launch sealed
throughout retirement.

The daemon closes participant admission and keeps control listeners and MPS
available. Normal shutdown waits for known Instance identities from registrations
and retained Fabric owners before requesting Agent/Fabric quiesce. Existing
owner-loss and failure paths retain their generation-failure behavior. Heartbeat
expiry or deregistration does not prove physical exit of retained generation
owners. Controller stop follows confirmed participant exit and actual MPS client
absence. Ordinary daemon SIGTERM/SIGINT does not signal serving processes.

An empty MPS client snapshot proves current attachment absence, not a fence
against future connections. Serving owners enforce launch/retirement ordering;
they also retain CPU-only startup and helper processes through retirement.
Controller/server domain retirement remains the scope's responsibility.

An Instance rank's exit retires its Generation rather than leaving it reusable.
The daemon drives the remaining participants through
`QUIESCING -> DRAINING -> FINALIZING -> STOPPED`. While CUDA and NVSHMEM remain
live, Agents drain asynchronous work, release rank-local resources and Transport
arenas, and finalize Fabric collectively. Explicit Instance detach serves
startup rollback and controlled cleanup; unexpected process loss enters the
watchdog's owner-loss path and is not itself verified retirement. Destructors
are best-effort guards, not distributed recovery.

An already-admitted Agent may finish initialization and formal registration
after admission closes. Without a joined generation, formal registration
establishes its Agent.run signal-handler boundary and permits exact-PID SIGTERM.
Startup admission alone does not. Joined Agents follow collective Fabric
retirement; missing peers do not authorize an unconditional finalize in `finally`.

SGLang retains its pinned factories and ready handshake. Startup signals record
cancellation without interrupting an in-progress initialization transaction.
At ready, pending cancellation uses the actual returned tokenizer transport and
worker/cache handles before HTTP startup proceeds. Initialized shutdown prefers
local release. IPC consumers retire before their cache exporters. Optional cache
helper lifecycle uses these same boundaries; checkpoint-loader and cache-mode
compatibility require their own serving evidence.

At destructive engine boundaries, the integration selects its retained worker
and cache-helper identities and stops further resource creation. The daemon's
termination operation targets actual clients of its retained MPS scope under
the same user and PID namespace. The scope checks exact target, controller and
server identities, serializes with controller stop, waits for `terminate_client`
and requires NVIDIA's `CUDA_SUCCESS` result for every affected client before
forced host exit begins. The utility's exit code alone proves command transport.
This follows
[NVIDIA client termination](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html#client-early-termination).
Context termination does not kill a host process, finalize Fabric, retire direct
FFN participants or establish complete-domain exit. These remain owner duties.
Target and scope checks validate the operation; caller authentication and
multi-tenant authorization are outside the supported local deployment boundary.

Controller startup, management queries and initial membership inspection reuse
one selected 30-second bound, capped by an existing owner deadline. Startup
admission RPCs use this bound without changing ordinary HTTP timeouts. Retirement
has a 300-second budget from its first trigger; repeated requests never renew it.
An initialized serving owner reserves the final 30 seconds for controlled
termination and reaping. The existing engine fallback may reach this boundary
earlier. Startup cancellation consumes that same first-trigger budget even while
waiting for ready. Agent/Fabric retirement has no blind forced-exit fallback.

If retirement cannot be confirmed because of failed factory readiness, uncertain
ownership, failed context termination or budget expiry, the actual living owner,
controller and diagnostics remain available for manual resolution. Observation
errors mean unknown, not process absence.
The daemon and tooling seal new admission rather than replacing the controller
or reclaiming resources from an unproved domain. Automatic recovery after owner
loss or a partial-world/fatal failure is outside this contract.

| Managed daemon exit | Meaning |
| --- | --- |
| `0` | Normal deployment exit with verified resource retirement. |
| `20` | Deployment failed, but resource retirement was verified. |
| Other status or signal death | Cleanup unconfirmed; outer ownership remains. |

These statuses apply to the retained daemon after its MPS scope closes. The
enclosing task still proves complete-domain exit and preserves its original
test or benchmark verdict. `XpoolClient.close()` closes only its HTTP client.
