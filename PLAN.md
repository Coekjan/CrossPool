# xpool Project Plan

This document is the canonical design for the xpool rebuild. It is
self-contained and replaces chat history or the v2 worktree as the source of
truth for new implementation work.

## Mission

xpool implements intra-node colocated serving with the smallest practical
intrusion into SGLang. SGLang remains the attention-side serving program and
request scheduler. xpool replaces FFN calls with a graph-safe shim, arbitrates
shared GPU and communication-slot resources across multiple SGLang instances,
and owns the FFN transport/execution path.

The first production target is `deepseek-ai/DeepSeek-V2-Lite-Chat`. The project
is SGLang-only.

## Non-Goals

- Do not restore the v2 source architecture wholesale.
- Do not use SGLang to execute FFN work.
- Do not implement expert parallelism in the first design; FFN execution uses
  tensor parallelism.
- Do not make KV cache sharing part of the first runnable FFN-shim closure.
- Do not add Python NVSHMEM bindings unless a later accepted design requires
  Python-side NVSHMEM calls.

## Runtime Model

xpool currently has three runtime placements:

1. `xpool daemon` is a single global host control plane. It owns registration,
   policy configuration, readiness, health reporting, and global state for
   attention-side SGLang instance arbitration. It must not participate in
   request-time FFN progress or captured CUDA graph execution. Registration
   readiness is process-liveness aware: stale SGLang instance or agent
   pids do not satisfy `/ready`, and a restarted participant may replace a dead
   registration without restarting the daemon.
2. `xpool atnagent` is launched once per configured attention GPU. Each
   AtnAgent owns the rank-local CUDA IPC ingress/egress arenas and the local
   transport progress runtime.
3. SGLang instances are normal SGLang server processes. Each instance loads the
   xpool SGLang plugin from `xpool.integrations.sglang`. The plugin is
   model-neutral: public adapter contracts live in
   `xpool.integrations.sglang.adapter`, automatic adapter discovery lives in
   `xpool.integrations.sglang.registry`, and model-specific implementations
   live under `xpool.integrations.sglang.models`. The plugin loads the registry,
   registers adapter hooks, runs model lifecycle checks around
   `ModelRunner.load_model`, and starts transport after
   `ModelRunner.init_memory_pool` resolves request concurrency. Model-specific hook targets, construction
   compatibility, weight filtering, and post-load invariants live inside the
   owning adapter. For the first DeepSeek adapter, SGLang's original
   `ForCausalLM`, model body, decoder layer, attention module, communicator,
   logits, and attention weight-loading logic remain the program skeleton.

xpool requires a responsive CUDA MPS control daemon before transport execution.
`/health` reports only daemon process liveness. `/ready` reports MPS status and
accepts the current `atn` scope, which is also the default. The final result is
the logical AND of MPS readiness and ATN readiness. ATN readiness requires the
configured AtnAgents and instance ranks to be online and every rank to have a
matching published transport arena. FfnAgent readiness is deferred with the
rest of the FfnAgent control plane.

## Shim Component

The shim component is not a single Python object. It is a coordinated ABI and
runtime path with two placements:

- **Shim frontend** lives inside each SGLang instance. It is installed by the
  SGLang plugin through a model adapter, replaces target FFN/MLP/MoE classes
  during model construction, and calls a CUDA extension op. The frontend writes
  a fixed-address request descriptor into IPC-mapped device memory and waits for
  completion/result state. The DeepSeek adapter's FFN shim classes inherit the
  original SGLang FFN classes for `isinstance` compatibility but do not call the
  original FFN constructors, so dense FFN weights, MoE experts, and shared
  experts are never allocated in the attention-side SGLang process.
- **Shim agent** lives in each local AtnAgent. It is a
  persistent kernel that polls rank-local queues, arbitrates resources,
  advances attention-to-FFN transport, and writes grants/results.

Both eager execution and CUDA graph capture/replay use the same device ABI.
Captured execution must not allocate memory, call host control-plane APIs,
perform Python control flow, or change descriptor addresses.

CUDA graph shape compatibility is derived from SGLang's actual graph capture
configuration and shim descriptors. xpool does not maintain an independent
decode-bucket list in repository config.

The first shim contract is deliberately narrow. A shim call represents one
decoder-layer FFN over a contiguous two-dimensional CUDA tensor
`[num_tokens, hidden_size]` and returns an out-of-place tensor with the same
shape and dtype. The first supported forward modes are normal SGLang decode,
extend/prefill, and data-parallel idle-rank calls with zero live work. The
SGLang plugin owns a model-neutral server-argument support
matrix and fails closed for SGLang runtime modes that are not yet represented in
the xpool shim ABI, including speculative execution, pipeline parallelism,
LoRA, quantized first-stage loading, expert parallelism, EPLB, DeepEP, expert
distribution recording, two-batch overlap, context parallel prefill, all-reduce
fusion, SGLang CPU/layer offload, hierarchical cache offload, decode KV offload,
mixed chunked prefill, PD disaggregation, diffusion LLM inference,
PD multiplexing, reduce-scatter FFN output, and SGLang DP Attention. DP
Attention remains a target capability, but the current integration rejects it
until the native transport can return the reduce-scattered FFN output SGLang
expects. Ordinary continuous batching and ordinary chunked prefill remain
allowed when SGLang presents shim calls as exact `DECODE`, exact `EXTEND`, or
exact `IDLE`; xpool rejects modes that can surface composite forward modes such
as `MIXED`,
`TARGET_VERIFY`, `DRAFT_EXTEND`, `DRAFT_EXTEND_V2`, `SPLIT_PREFILL`,
`DLLM_EXTEND`, or `PREBUILT`. SGLang server-argument rules intentionally
reference SGLang's resolved fields directly; an SGLang upgrade that renames or
changes those fields is an explicit adapter maintenance point.
Model adapters may add model-construction checks, but they do not own global
SGLang runtime policy. Server-argument rules remain part of the SGLang plugin
layer because they describe global plugin ABI support, not model architecture.
These rules run on SGLang's resolved `ServerArgs`; they are not a replacement
for SGLang's own CLI parsing, defaults, and server-argument validation.

The SGLang shim frontend uses a single concrete `FfnShimModule` base class,
not a mixin/protocol pair. Model-specific classes inherit
`FfnShimModule, OriginalSGLangClass`, and `FfnShimModule.__init__` directly
initializes `nn.Module` to avoid running original FFN constructors. Shim layer
kinds use symmetric names: `DENSE` and `SPARSE`. The shim's model architecture
string is diagnostic metadata injected from the actual SGLang/Hugging Face
model config after load; native routing uses the integer model index, not a
hard-coded architecture string.

## Resource Policy

The daemon configures policy, but the attention-side shim agent performs
the request-time grant on device.

The initial policy is intentionally conservative:

- For each attention GPU, only one SGLang instance may execute attention at a
  time.
- For each attention GPU, only one SGLang instance may own the communication
  slot at a time.
- Communication slots are explicit resources used to bound parallelism and
  enable pipeline scheduling.

## FFN Execution

FfnAgents own xpool FFN execution. The planned execution stack is:

- persistent-kernel scheduler,
- captured graph execution and lowering,
- declared host fallback for unsupported shapes,
- FlashInfer-backed kernels where available,
- tensor parallelism across participating FfnAgents.

Correctness is first proven against layer-level oracles before serving metrics
are claimed.

The next execution milestones are ordered and independently gated. First,
`debug.loopback.site=ffnagent` must prove the complete device-side path from
the instance CUDA IPC arena through the local AtnAgent, over NVSHMEM to an
FfnAgent, through the pairwise-rotation debug executor, and back through
the AtnAgent to the instance. A local FFN kernel or one-way delivery is not
sufficient evidence. Second, the DeepSeek-V2-Lite executor must replace the
debug rotation when loopback is disabled and prove dense, routed-expert,
shared-expert, graph, eager, and FFN-TP correctness against layer-level oracles. Debug
loopback never acts as a fallback for unsupported production executor shapes.

## NVSHMEM Transport

Every participating GPU owns one NVSHMEM rank through its agent. The
daemon does not own an NVSHMEM rank. SGLang instances do not initialize NVSHMEM.

NVSHMEM rank assignment is deterministic from configured device order: ATN
devices precede FFN devices. The daemon brokers opaque bootstrap metadata and
generation membership but never initializes NVSHMEM or enters the data plane.
All participants in one generation complete rendezvous before symmetric arenas
and persistent kernels become ready. Replacing any member invalidates the old
generation; old and new symmetric heaps must never be mixed. Shutdown stops
admission, resolves in-flight slots, stops persistent kernels, finalizes the
collective runtime, and only then removes control-plane ownership.

The transport uses NVSHMEM symmetric memory and device-side signal/put/get or
collective primitives between agents. Each SGLang shim communicates with its
local AtnAgent through CUDA IPC-mapped queues and descriptors; all
other attention and FfnAgents participate only through device-side channels
owned by the agent runtime.

Agent metadata is the daemon's live transport table, not a readiness bit and
not request-time routing state. An AtnAgent publishes the transport
arenas it owns after registering with the daemon and after instance ranks declare
their transport requirements. The daemon validates those arenas against static
config-derived placement and the registered rank requirements, stores them with
the agent's liveness, and returns only filtered per-instance-rank metadata to
SGLang. The instance-side key is `(instance_id, rank)`; the local attention CUDA
device is derived from `devices.atn_cuda_devices[rank]`, not repeated in the
instance response. The daemon brokers only the lowercase CUDA IPC arena handle.
Arena geometry is native-owned: the agent writes a `TransportArenaLayout`
header at the beginning of the CUDA IPC allocation, and the instance runtime
maps the handle and reads that header before installing the arena.

### Control-Plane Ownership And Recovery

The daemon is a trusted same-host broker and binds only to loopback addresses.
PID and ABI fields are ownership proofs inside that trust boundary, not remote
authentication. Registration and transport transactions snapshot registry
state under locks, perform process-liveness probes without locks, then reacquire
the locks and verify object identity before committing.

Transport arena publications are daemon-internal records keyed by
`(instance_id, rank)`. Each record stores the opaque handle together with the
exact transport attributes captured from the matching instance registration.
Publication fails unless rank placement, TP/DP geometry, dtype size, hidden
width, token capacity, and cross-rank agreement match current registrations.
An instance lease is valid only for the exact publishing agent generation.

A replacement agent registration synchronously drains a dead previous
generation for at most 60 seconds. All live instance owners holding leases to
that generation are terminated concurrently, receive one shared 15-second
grace period, and are then killed if necessary. The daemon removes their
registrations only after process death and installs the new agent generation
only after every old user is gone. A timed-out replacement returns retryable
`not_ready`.

Daemon restart does not require recreating native arenas. A published ATN
agent that loses daemon registration re-registers and republishes each
original handle as the corresponding instance rank re-registers. Different
models may load and register at different times: each local rank receives its
arena without waiting for unrelated configured models, while the agent keeps
one aggregate lifecycle state over all resources it owns. Each instance
re-registers and reacquires once per heartbeat; it restores the lease only when
the daemon returns the same handle already mapped by that process. A different
handle is fatal because hot-switching an attached CUDA IPC arena is unsupported.

## KV Sharing

KV cache sharing is phase 2. The phase-2 design must audit:

- SGLang 0.5.13 KV allocator and memory-pool internals,
- kvcached's SGLang autopatch and virtual memory allocator pattern,
- compatibility with xpool's per-agent worker ownership model.

Until that audit is complete, xpool must not claim cross-model KV sharing.

## Public Commands

The package exposes one `xpool <subcommand>` CLI plus an SGLang general plugin.
SGLang 0.5.13 loads general plugins from the `sglang.srt.plugins` entry point
group in `sglang serve` and `python -m sglang.launch_server`; `SGLANG_PLUGINS`
is SGLang's comma-separated plugin whitelist.

The xpool plugin entry point is
`xpool.integrations.sglang.plugin:install`. It does not patch generic FFN
`forward()` functions and does not redirect SGLang `ModelRegistry` entries. The
plugin loads a model adapter registry, installs adapter hooks, and installs
generic `ModelRunner.load_model` and `ModelRunner.init_memory_pool` lifecycle
hooks. The registry auto-discovers all
zero-argument concrete `SglangModelAdapter` subclasses owned by modules under
`xpool.integrations.sglang.models`; new model support is added by adding a model
module, not by editing a global adapter list. Production plugin discovery fails
closed if any adapter module cannot be imported, because a broken adapter tree
means the installed SGLang integration is ambiguous. Best-effort skipping exists
only for explicit non-production discovery calls such as tests. The lifecycle
hook validates SGLang server arguments, resolves xpool runtime policy only for
the currently loading model, checks SGLang TP/DP against that policy, and only
then attaches the model binding to the runner. This ordering keeps rejected
loads from leaving a half-bound model runner. There is no `XPOOL_INSTANCE_ID`;
the instance id is derived from the
one-model-one-instance mapping in TOML.

The daemon `/config` endpoint returns its complete validated process-global
`XpoolConfig`. Config-derived agent and instance placement remains available
through that model's properties and is not wrapped in a second response schema.
The endpoint must not perform expensive model metadata resolution or read model
`config.json`; SGLang model metadata and full model-derived parallel policy
remain owned by `xpool.integrations.sglang.topology` and plugin load-time
validation.

The first DeepSeek adapter directly replaces SGLang's DeepSeek FFN classes with
xpool shim classes and hooks DeepSeek weight loading to skip
`model.layers.*.mlp.*` tensors. It does not expose a module-level adapter
singleton; the registry instantiates `DeepseekV2Adapter` through auto-discovery.
It does not support DeepSeek-V3/V3.2 until those architectures have explicit
adapter coverage and validation.

```bash
uv run xpool config dump --config configs/xpool.example.toml
uv run xpool daemon check --config configs/xpool.example.toml
uv run xpool daemon serve --config configs/xpool.example.toml
uv run xpool atnagent --config configs/xpool.example.toml --cuda-device 0
UV_ENV_FILE=/path/to/xpool/.env uv run sglang serve ...
```

`xpool config dump` validates local configuration without starting resident GPU
work and reports the resolved config with registry setting sources. `xpool
daemon check` reports daemon readiness through the daemon API client because the
daemon owns service readiness. Its current scope is `atn`. `xpool daemon serve`
starts the daemon, and `xpool atnagent` starts one resident process selected
from the configured attention CUDA devices. FfnAgent CLI, runtime, and
control-plane APIs are deferred until the NVSHMEM FFN milestone.

For repeated local development commands, copy `.env.example` to an untracked
`.env`, edit local values, and set `UV_ENV_FILE=/path/to/.env` before invoking
`uv run`.

## Configuration

The repository configuration is TOML. It defines:

- daemon bind address,
- scheduler concurrency limits,
- attention-side CUDA devices,
- FFN-side CUDA devices,
- optional vendor model-base URI,
- target model ids and optional absolute model-path overrides.

It does not configure agent records, NVSHMEM ranks, instances, model
family, hidden size, or attention topology directly. These are derived:

- one `AtnAgent` per configured attention CUDA device,
- one future `FfnAgent` per configured FFN CUDA device,
- future NVSHMEM ranks assigned to attention devices in configured order,
  followed by FFN devices in configured order,
- one instance per configured model,
- SGLang-specific model family, hidden size, KV heads, and attention layout in
  `xpool.integrations.sglang.topology`, not in `xpool.config`.

Runtime policy must be derived from configuration and model metadata, not from
hard-coded planner decisions.

## Code Quality

Python checks use Ruff formatting, Ruff linting, Astral ty, and pytest through
the installed pre-commit hooks. Ruff enables `E`, `F`, `I`, `FAST`, `RUF`, `UP`,
`W`, and `ANN401`, plus public Python docstring checks for classes, functions,
methods, constructors, and documented Google-style parameters. The explicit
documentation quality test checks only Pydantic field descriptions because
they feed generated JSON Schema and OpenAPI surfaces. Broad `object`
annotations are not banned with a custom AST test. The SGLang integration layer is allowed and
expected to import SGLang concrete types directly; use local runtime guards or
casts only around SGLang attributes that are assigned dynamically and are not
visible to the type checker.

C++ and CUDA public APIs use Doxygen comments. The root `Doxyfile` is a
warning-as-error gate for public native headers under `src/cext-include`.
`clang-format` remains the native formatter; Doxygen is the public declaration
documentation completeness check.

The scheduler section uses concurrency terms, not slot-per-device terms:

```toml
[scheduler]
atn_concurrency = 1
ffn_concurrency = 1
```

The device section is the source of truth for agent placement:

```toml
[devices]
atn_cuda_devices = [0]
ffn_cuda_devices = [1]
```

The two device lists must be non-empty, unique, sorted in ascending order, and
disjoint. A CUDA device may host only one xpool role.
`devices.atn_cuda_devices` is an ordered physical device list: xpool maps SGLang
rank `i` to `devices.atn_cuda_devices[i]` without depending on
`CUDA_VISIBLE_DEVICES` remapping. The current SGLang integration can only launch
through SGLang's `base_gpu_id + rank * gpu_id_step` form, so it rejects
non-arithmetic ATN device lists in the SGLang adapter layer rather than in core
config validation.

The vendor section may define the shared local model-cache root:

```toml
[vendor]
model_base_uri = "/absolute/path/to/models"
```

Each model entry uses a full `org/name` model id. When `models[].path` is
omitted, xpool resolves the weight path as
`vendor.model_base_uri / models[].id`:

```toml
[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
```

`models[].path` remains available as an explicit absolute local override for
non-standard layouts or temporary experiments. A model entry must have either
an explicit absolute `path` or a resolved `vendor.model_base_uri`.
`vendor.model_base_uri` is config-file only; do not set it through `.env`.
The config loader must not materialize a vendor-derived path back into
`models[].path`; that field represents only the explicit model-entry override.
`ModelConfig.path` remains the schema field, but direct `model.path` access is
blocked so runtime code must call `XpoolConfig.model_path_of(model_id)`, which
returns the explicit override first and otherwise derives
`vendor.model_base_uri / model_id`. Local labs should point `.env` at an
untracked `configs/dev.local.toml`, and `*.local.toml` files are ignored so
host-specific device and model-cache paths do not leak into repository
examples.

Model entries do not carry tensor-parallel placement. `xpool.config` derives
only static serving placement: the attention world size is
`len(devices.atn_cuda_devices)`, and FFN TP is
`len(devices.ffn_cuda_devices)`. Multiple configured models share the same FFN
agent pool; the runtime scheduler arbitrates ownership. If future work
needs model-specific FFN subsets, that should be introduced as an explicit
placement policy rather than a scalar `models[].tp` knob.
Derived instances store the explicit `atn_cuda_devices` and
`ffn_cuda_devices` lists only; `atn_world_size` and `ffn_world_size` are
properties derived from those lists, not serialized schema fields.

SGLang runtime policy is owned by `xpool.integrations.sglang.topology`. SGLang's
`--tp-size` is treated as the attention world size. SGLang's
`ModelConfig.attention_arch` is the authoritative MLA/non-MLA boundary; xpool
does not infer MLA from model names or loose fields such as `kv_lora_rank`.
When SGLang reports MLA, xpool validates the positive MLA dimensions it needs
for compressed-cache layout and uses `atn_tp = 1`. When SGLang reports a
regular attention layout, xpool derives MHA/GQA/MQA from SGLang's total
attention heads and KV heads, then uses
`atn_tp = min(num_key_value_heads, atn_device_count)`. XPool then derives the
SGLang attention DP size from the configured attention devices and rejects
values greater than one until reduce-scatter FFN output is implemented. The
SGLang plugin validates `--tp-size`, `--dp-size`, `--base-gpu-id`,
`--gpu-id-step`, and `enable_dp_attention` against this derived policy.
Reduce-scatter FFN output is still unsupported and remains fail-closed until
the native ABI defines an explicit partial-result contract.

All TOML fields, CLI overrides, and xpool process environment variables for
registered settings are declared in the config registry. Every setting has one
canonical `XpoolConfig` path; environment variables are only another source for
that same logical field, never a separate environment subtree. Sources resolve
in this order:

1. CLI arguments,
2. allowlisted environment variables,
3. TOML config,
4. registry defaults.

Required settings without a default must fail fast when none of their allowed
sources provides a value. Optional paired settings may be absent individually
when their owning validator can prove the combined contract, such as
`models[].path` and `vendor.model_base_uri`: each model must resolve to exactly
one absolute local path, but either the model-specific override or the vendor
model-cache root may provide it. xpool deployment policy must not be controlled
by arbitrary environment variables: every accepted `XPOOL_*` variable must
appear in the registry with `ENV` in its allowed sources, and startup/config
helpers warn when they see an unknown `XPOOL_*` variable. Every registry setting
that allows `ENV` must declare an environment-variable name and that name must
appear in `.env.example`, either enabled as a development default or commented
as an opt-in switch. `XPOOL_CONFIG` is an
env-backed bootstrap registry setting used before TOML can be loaded, not a
runtime `XpoolConfig` field. `SGLANG_PLUGINS` belongs to SGLang's plugin loader
and is documented in `.env.example`, not in xpool's config registry.
`XPOOL_DEBUG_LOOPBACK_ENABLE=1` and `XPOOL_DEBUG_LOOPBACK_SITE` are paired
development-only env sources for `debug.loopback.enable` and
`debug.loopback.site`. The site is one of `instance`, `atnagent`, or
`ffnagent`. `instance` routes the Python FFN shim to the direct loopback debug
op. `atnagent` keeps the production `ffn_shim` route and executes the loopback
in the local AtnAgent persistent kernel. `ffnagent` is
reserved for the complete NVSHMEM request/result path and fails explicitly
until that runtime exists. Enabling loopback requires a site, while disabling
loopback forbids a site. Both fields allow only `ENV` and `DEFAULT`, so TOML
attempts fail closed. The corresponding agent process must stay resident and
continue publishing arena handles as instance ranks register so daemon pid
liveness never leaves stale handles installed. `XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE` and
`XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR` are paired development-only env sources for
`debug.graph_observer.enable` and `debug.graph_observer.outdir`: they must be
enabled/present together or disabled/absent together. The output directory is
where the devkit SGLang graph observer writes per-process JSONL capture/replay
events, and relative paths are resolved against the process working directory
during config validation. Both graph observer settings are env-only debug
settings, so TOML attempts to set them fail closed.
Runtime code must not copy or cache config values outside `xpool.config`; entry
points install one process-global resolved config through `init_global_config()`,
and business code reads it through `get_global_config()` when needed. The only
global config write API is `init_global_config`; test injection uses
`init_global_config(config=...)` rather than a separate setter.
Config source provenance is produced during the same resolution pass that
validates the config and is exposed by the resolved `XpoolConfig` object.
The loopback site is config-only: daemon registration and metadata payloads do
not carry loopback policy.
Loading the xpool SGLang plugin enables the shim; there is no separate xpool
enable flag or manually configured instance id. The instance id is derived from
the one-model-one-instance mapping.

## ABI Surfaces

The shared Python/C++ ABI version 24 starts with versioned FFN request/result
descriptors and structured transport trace records.
Native debug options use one non-negative 64-bit encoding. Bits 32 through 62
are feature flags, with bit 32 enabling loopback and bit 33 enabling transport
observation; bit 63 is reserved so the value remains representable by Torch's
signed integer schema. Bits 0 through 31 are feature-specific option fields.
Bits 0 and 1 encode the loopback site as zero for none, one for `instance`, two
for `atnagent`, and three for `ffnagent`; all remaining low bits are
reserved. Loopback enablement and its site must agree, and unknown feature or
reserved option bits fail closed. Future parameterized features receive
disjoint low-bit fields rather than reinterpreting the whole option word.
Both descriptors contain:

- ABI version and byte size,
- status/error state,
- output slot id.

FFN request descriptors also contain:

- instance index,
- layer id,
- forward mode,
- dtype,
- collective and DP padding policies,
- input/output/DP-token-count arena offsets,
- token count,
- hidden size,
- communication-slot id,
- attention TP/DP topology.

FFN result descriptors also contain:

- error code,
- output arena offset.

Descriptors must not carry raw CUDA pointers across processes. SGLang and the
local agent may map the same CUDA IPC allocation at different virtual
addresses, so all cross-process references use arena offsets. Each arena owns two
device-resident bounded ring queues backed by the generic
`xpool::utils::queue::RingQueue<uint32_t>` primitive: a free-slot queue
initialized with every slot id, and a used-slot queue initialized empty.
Instance-side shim kernels pop the arena slot from the free queue, stage input
and DP token counts, stamp a descriptor, and push the slot id to the used queue.
Each instance/rank pair binds one local transport arena. The local transport
checkpoint defaults to one reusable slot and one resident agent warp that
consumes its used queue. Queue depth is nevertheless a real native layout
parameter so shutdown and future microbatching are correct for multiple slots.
Future FFN-side concurrency belongs to the
FfnAgent scheduler/executor, not to the local attention arena queue depth.
The shim copies the completed output before pushing the slot back to the free
queue, so the agent cannot reuse output storage before the producer has
consumed it. The shim publish path must split payload staging from queue
publication so agents cannot
observe partially staged input, including when CUDA graph replay reuses the same
graph body. Queue cells use system-scope acquire/release atomics for slot
ownership publication, while descriptor status fields use the same system-scope
ordering for request/result state transitions. Descriptor headers do not carry a
separate request sequence; slot ownership is determined by the free/used queues
and the descriptor status protocol.

Transport performance observation is an explicit debug facility configured by
`debug.transport_observer.enable` and `debug.transport_observer.outdir`. The two
settings must be present or absent together. When disabled, transport arenas do
not allocate trace storage and request kernels do not read timers or write trace
records. `TransportArenaLayout` resolves this process-wide native debug option
when it constructs the arena geometry; callers provide workload dimensions but
do not pass a second observer-policy flag. When enabled, every instance request receives a monotonic trace id and
the instance and agent write device-global timestamps into the same bounded
arena record for slot acquisition, input staging, publication, agent dequeue,
descriptor grant, executor entry/exit, result publication/observation, output
copy, and slot recycling. `TransportTraceRecord` and
`TransportTraceSnapshot {sequence, dropped, records}` are matching C++ and
Python ABI structures. The destroy operator transports snapshots as a named
primitive tuple rather than an opaque tensor, and devkit exports them as
structured JSON for cross-process analysis. Nsight Systems
remains the system timeline and host-synchronization authority; Nsight Compute
provides source-correlated instruction, memory, occupancy, and stall evidence for
focused native integration workloads. Neither tool replaces the device phase
timestamps for cross-process request wall time.

Native transport interfaces use references for required arena state,
descriptors, atomic targets, and queue outputs. Pointers are reserved for
nullable observer records, optional payloads, raw device buffers, externally
owned storage views, and CUDA C API boundaries. Device-only range checking and
slot-address derivation are private `TransportArena` operations rather than
free helpers that accept an arena as their first argument.

Expected executor failures are device-visible protocol results, not CUDA traps.
The agent publishes a typed `FfnResultErrorCode`, records the first fatal
executor error in arena-local sticky state, and lets the instance request kernel
complete with poison output while preserving the CUDA context. A process-local
background monitor reads that sticky state outside the request path and
terminates the instance fail-closed. Device traps are reserved for corrupted
ABI, range, queue, or descriptor-state invariants where continuing to access the
shared arena is unsafe.

Host error inspection is exposed as a typed lifecycle snapshot rather than a
request operator. Transport observation is exported only by the arena destroy
transaction: destroy requires a live arena handle, drains the resident kernel,
captures a structured trace snapshot, releases the arena, and returns the
snapshot. Disabled observation returns zero counters and no records; enabled
observation returns the complete ring even when no request has run. Unknown or
repeated destroys are lifecycle errors rather than idempotent operations. Live
trace-ring snapshots and a second observer-specific destroy path are unsupported.

The ordinary shim path is stream ordered and must not synchronize the host after
each request. Arena detach, replacement, and shutdown are the synchronization
boundaries: they stop new launches, wait for recorded in-flight completion, and
only then close the CUDA IPC mapping. CUDA graph capture contains device work
only and must not create host callbacks or capture host-side event management.

CUDA MPS is a production transport prerequisite. The daemon does not own the
MPS lifecycle, change GPU compute mode, or repair a missing controller. Its
readiness snapshot probes the controller selected by
`CUDA_MPS_PIPE_DIRECTORY`, reports `mps_status`, and combines that status with
every participant readiness scope. `/health`, registration, heartbeat, and
configuration endpoints remain available while MPS is offline so the control
plane can diagnose and recover. The controller may have no server before the
first CUDA client connects; controller reachability, rather than a non-empty
server list, is therefore the readiness boundary. Deployment must start the
controller with every GPU visible to xpool clients and give controller and
clients the same MPS pipe and log directories.

Shutdown is a bounded drain protocol. The daemon blocks new leases and
concurrently terminates every live instance owner of the agent's arenas;
heartbeat freshness is not evidence that a live CUDA IPC mapping is safe to
free. Once shutdown is observed, producers may not enqueue new work. The
resident agent marks already-used slots failed with the shutdown error
instead of executing them and exits after the used queue is empty. A slot held
by a producer that died before publication is abandoned rather than restored to
the free queue because the arena is destroyed immediately after the resident
kernel exits. Host cleanup waits at most 60 seconds and never frees an arena
whose resident kernel failed to drain.

Current Python tests check descriptor packing and native byte-size parity before
native runtime work is extended. Explicit enum, field-offset, and alignment
parity tests must be added before xpool relies on those properties as stable ABI
guarantees.

## Implementation Order

1. Land this plan and the repository skeleton.
2. Add Python package, `xpool <subcommand>` CLI, config validation, and daemon API
   skeleton.
3. Add agent launcher and runtime preflight checks.
4. Add model-neutral SGLang plugin registration and DeepSeek FFN class adapter.
5. Add shared ABI headers and Python packing tests.
6. Close review-gate correctness gaps in SGLang plugin policy, model binding,
   and daemon readiness/config snapshots before adding native runtime state.
7. Prove single-GPU eager shim dispatch with a devkit loopback executor that
   performs a non-identity pairwise hidden-state 45-degree rotation through
   build-time `libxpool_cext.so` Torch ops.
8. Prove the same debug loopback op under direct eager execution, direct CUDA
   graph capture/replay, and isolated SGLang offline inference for eager,
   decode full CUDA graph, and prefill piecewise CUDA graph modes.
9. Implement the production `ffn_shim` publish/wait path with communication-slot
   ownership and replay-safe descriptor side effects.
10. Prove one NVSHMEM rank per agent transport with device-side progress.
11. Attach DeepSeek-V2-Lite FFN executor and correctness oracle tests.
12. Run SGLang E2E with multiple instances on the same attention GPU.
13. Start phase-2 KV sharing design and implementation.

The loopback executor in step 7 is only an ABI and CUDA graph validation tool.
It must not be used as serving evidence; SGLang E2E readiness requires the real
DeepSeek-V2-Lite FFN executor.

Native Torch ops are produced during package build/install as Linux-only
`libxpool_cext.so` and loaded from the installed xpool package by `xpool.cext`
through an idempotent startup preflight that checks only the native
`torch.ops.xpool.abi_version()` result against Python's `xpool.abi.ABI_VERSION`
and then imports `xpool.ops` so Python graph wrappers are registered before
serving code calls them.
Runtime shim code calls the compile-friendly `xpool.ops.instance.ffn_shim`
Python custom-op wrapper. The wrapper preserves SGLang's symbolic token
dimension during piecewise CUDA graph compilation and dispatches at runtime to
`torch.ops.xpool.instance.ffn_shim`. Agent-owned arena operations are exposed
through `xpool.ops.atnagent` and dispatch to `torch.ops.xpool.atnagent.*`;
instance-owned arena and shim operations are exposed through
`xpool.ops.instance` and dispatch to `torch.ops.xpool.instance.*`.
This two-layer shape contract avoids per-capture-bucket whole-model
recompilation while keeping decode CUDA graph capture and SGLang piecewise CUDA
graph prefill on the same native execution ABI. Runtime JIT extension builds
are not part of the XPool serving design.

The current production `torch.ops.xpool.instance.ffn_shim` is implemented for the
daemon-brokered transport checkpoint slice. Without a native transport arena handle
registered for the locally derived `(instance_index, rank)`, it fails closed
before launching work.
When `debug.loopback.enable=true` and `debug.loopback.site=atnagent`, the SGLang plugin registers the
requesting instance rank with the daemon after model load, fetches that rank's
filtered transport arena handle, and installs a native registry entry. The native op
publishes descriptors into the CUDA IPC arena, waits for the local attention
agent arena to complete, and returns the temporary 45-degree rotate executor
output through the production `ffn_shim` route. This proves registry-driven
production shim dispatch, CUDA IPC arena handoff, descriptor sequencing, and
local agent device progress for one same-device arena; attention admission
arbitration, the NVSHMEM hop to FfnAgents, and the real FFN executor remain
the next runtime implementation slice.
Because `debug.loopback.enable` defaults to false, the default shim route is
still not a serving-capable configuration. When the resolved global config has
the `instance` loopback site, process initialization installs the matching
native debug option, so `torch.ops.xpool.instance.ffn_shim` runs the temporary
loopback executor while preserving the same graph wrapper and native ABI. This lets tests
cover SGLang's current eager-compiler piecewise CUDA graph prefill path and
decode full-graph capture mechanics. This is not a serving readiness claim and
not a blanket torch.compile/Inductor support claim: SGLang's explicit global
`enable_torch_compile=True` remains rejected while SGLang piecewise CUDA graph
prefill with the default eager compiler remains accepted. Non-eager piecewise
CUDA graph compiler modes also remain rejected until XPool defines a compiler
contract for request publication,
communication-slot ownership, stream/event ordering, agent progress, and
replay-safe descriptor side effects.

The current daemon control-plane routes are AtnAgent-specific. AtnAgents register with
`POST /atnagent/register`, each AtnAgent
publishes rank-local transport arena handles with
`POST /atnagent/{cuda_device}/transport-arenas` and use an AtnAgent heartbeat
route. There is no generic agent endpoint and no current FfnAgent route.
Instance ranks register with
`POST /instance/register`, and each instance rank acquires its local handle
through `POST /instance/{instance_id}/transport-arena/acquire?rank={rank}`.
There is no aggregate fetch on the instance route; each SGLang rank installs
only its local arena. Runtime participants access these routes through
`xpool.service.client.XpoolClient`, not ad hoc `httpx.Client` calls. The
agent publish body contains the publisher process reference and a list of
`{instance_id, rank, handle}` bindings, where `handle` is the lowercase CUDA IPC
arena handle. Drain marks the publisher's arenas as terminating for heartbeat
warnings until every live instance lease owner exits. The instance acquire response is the bare rank-local
`TransportArenaHandleRecord`, not a wrapper object. The supported startup order
is agents first, then instance ranks: agents may start resident before
any rank is registered, then publish arenas as live instance registrations
appear. The daemon rejects handles for unknown or unregistered instance ranks,
so an early instance acquire receives a retryable `not_ready` response rather
than waiting for all ranks.

The daemon process-global `XpoolConfig` is the control-plane configuration
authority. `GET /config` returns that Pydantic model directly, without a
separate derived-view wrapper. Before every initial or recovery registration,
agents and instances submit their complete local effective config to
`POST /config/check`. The daemon compares it with its own config, including
env-only debug settings and ordered model/device lists but excluding source
provenance. A match returns `204 No Content`; a mismatch returns the existing
`409 Conflict` daemon error with stable dotted/indexed field differences, and
the client must not send its registration. Config validation stays daemon-side:
runtime clients submit their local Pydantic JSON but never reconstruct the
daemon config. The check and registration are separate requests because daemon
config is immutable for the daemon process lifetime.

Agent-published handle bindings carry only control-plane ownership facts.
They use `(instance_id, rank)` as the route-table key and leave arena geometry
opaque to Python and the daemon. Instance-scoped responses do not repeat
`instance_id` because the route already identifies the requester. The instance
process derives `instance_index` from its local xpool config before installing
the native runtime. Instance registration stores process identity, local rank,
and transport requirements including hidden-state element size and attention DP
size; the daemon uses those requirements to reject undersized or
topology-mismatched agent arenas before SGLang installs them. The transport
runtime starts only after SGLang applies its resolved memory-pool configuration.
The registered transport token capacity is the maximum of SGLang's resolved
prefill, CUDA-graph batch, piecewise-graph token, and
`ModelRunner.max_running_requests` limits. The last value covers eager decode
when a legal running batch is larger than the captured CUDA-graph buckets.
Every SGLang rank talks only to the local AtnAgent on the same
physical CUDA device. Non-local agents participate through persistent kernels
and the on-device/NVSHMEM agent network. The native transport registry is
keyed by `(instance_index, rank)`.

Daemon registration payloads carry only `pid`; the daemon samples the process
`create_time` with `psutil` during registration. Liveness checks compare the
full identity so PID reuse and permission failures cannot refresh old
registrations. Stale registrations remain visible for observability, but they
do not satisfy `/ready` and cannot serve transport arena handles. Liveness probing
must not run while holding daemon state locks. Agent metadata is bound to the
exact `ProcUniqId` that published it; if that owner no longer matches the live
agent registration, the daemon returns retryable `not_ready` instead of
handing stale CUDA IPC handles to an instance rank.
The `atnagent` loopback site remains an explicit debug checkpoint path: it
proves daemon-brokered metadata, native registry handoff, CUDA IPC descriptor
publication, and local agent progress, while warnings make clear that real
attention admission arbitration, NVSHMEM FFN routing, and FFN execution are not
implemented by this debug slice. This checkpoint still is not a complete
serving lifecycle protocol: full graph-replay error propagation and the real
executor remain part of the production transport work.

SGLang's DP padding mode is an execution hint as well as a distributed-buffer
description: CUDA graph capture uses `MAX_LEN` even when attention DP size is
one, and capture placeholders may omit `global_num_tokens_gpu`. The xpool ABI
uses DP padding mode only to describe a real multi-rank DP buffer. Therefore the
SGLang shim normalizes every attention-DP-size-one request to
`DpPaddingMode::kNone`, omits DP token counts, and sets the global DP buffer
length to the hidden-state row count. Multi-rank requests retain SGLang's
padding mode, global buffer length, and per-rank token counts; native validation
continues to reject non-none padding without those counts. SGLang DP attention
is enabled only after the real FFN executor can aggregate the described DP
input and return the reduce-scattered output expected by each ATN DP rank. Its
acceptance gate uses at least two ATN DP ranks and covers unequal rank-local
batches, idle ranks, `MAX_LEN`, and `SUM_LEN` under eager, decode full-graph,
and piecewise prefill graph execution. Layer outputs must match the model oracle
within dtype tolerance and generated token ids must match an unsplit SGLang
baseline; merely removing the topology rejection is not compatibility evidence.

Current loopback validation must cover three compatibility surfaces that SGLang
uses together in normal high-performance serving:

- eager execution,
- decode full CUDA graph capture/replay,
- prefill piecewise CUDA graph capture/replay.

SGLang graph-mode tests should use `xpool.devkit.sglang.graph_observer` when
they need proof that SGLang actually entered capture/replay paths, while native
transport timing belongs to `xpool.devkit.common.transport_observer`. Process
runtime initialization flows through `xpool.bootstrap.init(cuda_device, role)`,
which locks one Python process to a CUDA device and `RuntimeRole`, loads and
initializes the native runtime, and rejects conflicting reinitialization. After
bootstrap, the SGLang plugin and common agent constructor call
`xpool.devkit.install()`. Its unified registry recursively discovers observer
modules under `xpool.devkit`, imports only observers enabled by matching
`debug.<module_name>` config, and installs only observers declaring support for
the bootstrapped runtime role. Each observer exposes a zero-argument `install()`
entry point, declares its supported runtime roles, and reads process-global
configuration rather than receiving debug policy from production code.
The graph observer wraps SGLang full CUDA graph and piecewise CUDA graph runner
methods for recording only and must not change runner arguments, tensors,
return values, or exception behavior. It records event kinds as
`full_cuda_graph` and `piecewise_cuda_graph`. Event output is synchronous and
deliberately simple: first install creates/truncates
`xpool.graph-observer.<pid>.jsonl` in the configured output directory with
write mode so stale data is cleared and output-path errors fail early. A repeat
install updates the target event file without truncating existing events. Each
runtime event is appended to that file and flushed immediately under a
process-local lock. Runtime event-recording failures are warnings and must not
change SGLang graph runner return values or exception behavior. The observer
belongs to devkit, not the production runtime or model-adapter layer.

The transport observer follows the same boundary in agent processes.
Production `AtnArenaResource` owns only instance identity, registration, and the
native arena handle; destruction returns the ABI-defined trace snapshot without
consulting debug configuration. When enabled for `RuntimeRole.ATNAGENT`, the
devkit transport observer patches that destruction method, calls the original
method exactly once, and serializes the returned snapshot using the output
directory resolved from global config. Output-directory creation fails during
observer installation. A later per-snapshot write failure is logged as a
warning because native destruction has already completed and cannot be rolled
back.

The debug loopback op implements a non-identity pairwise 45-degree hidden-state
rotation so tests can assert that shim output was computed rather than returned
unchanged. SGLang validation has two explicit levels. Instance loopback runs
all four `(cuda graph, piecewise CUDA graph)` combinations in isolated
processes: `(off, off)`, `(off, on)`, `(on, off)`, and `(on, on)`. A separate
AtnAgent loopback test starts an isolated daemon and every configured ATN
agent, then runs eager `(off, off)` and combined graph `(on, on)` SGLang
instances through daemon registration, arena publication/acquisition, CUDA IPC,
descriptor queues, and the persistent transport kernel. It deliberately does
not start the unimplemented FfnAgent.

Both levels use the normal offline `Engine` API with `SGLANG_PLUGINS=xpool`, a
short deterministic generation, and SGLang's default graph bucket settings.
The E2E harness must not reduce SGLang's native graph buckets, even when doing so
would shorten the test, because graph-construction scaling is part of the
transport evidence and preserves coverage for future workloads.
Each graph configuration runs in a fresh process, and each transport probe owns
a fresh daemon/agent lifecycle on a temporary loopback port. The graph
observer must record capture and replay for every enabled full or piecewise
graph path, and no events for disabled paths. Token ids must match the eager
baseline within and across instance and AtnAgent loopback sites. These tests are
transport-checkpoint evidence only: production readiness still requires the
FfnAgent, NVSHMEM routing, and real FFN execution.

## Validation

Repository validation uses:

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
CMAKE_BUILD_PARALLEL_LEVEL=<jobs> uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool --config-settings-package xpool:cmake.define.XPOOL_EXPORT_COMPILE_COMMANDS=ON
uv run --no-build-isolation-package xpool ctest --test-dir "$(dirname build/*/CTestTestfile.cmake)" --output-on-failure
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
uv run pytest
uv run pytest tests/e2e -s
uv run pytest tests/unit --cov=xpool --cov-branch --cov-report=term-missing --cov-report=xml:build/coverage/unit.xml
find src/cext src/cext-include -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.cuh' \) -print0 | xargs -0 clang-format --dry-run --Werror
doxygen Doxyfile
```

Python tests use three execution layers. `tests/unit/` mirrors xpool modules and
does not exercise native behavior, launch subprocesses, or load model weights.
`tests/integration/` covers cross-module, pinned SGLang, and Python/native
contracts, including native CUDA loopback components that do not launch the
serving engine. Full SGLang Engine and model-weight workflows live under
`tests/e2e/` and use `test_e2e_*.py` filenames. Every
pytest session preflights the native operator library before collection; native
availability is therefore not a marker or a skippable resource. The
`requires_cuda`, `requires_config`, and `requires_model_weights(model_id)`
markers are executable requirements. Unavailable resources skip by default and
fail with `--strict-requirements`; malformed explicit configuration always
fails. E2E configuration is accepted only through `XPOOL_CONFIG`; pytest does
not parse dotenv files itself. Canonical pytest commands and the pre-commit hook
conditionally set `UV_ENV_FILE` to the ignored repository-root `.env`, allowing
uv to supply `XPOOL_CONFIG` when the file exists. Existing shell variables take
precedence. Default pytest collection includes unit, integration, and E2E
tests. E2E runs without strict mode whenever its declared and derived
requirements are available; use `uv run pytest tests/e2e/sglang -s` after the
same conditional env setup for a targeted run, and add
`--strict-requirements` only when unavailable resources must fail instead of
skip.
Reusable process and runtime test tools live under `tests/harness/`. Unit
coverage produces a branch-aware report without enforcing a percentage
threshold until a stable baseline exists.

C++ and CUDA code is managed by `CMakeLists.txt`, which discovers extension
sources under `src/cext` and public headers under `src/cext-include`.
The normal native build entrypoint is uv/scikit-build; use
`uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool`
to rebuild `libxpool_cext.so` before running native-op tests, optionally prefixed
with `CMAKE_BUILD_PARALLEL_LEVEL=<jobs>`. CMake enables `ccache` by default for
C, C++, and CUDA when it is found and the corresponding compiler launcher is
not already configured; disable it with
`--config-settings-package xpool:cmake.define.XPOOL_ENABLE_CCACHE=OFF`. C++/CUDA unit tests
under `tests/cext` are built by default; disable them only when needed with
`--config-settings-package xpool:cmake.define.XPOOL_BUILD_CEXT_TESTS=OFF`.
They use GoogleTest for test cases and assertions, CTest for discovery
and execution, and run before pytest in the pre-commit sequence. Run CTest
through `uv run --no-build-isolation-package xpool` so uv sync keeps the same
stable build paths as native rebuilds. Generate `compile_commands.json` for IDE indexing by passing
`--config-settings-package xpool:cmake.define.XPOOL_EXPORT_COMPILE_COMMANDS=ON`.
Native code is
formatted with `clang-format`.

Performance work must use Nsight Systems before hot-path optimization. TBT and
TPOT must be measured at generated-token boundaries, not derived from
end-to-end request latency.
