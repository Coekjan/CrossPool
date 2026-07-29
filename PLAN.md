# xpool Architecture And Plan

This document is the canonical architecture and implementation plan for xpool.
It describes the current supported design, the accepted staged-review
remediation record, and the remaining production feature gate.
Source declarations, generated native stubs, and the E2E manifest are the
authoritative detailed interfaces for implemented code; this document records
cross-module architecture, protocol invariants, ownership, and support limits.

## Status

| Phase | Status | Implemented capability | Core source |
| --- | --- | --- | --- |
| 0. Canonical documentation | Implemented | `PLAN.md` owns architecture and `docs/code-style.md` owns code conventions. | `PLAN.md`, `docs/code-style.md` |
| 1. ABI and native boundary | Implemented | ABI 54, typed pybind lifecycle, focused headers, generated stubs, and a sole Tensor dispatcher operation. | `src/cext-include/xpool/`, `src/cext-bindings/` |
| 2. Transport mailbox | Implemented | One rank-local CUDA IPC mailbox with device readiness, bounded drain, failure propagation, and optional tracing. | `src/cext/transport/`, `src/cext-include/xpool/transport/` |
| 3. Fabric core | Implemented | All-AtnAgent invocation formation, FfnAgent coordination, scheduling, publication, failure, and trace protocols. | `src/cext/fabric/`, `src/cext-include/xpool/fabric/` |
| 4. Python control plane | Implemented | Generation lifecycle, owner-authenticated control, readiness, and watchdog fail-stop. | `src/xpool/service/daemon/`, `src/xpool/runtime/` |
| 5. SGLang integration | Implemented | DeepSeek-V2, Qwen3, GLM-4.7-Flash, and Qwen3-MoE adapters, workload derivation, topology checks, and eager/full/piecewise graph-safe shim paths. | `src/xpool/integrations/sglang/` |
| 6. Test orchestration | Implemented | Resource-aware CTest/pytest scheduling, descendant supervision, in-run GPU leasing, explicit MPS-pipe ownership, and corrected harness ownership boundaries. | `tests/__main__.py`, `tests/harness/` |
| 7. Loopback evidence | Implemented | Installed-command serving, graph, topology, observer, parity, multi-model, and shutdown evidence with terminal Attempt ownership and the production serve shutdown hook. | `tests/suites/e2e/`, `tests/harness/sglang/manifest.toml` |
| 8. Real FFN execution | Blocked | No production weight loading or FFN calculation is claimed. | `src/cext/ffnagent/executor.cu` |

Implemented status requires both the code and its validation gate. Symbol
presence alone is not evidence. Phase 8 remains blocked until its independent
design is accepted and merged into this document.

The Phase 7 model-adapter expansion for GLM-4.7-Flash and Qwen3-30B-A3B passed
its complete-suite gate. A later staged review reopened Phases 6 and 7 for the
bounded lifecycle, ownership, evidence, and documentation corrections recorded
below. That remediation and its accepted focused validation completed on
2026-07-29, returning both phases to Implemented.

## Mission And Boundaries

xpool separates attention-side serving from FFN execution within one host.
SGLang owns request scheduling, attention, KV cache, graph selection, and output
postprocessing. xpool intercepts supported FFN calls, transports rank-local
requests through CUDA IPC, forms one distributed invocation across AtnAgents,
coordinates execution across FfnAgents, and returns the contribution expected
by SGLang.

The current E2E validation targets are
`deepseek-ai/DeepSeek-V2-Lite-Chat`, `Qwen/Qwen3-14B`,
`zai-org/GLM-4.7-Flash`, and `Qwen/Qwen3-30B-A3B`. These IDs are evidence
identities, not production adapter allowlists. SGLang is the sole serving
engine.

The current milestone does not:

- implement real FFN weight loading or production FFN kernels;
- infer FFN tensor or expert parallelism from the number of FfnAgents;
- support expert parallelism, attention context parallelism greater than one,
  non-`FULL` SGLang scatter modes, or multi-publisher input assembly;
- support one model with both attention TP greater than one and attention DP
  greater than one;
- support attention DP greater than one for dense Qwen3; the current Qwen3
  boundary is attention TP in `{1, 2}` with attention DP fixed to one;
- claim DP-attention piecewise Prefill graph support;
- add vLLM integration, Python NVSHMEM bindings, or compatibility aliases for
  superseded greenfield contracts;
- deliberately exercise device traps in routine tests, because a trap can
  invalidate the shared MPS server and subsequent evidence.

## Domain Language

- **Instance**: one SGLang model process and one rank-local FFN request
  producer.
- **AtnAgent**: the process role bridging rank-local Transport arenas into the
  generation Fabric.
- **FfnAgent**: one NVSHMEM participant owning FFN execution resources for one
  GPU.
- **Coordinator**: the first FfnAgent PE. It forms Invocations, schedules
  Executors, publishes admissions, and aggregates completion. It is not a
  separate process.
- **Request**: one rank-local Transport mailbox operation.
- **Submission**: one AtnAgent publication for one model-layer step.
- **Invocation**: the distributed operation formed from matching Submissions
  from every configured AtnAgent.
- **Execution**: one cooperative FFN evaluation across all FfnAgents.
- **Executor**: one aligned distributed execution slot present on every
  FfnAgent. It is independent of FfnAgent count.
- **Input Publisher**: the sole AtnAgent that publishes replicated `FULL`
  input. It is PE zero in the current milestone.
- **Result Handoff**: placement of the returned rank-local contribution for
  SGLang postprocessing.
- **Fail-Stop**: the owning process emits bounded, ownership-safe diagnostics
  and exits nonzero instead of continuing from an untrusted state.

## System Architecture

### Processes And Native Boundary

xpool has four process roles:

1. `xpool daemon` owns registration, process identity, generation planning,
   Transport leases, readiness, and shutdown selection. It initializes
   `xpool.native` as the daemon role but never initializes CUDA or joins
   NVSHMEM. Its only business-level native operation is Fabric UID creation.
2. `xpool atnagent` owns CUDA IPC Transport arenas and one AtnAgent NVSHMEM PE.
3. `xpool ffnagent` owns one FfnAgent NVSHMEM PE and its resident execution
   path. The first FfnAgent PE also hosts the Coordinator.
4. SGLang Instance processes attach their Transport arena and invoke
   `xpool.ops.ffn_shim`.

Native control and resource lifecycle use typed `xpool.native.fabric` and
`xpool.native.transport` bindings. `xpool.ops.ffn_shim` is the sole Torch
dispatcher operation because it is the compile-visible Tensor data path. The
daemon does not use Torch operators for control-plane lifecycle.

Native opaque binary identifiers are strong values, not string aliases.
`FabricUid` and `TransportArenaHandle` are distinct instantiations of one
trivially-copyable `HexValue<T>` utility with static lowercase-hex `decode()`,
member `encode()`, byte equality, and byte hashing. Native metadata and
registries retain those binary values. Pybind accepts and returns Python strings
at the process boundary and converts exactly once; Python keeps its existing
validated Fabric UID and Transport-handle domain values rather than exposing a
second native mirror type.

The immutable generation PE order is:

```text
[all AtnAgent PEs in configured order][all FfnAgent PEs in configured order]
coordinator_pe = atnagent_count
```

One generation replaces the complete Agent and Instance world. A failed
participant cannot recover or rejoin the retained generation.

### Startup And Readiness

Startup has two control-plane barriers:

1. Every Instance rank registers the same model workload after SGLang resolves
   memory-pool and request-concurrency geometry. AtnAgents publish each
   available Transport arena incrementally, while Fabric planning waits for
   complete configured membership and workload agreement.
2. After graph capture, every Instance rank publishes the same generation and
   plan digest.

External readiness is the conjunction of:

- all configured Agent and Instance registrations being live;
- every Instance rank having a geometry-matching, lease-eligible Transport
  publication;
- the Fabric generation being `EXECUTABLE`;
- every Instance rank completing the generation/digest initialization barrier;
- the externally managed MPS controller being online; and
- invocation, owner, and protocol failure all being absent.

Device-published readiness is authoritative. Host kernel launch return is not
evidence that a resident kernel crossed its startup barrier.

## Control Plane

`ControlPlane` is the daemon aggregate root and owns the lock protecting
registration, Transport, Fabric, and readiness invariants. Its subsystem
registries are caller-synchronized, not lock-free algorithms.

Generation lifecycle:

```text
JOINING -> EXECUTABLE -> QUIESCING -> DRAINING -> FINALIZING -> STOPPED
any nonterminal phase -> ABORTING -> STOPPED
```

Participant lifecycle:

```text
JOINING -> JOINED -> ACTIVE -> QUIESCED -> DRAINING -> DRAINED -> FINALIZED
```

Every configured AtnAgent and FfnAgent PE participates in each generation
barrier. Instances are generation owners and initialization-barrier members,
but not Fabric participants. Participant reports commit local progress only
after daemon acknowledgement. Heartbeats prove liveness and return the latest
daemon command/failure snapshot; they never commit participant progress.

The watchdog is the sole periodic driver for owner-loss detection, MPS status,
lifecycle deadlines, and retrying abort cleanup. Any watchdog exception other
than normal task cancellation is a daemon-local fail-stop: the application
preserves the traceback and exits nonzero. It does not restart the watchdog or
retain a responsive HTTP service with stale control-plane state. A first-error
application failure latch carries the cause to the CLI-owned Uvicorn server;
the server observes it in its normal tick loop, performs bounded shutdown, and
returns a nonzero daemon command status without self-signals or `os._exit`.

Failure and lifecycle remain orthogonal:

- **Invocation failure** is the canonical device-side failure of one
  distributed invocation.
- **Owner failure** is daemon-observed process exit, staleness, or replacement.
- **Protocol failure** is a control/data protocol invariant violation described
  by a diagnostic string.

Each category is first-writer-wins and may coexist with the others. Invocation
or Instance-owner failure can select cooperative quiesce while the complete
Fabric PE world remains live. AtnAgent/FfnAgent owner loss, protocol failure,
or inability to prove collective convergence selects `ABORTING`; the daemon
then terminates the generation process trees without invoking unsafe NVSHMEM
collectives. `STOPPED` remains retained until every original owner exits, then
the generation is retired.

Transport lease admission stops before Transport drain. Existing leases remain
owned until explicit quiesce. A racing acquisition must either commit against
one fully validated executable generation or fail without leaving a partial
lease. AtnAgent debug loopback is the sole exception: because it terminates in
the Transport Resident without entering Fabric, the daemon may admit its lease
before Fabric becomes executable. That exception is selected only from the
daemon's process-global debug configuration; clients cannot request it. Fabric
quiescing or terminal phases and the local AtnAgent admission gate still reject
new leases on every path.

Authoritative Python boundaries are `src/xpool/fabric.py`,
`src/xpool/service/wire.py`, `src/xpool/service/daemon/`, and
`src/xpool/runtime/agent.py`.

## Data Plane

### Transport

Each Instance rank and its AtnAgent share one single-producer/single-consumer
CUDA IPC arena. The immutable layout is stored at offset zero and describes one
mailbox, fixed input/output payloads, optional DP token counts, mutable arena
state, and optional trace storage. Process-local resident command state is not
part of the shared arena.

The normal mailbox cycle is:

```text
Dormant -> Idle -> Staging -> Published -> Evaluated -> Idle
```

- The AtnAgent Resident publishes `Dormant -> Idle` only after the complete
  cooperative grid crosses its startup barrier.
- The Instance owns `Staging`, publishes the complete request, and later owns
  result acknowledgement.
- The AtnAgent owns evaluation of `Published` and publishes the result as
  `Evaluated`.
- Drain races only for an `Idle` mailbox. In-flight ownership reaches a bounded
  terminal result before the mailbox closes.
- `Closed` is terminal and cannot be reused.

`TransportTraceRecord.closed` records the raw GPU timestamp at which a request
enters the `Closed` terminal state. It does not mean that the mailbox returned
to `Idle`; acknowledgement and reusable-slot recovery are separate lifecycle
actions. The pybind documentation and generated stub must preserve this
distinction without changing trace field order or ABI.

One Instance process issues no concurrent FFN request against the same arena;
there is no RingQueue, slot count, or free/used queue compatibility path.
Before launching the request kernel, the native Instance boundary validates a
two-dimensional contiguous CUDA hidden-state tensor on the attached arena's
device. Optional DP token counts must be contiguous, one-dimensional, and on
that same device. Shape, dtype, capacity, and token-count length are also host
preconditions; invalid inputs fail before device code and never use a trap as a
recoverable validation mechanism.
Transport arena handles cross the Python control plane as validated hexadecimal
identity values, while native code owns binary CUDA IPC representation.

### Fabric

One model-layer operation follows this publication chain:

```text
Submission -> Invocation -> Admission -> [Prefill InputReady]
           -> Completion -> Result -> Acknowledgement
```

Every AtnAgent publishes a matching Submission. The Coordinator forms one
Invocation only after all configured AtnAgents agree on model, layer, sequence,
shape, forward mode, and result semantics. The Scheduler grants one distributed
Executor lease and publishes Admission. Every FfnAgent publishes Completion;
the Coordinator publishes Result only after all required completions; every
AtnAgent acknowledges consumption before the Scheduler entry and Executor are
reused.

Decode and Prefill intentionally use different payload ownership:

- **Decode** uses fixed per-model input/output payloads. The Input Publisher
  stages the model payload before Submission, and each FfnAgent pulls it after
  Admission.
- **Prefill** uses the admitted Executor's reusable input/output payload. The
  Input Publisher pushes the larger payload after Admission and publishes
  `InputReady` to every FfnAgent before execution.

This keeps predictable, frequently reused Decode storage independent from
large Prefill capacity, while Prefill pays admission before consuming a shared
Executor workspace. A2F and F2A remain distinct because execution may not
overwrite input before every consumer has finished reading it.

The Scheduler is a tagged policy value with FIFO and random implementations.
Its mutable state is isolated from immutable arena geometry. Executor count
expresses concurrent distributed execution capacity, not the number of
FfnAgents.

Fabric failure publication is canonical and first-writer-wins. Request result,
generation failure, shutdown, scheduler state, lifecycle phase, and trace state
remain separate facts rather than overloaded status values.

Fabric symmetric allocation is an explicit collective owner. Normal shutdown
drains device work, explicitly destroys the arena while NVSHMEM is live,
unregisters the CUDA module, and only then finalizes the participant host
library. Its destructor never attempts an implicit `nvshmem_free`; a live owner
at destruction is a fail-stop lifecycle violation. This differs intentionally
from independently releasable CUDA and IPC owners, whose destructors may perform
best-effort local cleanup.

Transport and Fabric share generic checked layout, tracing, wait, cooperative,
and CUDA ownership mechanisms only. Their protocol records, publications,
arena states, views, and snapshots remain subsystem-owned.

## SGLang Integration

SGLang integration reads concrete pinned-SGLang types. Model binding happens
during model load; workload geometry is derived after memory-pool
initialization from the resolved `ModelRunner.server_args`, hidden-state dtype,
request concurrency, Prefill capacity, and graph capacities. Graph capture then
uses the installed shim without changing Transport or Fabric geometry.

The current supported graph paths are:

- eager execution;
- Decode full CUDA graph replay; and
- Prefill piecewise CUDA graph replay when attention DP is one.

DeepSeek-V2 supports attention `(TP, DP)` placements `(1, 1)`, `(2, 1)`, and
`(1, 2)`. With attention DP greater than one, piecewise Prefill is disabled for
the pinned SGLang behavior and full/eager paths remain eligible. Dense Qwen3
supports `(1, 1)` and `(2, 1)` only. The topology validator rejects combined
attention TP greater than one and DP greater than one for every model. Context
parallelism, expert parallelism, quantization, speculative decoding, offload,
disaggregation, and other unrepresented modes fail before serving readiness.

DeepSeek-V2 and Qwen3 model-specific binding lives under
`src/xpool/integrations/sglang/models/`. Generic hook installation, topology,
workload, shim, and server-argument policy live in the parent SGLang integration
package. Model paths are resolved only through
`XpoolConfig.model_path_of(model_id)`. Architecture selects the family adapter;
model ID is configuration identity and path resolution, not adapter evidence.

### Accepted Phase 7 Model-Adapter Expansion

This subsection is accepted implementation work, not current evidence. It adds
independent `Glm4MoeLiteAdapter` and `Qwen3MoeAdapter` families parallel to the
existing DeepSeek-V2 and dense Qwen3 adapters. Adapters match exact architecture
names `Glm4MoeLiteForCausalLM` and `Qwen3MoeForCausalLM`; they do not match model
IDs, depend on each other, or share a generic MoE adapter base class. Strict
eager registry discovery remains unchanged.

The acceptance boundary remains Phase 7: load real attention-side model
weights, filter decoder FFN weights, replace every decoder FFN with a
parameter-free shim, traverse FfnAgent loopback, and prove eligible graph modes
and token parity. It is not real FFN numerical correctness, model-quality, or
Phase 8 weight/execution evidence.

#### Common adapter and shim APIs

`src/xpool/integrations/sglang/adapter.py` adds the following lazy projection:

```python
def filter_decoder_ffn_weights[W](
    weights: Iterable[tuple[str, W]],
) -> Iterator[tuple[str, W]]: ...
```

It removes only names matching
`^model\.layers\.\d+\.mlp(?:\.|$)`. Family adapters retain their own typed
load hook and do not use substring matching.

`SglangModelAdapter` gains one class-level capability fact:

```python
class SglangModelAdapter(ABC):
    name: str
    supports_dp_attention: bool = False
```

`DeepseekV2Adapter`, `Glm4MoeLiteAdapter`, and `Qwen3MoeAdapter` set
`supports_dp_attention = True`; dense `Qwen3Adapter` inherits `False` until a
separate adapter-boundary and E2E change proves that path. `AtnKind` remains
descriptive model metadata and does not grant or deny DPA. No capability enum
or wrapper object is added for this binary policy.

Adapter selection moves before binding resolution. The selected capability is
passed through these exact signature changes:

```python
# Old
@classmethod
def XpoolModelBinding.resolve(
    cls,
    model_runner: ModelRunner,
    server_args: ServerArgs,
) -> XpoolModelBinding: ...

# New
@classmethod
def XpoolModelBinding.resolve(
    cls,
    model_runner: ModelRunner,
    server_args: ServerArgs,
    *,
    supports_dp_attention: bool,
) -> XpoolModelBinding: ...

# Old
@classmethod
def ParallelPolicy.from_server_args(
    cls,
    spec: ModelSpec,
    server_args: ServerArgs,
    *,
    atnagent_count: int,
) -> ParallelPolicy: ...

# New
@classmethod
def ParallelPolicy.from_server_args(
    cls,
    spec: ModelSpec,
    server_args: ServerArgs,
    *,
    atnagent_count: int,
    supports_dp_attention: bool,
) -> ParallelPolicy: ...
```

Binding and policy do not retain the adapter or the capability boolean. The
policy rejects attention DP greater than one when the selected adapter does not
support it, before CUDA bootstrap or model loading.

`ModelSpec` gains the bound geometry validator:

```python
def validate_attention_tp(self, size: int) -> None: ...
```

Query heads must divide evenly by effective attention TP. MLA has no ordinary
KV-head divisibility rule. For GQA, MHA, and MQA, KV heads divide across TP
when `num_key_value_heads >= size`; otherwise `size` must divide by
`num_key_value_heads` so KV heads replicate evenly. This replaces the current
unconditional `num_key_value_heads % atn_tp_size == 0` check.

Internal policy terminology no longer propagates SGLang's overloaded
`ServerArgs.tp_size` name. The structures change exactly as follows:

```python
# Old ParallelPolicy topology fields
atn_kind: AtnKind
sglang_tp_size: int
sglang_dp_size: int
atn_tp_size: int
atn_dp_size: int
enable_dp_attention: bool

# New ParallelPolicy topology fields
worker_world_size: int
atn_tp_size: int
atn_dp_size: int

# Old XpoolModelBinding fields
sglang_rank: int
sglang_tp_size: int
sglang_dp_size: int
enable_dp_attention: bool

# New XpoolModelBinding fields
worker_rank: int
worker_world_size: int
```

`XpoolModelBinding` continues to retain independent `atn_tp_rank`,
`atn_tp_size`, `atn_dp_rank`, and `atn_dp_size` fields. DPA enablement is
derived from `atn_dp_size > 1`; worker coordinates obey
`worker_rank = atn_dp_rank * atn_tp_size + atn_tp_rank`; and
`worker_world_size = atn_tp_size * atn_dp_size`. Every model still uses every
configured AtnAgent, so `worker_world_size == config.atn_world_size`. Visible
machine GPUs and FfnAgent GPUs are not part of this equality. Per-model
AtnAgent subsets remain a separate, unsupported placement design.

`SglangModelAdapter.require_ffn_shims()` changes from:

```python
def require_ffn_shims(
    self,
    model: nn.Module,
    *,
    expected_layer_count: int,
    allowed_shim_types: tuple[type[FfnShimModule], ...],
) -> tuple[FfnShimModule, ...]: ...
```

to:

```python
def require_ffn_shims(
    self,
    model: nn.Module,
    *,
    expected_layer_kinds: Sequence[FfnLayerKind],
    allowed_shim_types: tuple[type[FfnShimModule], ...],
) -> tuple[FfnShimModule, ...]: ...
```

The new method sorts shims by `layer_id`, requires IDs equal to
`range(len(expected_layer_kinds))`, requires each concrete shim type to be
allowed, requires every `layer_kind` to match its ordinal, and returns the
sorted tuple. Duplicate, missing, extra, wrong-kind, and wrong-type layers fail
closed.

`src/xpool/integrations/sglang/shim.py::FfnShimModule.__init__()` keeps its
signature but owns its scalar invariants: `layer_id` is a non-boolean,
non-negative integer and `hidden_size` is a non-boolean, positive integer.
Invalid values raise `ValueError`; family replacements do not duplicate these
checks.

The four typed weight hooks are exactly:

```python
def around_load_weights(
    original_fn: Callable[
        [DeepseekV2ForCausalLM, Iterable[tuple[str, torch.Tensor]], bool],
        None,
    ],
    model: DeepseekV2ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
) -> None: ...

def around_load_weights(
    original_fn: Callable[
        [Qwen3ForCausalLM, Iterable[tuple[str, torch.Tensor]]],
        None,
    ],
    model: Qwen3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None: ...

def around_load_weights(
    original_fn: Callable[
        [
            Glm4MoeLiteForCausalLM,
            Iterable[tuple[str, torch.Tensor]],
            bool,
            dict[str, nn.Parameter] | None,
        ],
        None,
    ],
    model: Glm4MoeLiteForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
    params_dict: dict[str, nn.Parameter] | None = None,
) -> None: ...

def around_load_weights(
    original_fn: Callable[
        [Qwen3MoeForCausalLM, Iterable[tuple[str, torch.Tensor]], bool],
        None,
    ],
    model: Qwen3MoeForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_mtp: bool = False,
) -> None: ...
```

No hook uses `ParamSpec`, `Concatenate`, a generic return value, variadic
arguments, a callable `Protocol`, or a callable type alias. DeepSeek and GLM
reject `is_nextn=True`; Qwen3-MoE rejects `is_mtp=True`.

#### Family replacements and validation

`src/xpool/integrations/sglang/models/glm4_moe_lite.py` defines:

```python
class XpoolGlm4MoeLiteMLP(FfnShimModule, Glm4MoeLiteMLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None: ...

class XpoolGlm4MoeLiteSparseMoeBlock(
    FfnShimModule,
    Glm4MoeLiteSparseMoeBlock,
):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        is_nextn: bool = False,
    ) -> None: ...

    def get_moe_weights(self) -> list[torch.Tensor]: ...

class Glm4MoeLiteAdapter(SglangModelAdapter):
    name = "glm4_moe_lite"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]: ...
    def matches(self, model_runner: ModelRunner) -> bool: ...
    def validate_after_load(self, model_runner: ModelRunner) -> None: ...
```

Its hooks are exactly:

- REPLACE `sglang.srt.models.glm4_moe_lite.Glm4MoeLiteMLP`;
- REPLACE `sglang.srt.models.glm4_moe_lite.Glm4MoeLiteSparseMoeBlock`; and
- AROUND `sglang.srt.models.glm4_moe_lite.Glm4MoeLiteForCausalLM.load_weights`.

`src/xpool/integrations/sglang/models/qwen3_moe.py` defines:

```python
class XpoolQwen3MoeSparseMoeBlock(FfnShimModule, Qwen3MoeSparseMoeBlock):
    def __init__(
        self,
        layer_id: int,
        config: Qwen3MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...

    def get_moe_weights(self) -> list[torch.Tensor]: ...

class Qwen3MoeAdapter(SglangModelAdapter):
    name = "qwen3_moe"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]: ...
    def matches(self, model_runner: ModelRunner) -> bool: ...
    def validate_after_load(self, model_runner: ModelRunner) -> None: ...
```

Its hooks are exactly:

- REPLACE `sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock`; and
- AROUND `sglang.srt.models.qwen3_moe.Qwen3MoeForCausalLM.load_weights`.

Replacement constructors mirror pinned SGLang argument order, types, and
defaults but do not call the original FFN constructors. Dense GLM derives the
decoder layer ID from the canonical prefix; sparse shims use the explicit layer
ID. They require `silu`, and GLM sparse rejects NextN. They initialize only
`FfnShimModule` and `get_moe_weights() -> []`; they do not retain or create
intermediate, quantization, prefix, stream, expert, top-k, shared-expert, TP, EP,
gate, or native execution state. The existing dense Qwen3 and DeepSeek shims
likewise remove unconsumed constructor state. DeepSeek sparse alone retains
`experts.moe_runner_config.inplace` and `get_moe_weights()`, because pinned
DeepSeek decoder control flow reads that surface.

The new sparse replacements deliberately do not expose `experts`, so pinned
SGLang's `hasattr(layer.mlp, "experts")` native-MoE detection does not admit
them to expert balancing or native expert execution. Their original-class type
identity and empty `get_moe_weights()` preserve only the observed model-loader
and lazy-weight-index compatibility surface.

Every concrete adapter implements its own `validate_after_load()` and calls
common validation helpers; the base class gains no abstract validation method
or declarative shim-contract structure. Post-load validation requires the
loaded model to be the exact pinned SGLang `ForCausalLM` family type and reads
`model.config`, the config actually used and possibly mutated during model
construction. It does not derive layer policy from raw checkpoint JSON or
`ModelRunner.model_config.hf_config`. This is required because pinned GLM sets
`config.moe_layer_freq = 1` before constructing decoder layers.

Dense Qwen3 expects all `DENSE` layers and Qwen3-MoE expects all `SPARSE`
layers. GLM and DeepSeek independently reproduce the pinned family policy:

```python
is_sparse = (
    n_routed_experts is not None
    and layer_id >= first_k_dense_replace
    and layer_id % moe_layer_freq == 0
)
```

`num_hidden_layers` is positive, `first_k_dense_replace` is non-negative,
`moe_layer_freq` is positive, and `n_routed_experts` is `None` or positive.
Adapters do not call the decoder's private `_is_layer_sparse()` and do not
require checkpoint-specific expert counts.

Every replaced FFN must retain `ScatterMode.FULL`. DeepSeek-V2,
GLM-4.7-Flash, and Qwen3-MoE additionally require every owning
`LayerCommunicator.allow_reduce_scatter` value to be `True`; dense Qwen3
requires it to be `False`. The concrete adapters prove this with
`require_full_mlp_boundaries(..., allow_reduce_scatter=...)` after loading.
The class-level DPA capability never substitutes for this loaded-model proof.
Runtime `use_reduce_scatter` continues to select
`FfnResultHandoff.REDUCE_SCATTER_INPUT`; otherwise the shim requests
`FfnResultHandoff.REPLICATED_FULL`.

Failure categories remain bounded: invalid constructor values raise
`ValueError`; unsupported shim paths such as NextN, MTP, or an unparseable
decoder prefix raise `ShimUnavailableError`; post-load model/shim/count/ID/kind
invariant failures raise `RuntimeError`; topology failures raise
`TopologyError`. Existing unique-adapter matching failures remain
`RuntimeError`; no exception hierarchy is added.

#### Topology and E2E evidence

`ParallelPolicy.from_server_args()` centrally rejects attention DP greater than
one unless the selected adapter declares `supports_dp_attention = True`. It
continues to reject combined effective attention TP greater than one and DP
greater than one for every model. Pure TP and pure DPA remain the only accepted
forms. The dense Qwen3 adapter's duplicate runtime DP check is removed because
the selected adapter capability is the single policy source.

Topology remains explicit deployment intent. xpool validates resolved SGLang
arguments and never derives TP or DP from visible or unused GPUs. In pinned
SGLang, `ServerArgs.tp_size` is the worker world size under DPA; effective
attention TP is `worker_world_size / atn_dp_size` because attention CP remains
one. A future recommendation tool may advise placement but must not mutate
production topology.

GLM-4.7-Flash is MLA and covers `(TP, DP)` placements `(1, 1)`, `(2, 1)`, and
`(1, 2)`. Qwen3-30B-A3B is GQA and covers the same three placements. Every
placement runs `(FfnAgents, Executors)` values `(1, 1)` and `(2, 2)`. DP-one
cases cover eager, Decode full graph, and Prefill piecewise graph. DP-two cases
cover eager and Decode full graph only because pinned SGLang disables piecewise
graph whenever DPA is enabled.

`tests/harness/sglang/manifest.toml` adds these exact cases:

| Case | TP | DP | FfnAgents | Executors | Modes | Estimate (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `glm4-7-flash-tp1-dp1-f1-e1` | 1 | 1 | 1 | 1 | eager/full/piecewise | 600 |
| `glm4-7-flash-tp1-dp1-f2-e2` | 1 | 1 | 2 | 2 | eager/full/piecewise | 700 |
| `glm4-7-flash-tp2-dp1-f1-e1` | 2 | 1 | 1 | 1 | eager/full/piecewise | 750 |
| `glm4-7-flash-tp2-dp1-f2-e2` | 2 | 1 | 2 | 2 | eager/full/piecewise | 850 |
| `glm4-7-flash-tp1-dp2-f1-e1` | 1 | 2 | 1 | 1 | eager/full | 800 |
| `glm4-7-flash-tp1-dp2-f2-e2` | 1 | 2 | 2 | 2 | eager/full | 900 |
| `qwen3-30b-a3b-tp1-dp1-f1-e1` | 1 | 1 | 1 | 1 | eager/full/piecewise | 700 |
| `qwen3-30b-a3b-tp1-dp1-f2-e2` | 1 | 1 | 2 | 2 | eager/full/piecewise | 800 |
| `qwen3-30b-a3b-tp2-dp1-f1-e1` | 2 | 1 | 1 | 1 | eager/full/piecewise | 800 |
| `qwen3-30b-a3b-tp2-dp1-f2-e2` | 2 | 1 | 2 | 2 | eager/full/piecewise | 900 |
| `qwen3-30b-a3b-tp1-dp2-f1-e1` | 1 | 2 | 1 | 1 | eager/full | 850 |
| `qwen3-30b-a3b-tp1-dp2-f2-e2` | 1 | 2 | 2 | 2 | eager/full | 950 |
| `glm4-7-flash-qwen3-30b-a3b-tp1-dp1-f2-e2` | 1 | 1 | 2 | 2 | eager | 1200 |

The model aliases are `glm4_7_flash` and `qwen3_30b_a3b`; model IDs are
`zai-org/GLM-4.7-Flash` and `Qwen/Qwen3-30B-A3B`; both use test-only
`max_total_tokens=16384`. Every case uses a 3600-second timeout and Transport
and Fabric trace capacities of 32768. Estimates are scheduler hints and are
adjusted only when measured duration differs by more than 50 percent.

New models appear only in `model_serving_cases`, whose current complete path is
FfnAgent loopback. The existing DeepSeek-only `loopback_serving_cases` remains
the sole Instance/AtnAgent/FfnAgent site sweep. The new two-model case reuses
`assert_two_model_executor_overlap()`: coordinator records are grouped by
`model_index`, and at least one pair on distinct executor indexes must have
overlapping `[scheduled, scheduler_released]` intervals. No trace or observer
schema changes.

Each DPA case must prove that both attention-DP participants load and bind,
that `global_num_tokens_gpu` and the selected padding mode reach the shim, that
the FFN result handoff matches SGLang's communicator decision, and that eager
and Decode full-graph token IDs match. It must also record that piecewise graph
was disabled by resolved SGLang policy rather than silently omit the assertion.

The manifest is the sole E2E model-ID and case catalog. External
`XPOOL_CONFIG` remains a complete valid config but its `models` do not select or
override E2E models; tests read only `vendor.model_base_uri` and host/base
settings. Requirement preflight resolves each manifest model ID beneath that
root. A missing config file, invalid config schema, or relative model root is
`RequirementMisconfigured`; a valid config without a model root, model
directory, or `config.json` is `RequirementUnavailable`; architecture mismatch
is a test failure. Each task writes a config containing only that case's models,
then runtime resolution and architecture validation use the task-local
`XpoolConfig.model_path_of()`.

Both families reuse the deterministic completion probe and existing inference,
observer, parity, duration, and graph artifacts. No prompt/chat schema is added.
No `configs/dev.local.toml` or example-config model list is expanded.

#### Implementation and validation surface

Production changes are limited to `adapter.py`, `plugin.py`, `shim.py`,
`topology.py`, the existing `models/deepseek_v2.py` and `models/qwen3.py`, and new
`models/glm4_moe_lite.py` and `models/qwen3_moe.py`. Registry behavior is
unchanged. Harness changes are limited to
`tests/harness/runner/requirements.py`, `tests/harness/sglang/launch.py`, and
`tests/harness/sglang/manifest.toml`.

Common adapter/filter tests live in
`tests/suites/integration/sglang/test_adapter.py`; generic shim behavior moves
to `tests/suites/integration/sglang/test_shim.py`; family-only tests live under
`tests/suites/integration/sglang/models/{deepseek_v2,qwen3,glm4_moe_lite,qwen3_moe}/`,
with `test_adapter.py` and `test_shim.py` where the family has both concerns.
`tests/suites/integration/sglang/test_registry.py` and `test_topology.py` cover
unique architecture selection, adapter-owned DPA capability, pure TP/pure DPA
acceptance, combined TP-by-DP rejection, MLA query-head geometry, and ordinary
KV-head sharding and replication. `test_model_binding.py` and
`plugin/test_model_runner.py` cover worker rank/world terminology, selected
capability propagation, and fail-fast ordering before bootstrap and model load.
`tests/suites/unit/harness/test_requirements.py`,
`tests/suites/unit/harness/sglang/test_manifest.py`, and
`test_materialization.py` cover base-root discovery, case declarations, and
task-local model projection. The existing E2E test remains
manifest-parameterized. Its observer helper keeps the accepted overlap
predicate but reports aggregate interval counts, Executor identities,
candidate-pair count, and maximum overlap duration when the predicate fails.
Tests assert public behavior rather than removed attributes, source strings, or
fixed manifest counts.

Implementation updates `AGENTS.md` with the accepted testing boundary: the
manifest owns E2E model IDs and cases; external `XPOOL_CONFIG` supplies the
model root and host settings; the task-local config owns the selected models.
No ADR, supported-model document, tracked glossary, or
`configs/dev.local.toml` change is added.

Validation runs focused Unit/Integration selectors, then verifies MPS for the
seven UUIDs selected by `.env` and runs exactly one full pre-commit invocation.
Its test-suite hook owns the canonical strict-requirements suite, so
`python -m tests` is not run separately immediately beforehand. Artifacts must
prove inference, graph selection, token parity, observer completeness, and
two-model overlap. Estimate-only calibration does not rerun the suite.

This slice does not change `XpoolConfig`, `ModelConfig`, E2E manifest schema,
FFN request metadata, Fabric/Transport metadata, native ABI, bindings, generated
stubs, C++/CUDA, protocol state, or observer schema. Experts, intermediate
widths, shared-expert shapes, real weights, and real execution remain Phase 8.

Debug loopback is selected through `debug.loopback.enable` and
`debug.loopback.site`. Instance and AtnAgent sites bypass progressively more of
the data plane. Instance loopback acquires no Transport lease. AtnAgent loopback
uses Transport but does not require an executable Fabric generation. FfnAgent
loopback traverses the complete orchestration and protocol path and therefore
requires executable Fabric, but performs only the debug pair rotation. None is
evidence of real FFN weight execution.

## Configuration And Observation

Runtime configuration flows through `xpool.config`. Entry points install one
process-global config and business logic reads it from the global accessor.
Resolution order is CLI, allowlisted environment, TOML, then defaults. Debug
settings are environment/default-only and are rejected in TOML.

Python passes validated debug options directly with Pydantic JSON. Native host
and device code read target-specific storage through the same
`debug::options()` interface. Arena layout construction reads observer capacity
internally; debug options never leak into layout call signatures.

Transport and Fabric observers use generic bounded trace storage but retain
subsystem-specific records and event dependency checks. Observer JSON has no
schema version. Traces are evidence and diagnostics, not control-plane state.
Native trace state-machine misuse remains fail-stop, while pybind validates
that a queried Fabric event enum matches the record kind and raises
`ValueError` for an invalid Python combination before entering native state
access. Sender-side `Published` timestamps are local publication edges; for a
fan-out they prove that every destination publication call was issued, not that
every destination observed it. Receiver-side `Observed` timestamps carry that
observation meaning. Transport records its sender edge immediately before the
release-store to preserve the trace event dependency against a racing receiver.

## Orderly SGLang Instance Departure

Orderly Instance departure is a production lifecycle invariant, not a test-only
cleanup policy. A normal SGLang SIGTERM must stop scheduler submission,
synchronize all process-local CUDA work, detach the imported Transport arena,
and deregister the Instance rank before the scheduler process exits. The first
successful deregistration remains the daemon-authoritative event that selects
generation-wide quiesce; no reverse daemon RPC, new wire value, or native ABI is
introduced.

The pinned SGLang integration registers three fail-closed hooks:

```text
sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler
sglang.srt.managers.scheduler.Scheduler.run_event_loop
sglang.cli.serve.kill_process_tree
```

The parent SIGTERM hook marks only normal request-draining shutdown. Immediately
before SGLang's normal process-tree cleanup, the `kill_process_tree` wrapper
enters orderly drain when the process-local marker is set and
`parent_pid == os.getpid()`. `include_parent` is forwarded but is not an
eligibility condition because the installed serve path passes
`include_parent=False`. The wrapper uses SGLang's scheduler-process discovery,
sends SIGTERM to every scheduler, and waits against one shared deadline. Each
scheduler's event-loop wrapper converts SIGTERM into a private control-flow
interruption, stops submitting work, executes
`torch.cuda.synchronize(model_runner.device)`, and calls the existing
`XpoolModelRuntime.detach(model_runner)`. Its signal handler remains installed
during cleanup and ignores repeated SIGTERM so daemon lease quiesce cannot
interrupt an in-progress detach.

Scheduler discovery, signaling, waiting, failed exits, and timeout are
best-effort parent diagnostics. Internal drain exceptions are logged and never
prevent SGLang's original cleanup. `around_kill_process_tree()` invokes the
original callable from `finally` with the unchanged `parent_pid`,
`include_parent`, `skip_pid`, and `wait_timeout`; an exception from that
original callable propagates unchanged.

CUDA synchronization or detach failure leaves the Instance registered and is a
fail-stop scheduler error; it must never be reported as orderly departure.
SIGQUIT, scheduler exceptions, external SIGKILL, and scheduler-shutdown timeout
remain forced paths. They retain explicit diagnostics, allow SGLang's original
cleanup to run, and require MPS postflight before another GPU test is admitted.
No sleep or Transport-drain delay is a correctness mechanism.

The E2E harness sends normal SIGTERM only to the `sglang serve` leader and waits
for the owned process group. Whole-group SIGTERM/SIGKILL is bounded fallback,
not the normal path. Multi-model servers close concurrently; all scheduler
ranks converge through their own detach and the existing generation-wide daemon
barriers.

The orderly-departure implementation is private to the SGLang plugin
composition root. `src/xpool/integrations/sglang/shutdown.py` is removed, and
its existing hook targets, timeout, process-local Event, control-flow exception
types, and `after_sigterm_handler()`, `around_scheduler_run_event_loop()`, and
`around_kill_process_tree()` implementations move into
`src/xpool/integrations/sglang/plugin.py`. The callable signatures remain
unchanged; the parent wrapper receives the corrected target, eligibility, and
best-effort cleanup behavior above. The removed module's `__all__` surface is
not preserved. Focused shutdown tests continue in
`tests/suites/integration/sglang/plugin/test_shutdown.py`, import and
monkeypatch the owning `plugin` module, and apply the wrapper to the real
`sglang.cli.serve` alias with the production `include_parent=False` call shape.

## Implementation Map

- `src/xpool/service/daemon/` and `src/xpool/runtime/` own Python control-plane
  authority and participant lifecycle.
- `src/xpool/fabric.py`, `src/xpool/transport.py`, and
  `src/xpool/service/wire.py` own Python domain and wire values.
- `src/xpool/integrations/sglang/` owns serving-engine integration and model
  adapters. `src/xpool/integrations/sglang/plugin.py` owns both plugin
  composition and its private pinned-SGLang orderly-departure hooks;
  `src/xpool/ops.py` owns the sole Tensor dispatcher facade.
- `src/cext-include/xpool/transport/` with `src/cext/transport/` owns CUDA IPC
  layout, protocol, resident, request, and trace behavior.
- `src/cext-include/xpool/fabric/` with `src/cext/fabric/` owns NVSHMEM layout,
  publication, scheduling, resident, lifecycle, and trace behavior.
- `src/cext-bindings/` owns pybind and dispatcher registration; core native
  implementation does not depend on pybind or `torch/library.h`.
- `src/xpool/native/*.pyi` is generated from the built extension and is the
  detailed Python binding contract.

Public-boundary documentation covers semantics that callers cannot derive from
a signature: state-machine meanings for exported domain enums, protocol-event
meanings for pybind trace values, and route-specific success and principal
failure behavior for daemon HTTP endpoints. It does not duplicate internal
control-plane branches or add ceremonial documentation to private helpers. The
generated native stub receives binding documentation from pybind definitions;
it is never edited by hand. Documentation must distinguish the CUDA-only native
submit boundary from the dispatcher fake path, list every accepted DP token
count dtype, and use current Request, Agent, and CTest terminology.

## Test Architecture

`python -m tests` is the canonical complete-suite composition root. Direct
pytest and CTest commands are focused debugging interfaces.

- `tests/suites/cext/` owns C++/CUDA value, protocol, layout, scheduler,
  resident, trace, and utility behavior.
- `tests/suites/unit/` owns deterministic Python behavior. It performs no
  native operation, CUDA initialization, subprocess launch, or weight access;
  the mandatory session native/dispatcher preflight still runs.
- `tests/suites/integration/` owns cross-module, pinned-SGLang, native binding,
  component CUDA subprocess, service, CLI, and process-management contracts.
- `tests/suites/e2e/` owns installed `xpool` and `sglang serve`, model weights,
  HTTP inference, graph evidence, observer traces, topology, token parity,
  multi-model concurrency, and shutdown.
- `tests/harness/` owns reusable Runner Framework, Component Drivers, and Test
  Support and does not import collected test modules. Its top-level packages are
  `runner/`, `native/`, `sglang/`, and `support/`; package initializers remain
  empty and do not re-export symbols. Runner is component-agnostic, Native and
  SGLang drivers are independent, and Support owns no canonical scheduling or
  process topology.

`tests/README.md` is the user-facing testing guide: it describes suite layers,
placement rules, requirements, artifacts, and canonical commands without
enumerating individual cases. `tests/harness/README.md` is the harness
architecture reference. It documents the collection-to-execution pipeline,
module boundaries, process tree and typed supervision protocol, GPU/MPS lease
ownership, endpoint-family reservations, result/artifact flow, and extension
rules. Superseded research notes are not retained as a second source of truth.

The accepted Runner Framework module layout is:

```text
tests/harness/runner/
  artifact.py
  bootstrap.py
  child.py
  collection.py
  ctest.py
  ctest_launcher.py
  gpu.py
  network.py
  plan.py
  process.py
  pytest_plugin.py
  pytest_report.py
  requirements.py
  results.py
  suite.py
  supervisor.py
  task.py
```

The old `test_plan.py`, `execution_task.py`, and `runner.py` become `plan.py`,
`task.py`, and `suite.py`. `tests/__main__.py` remains the sole composition
root. `process.py` owns direct subprocess and POSIX process-group primitives;
`child.py` renames `SpawnedProcess`, `SpawnedProcessStarted`, and
`SpawnedProcessFailure` to `PythonChildProcess`, `PythonChildStarted`, and
`PythonChildFailure`; `supervisor.py` owns the complete task-descendant domain.
Component Drivers may nest process groups and typed Python children inside a
Task Scope, but only the Task Scope can authorize release of a GPU lease.

`runner/artifact.py` removes the Runner-to-SGLang dependency through these
generic values:

```python
@dataclass(frozen=True, slots=True)
class ArtifactGroupRef:
    kind: str
    name: str
    expected_case_count: int


@dataclass(frozen=True, slots=True)
class ArtifactGroupResult:
    name: str
    result_code: int
    detail: str | None


class ArtifactGroupAdapter(Protocol):
    kind: str

    def evaluate(
        self,
        group: ArtifactGroupRef,
        reports: Sequence[PytestCaseReport],
        artifact_directories: Sequence[Path],
    ) -> ArtifactGroupResult: ...
```

Collection keeps the accepted `token_parity_group` marker but projects it to an
`ArtifactGroupRef(kind="token_parity", ...)`. `sglang/parity.py` implements
`TokenParityAdapter`; `tests/__main__.py` injects it through the new
`SuiteRunner(..., artifact_group_adapters: Sequence[ArtifactGroupAdapter])`
parameter. Missing or duplicate adapter kinds and invalid result codes are
infrastructure failures. Dynamic imports and global adapter registries are not
used.

The suite collects concrete pytest items into a typed plan, runs CTest before
Python stages, runs Unit before Integration, and admits E2E only after
Integration succeeds. GPU tasks are sorted by required GPU count and estimated
duration, then backfilled over the idle pool. The complete MPS pool-usability
probe, the complete CTest stage, and every pytest GPU task each run in a
`SupervisedTaskScope`; CTest retains its internal resource-spec scheduling
inside that one stage scope. A task-local GPU lease is returned only after its
scope reaches `CLOSED`. An unproven descendant domain retains its in-run lease
and fails closed even when `SuiteRunner` was never created.

The runner-to-supervisor control pipe is also the parent-liveness boundary. If
the runner exits abruptly, EOF makes the dedicated supervisor drain its task
root and all adopted descendants before the runner-level subreaper reaps the
supervisor. This preserves the same empty-domain invariant for graceful
cancellation, timeout, and parent death.

`TaskCompletion` also represents a Supervisor-local infrastructure failure when
the Supervisor has nevertheless drained the task domain and proved it empty.
That task contributes suite exit code 2, stops pending admission, drains every
active scope, and terminates the Test Run; infrastructure failure never permits
unrelated same-stage work to continue.
`TaskStartFailure` represents startup rollback that proved the attempted domain
empty. `TaskSupervisionFailure` represents loss of the normal Supervisor
protocol and triggers runner-wide fallback; it is not yet a final emptiness
result. `TaskScopeFailure` is reserved for fallback that still cannot prove the
descendant domain empty or close its Supervisor, and only that terminal failure
quarantines the affected in-run GPU lease until the failing test invocation
terminates.

Successful runner fallback advances failed scopes to `DRAINED` and returns
normally without manufacturing task completions. Concurrent `SuiteRunner`
closes those scopes, releases their leases, stops scheduling, and exits with
infrastructure code 2. `SupervisedTaskScope.run()` owns the complete serial
start, wait, fallback, and close lifecycle for MPS pool usability and CTest. It
returns `INFRASTRUCTURE_FAILED` after resource-safe startup rollback or a
recovered supervision failure, and otherwise returns only after the scope is
`CLOSED`.

Suite result code `0` means every executed case passed or legally skipped.
Code `1` is reserved for ordinary behavioral failure represented by a complete,
trusted JUnit report. `TIMED_OUT`, `LEAKED`, `INFRASTRUCTURE_FAILED`, missing or
inconsistent JUnit, and resource-lifecycle failure all produce code `2`.
Ordinary code `1` completes the current stage's pending and active tasks but
prevents admission of the next stage. Code `2` immediately stops admission and
drains active scopes. CTest uses the same distinction between a reported native
test failure and an infrastructure/report failure.

Concurrent task launch performs MPS checks, directory creation, and command
construction before acquiring a task-local GPU lease. Lease acquisition and
`SupervisedTaskScope.start()` are the sole launch transaction:
`TaskStartFailure` returns the lease, while `TaskScopeFailure` quarantines it.
The final resource-release proof also requires `GpuPool.active_leases` to be
empty so runner bookkeeping cannot claim the pool closed while a lease is still
outstanding.
Supervisors are isolated from terminal signals and normal cancellation uses the
typed control pipe. A signal sent directly to a Supervisor is owner loss, not a
local cleanup request; the runner subreaper performs emergency cleanup and stops
the suite.

The GPU pool is derived only from startup `CUDA_VISIBLE_DEVICES` and normalized
to physical UUIDs. The environment guarantees exclusive ownership of that
visible set; xpool does not coordinate independent test invocations, other
users, or containers. Runner leases those GPUs only among tasks in one
invocation. Every selected GPU must pass MPS preflight.
`CUDA_MPS_PIPE_DIRECTORY` is mandatory and identifies the
externally managed controller; xpool does not fall back to NVIDIA's shared
`/tmp/nvidia-mps` endpoint. Queries to the same explicit controller pipe are
serialized across processes because its control client does not support
concurrent commands. Each SGLang process receives an owned endpoint family,
including HTTP, NCCL, gRPC, and DP-derived ports when needed.

`tests/harness/sglang/manifest.toml` is the sole source for E2E model IDs,
topologies, graph modes, FfnAgent/Executor counts, observer capacities,
estimates, timeouts, and test-only KV limits. `XPOOL_CONFIG` supplies the
external model root and system configuration but does not select test models;
each task materializes a local config containing only the models selected by
its manifest case. `model_serving_cases` exercise the complete serving contract
through FfnAgent loopback until Phase 8; separate
`loopback_serving_cases` sweep all explicit debug sites. Cross-mode token
parity is evaluated only for complete declared groups. Every successful HTTP
attempt retains a versionless `*.inference.json` record containing the exact
model ID, URL, request body, response status/content type, and decoded response;
the record is written before status and token-shape validation so failed
inference remains inspectable.

The manifest is a workload catalog, not a duplicate adapter registry. It does
not repeat production capability fields such as `supports_dp_attention`.
Integration tests prove each adapter capability and E2E cases prove it through
the installed service path. Manifest validation owns unique and resolvable
identities, homogeneous colocated Attention topology, rejection of combined
TP-by-DP, rejection of piecewise mode under DPA for pinned SGLang, and GPU-count
derivation. Executor count may differ from FfnAgent count. Qwen3-30B-A3B remains
DPA-capable and retains its accepted `(TP, DP)=(1,2)` cases.

E2E startup treats daemon health and agent registration as distinct phases.
Daemon HTTP readiness has a 60-second deadline and records its measured startup
duration in the case artifact; agent registration retains a separate 30-second
deadline so a slow daemon cannot consume the participant-registration budget.
Every readiness phase writes a versionless bounded final observation under
`attempt-N/readiness/`: `daemon-health.json`, `agent-registration.json`, one
`sglang-<model-slug>-health.json` per server, and `system-readiness.json`.
Evidence contains its name and URL, attempt count, elapsed seconds, final
transport error type/message, final HTTP status and bounded response excerpt,
and relevant process return codes. It never retains every poll. Successful,
timed-out, and early-process-exit waits all write evidence. Every terminal
exception from `SglangServerProcess.healthy()` is recorded in that server's
evidence before the same exception object is raised, including early process
exit and a TCPStore startup blocker. Server evidence owns the concrete service
failure; system evidence may retain the propagated exception as the overall
barrier summary. Timeout raises `ReadinessTimeout` carrying the same
`ReadinessEvidence` value.

`tests/harness/sglang/readiness.py` defines:

```python
@dataclass(slots=True)
class ReadinessEvidence:
    name: str
    url: str
    attempt_count: int
    elapsed_seconds: float
    last_error_type: str | None
    last_error_message: str | None
    last_status_code: int | None
    last_response_excerpt: str | None
    process_statuses: dict[str, int | None]

    def record_error(self, error: BaseException) -> None: ...
    def record_response(self, response: httpx.Response) -> None: ...
    def finish(
        self,
        *,
        elapsed_seconds: float,
        processes: Sequence[OwnedProcessGroup],
    ) -> None: ...
    def write(self, path: Path) -> None: ...


class ReadinessTimeout(RuntimeError):
    evidence: ReadinessEvidence
```

The accepted Component Driver and Support layout is:

```text
tests/harness/native/
  case.py
  debug.py
  fabric/{bootstrap,instance,participant,protocol,topology,trace}.py
  transport/{owner,protocol,trace}.py
tests/harness/sglang/
  attempt.py
  cluster.py
  endpoints.py
  graph.py
  launch.py
  manifest.py
  manifest.toml
  parity.py
  probe.py
  readiness.py
  server.py
tests/harness/support/
  config.py
  devkit.py
  process_probe.py
  wait.py
  native/{fabric,loopback,transport}.py
  runtime/{atnagent,instance}.py
  service/{client,daemon}.py
  sglang/{deepseek,fakes,graph,graph_observer,observer,plugin}.py
```

Protocol modules contain only cross-process wire values; child modules own
entrypoints; trace modules project bound values; topology/attempt objects own
complete scenarios. Assertions, fixtures, fakes, and builders live in Support
and are imported explicitly by tests. Every process-owning driver borrows a
caller-owned `workdir: Path`; native drivers remove self-deleting temporary log
directories. Logs use deterministic role-plus-ordinal names and remain in the
runner-retained task tree.

The Native Driver API changes are additive only at the call signature and are
applied without compatibility aliases because the repository is greenfield:

```python
def run_native_case(
    target: Callable[..., None],
    *arguments: object,
    workdir: Path,
    timeout_seconds: float = NATIVE_CASE_TIMEOUT_SECONDS,
) -> None: ...

def create_fabric_uid(*, workdir: Path) -> FabricUid: ...
def fabric_bootstrap(*, workdir: Path) -> ContextManager[FabricUid]: ...

def run_fabric_topology(
    uid: FabricUid,
    *,
    workdir: Path,
    atnagent_count: int,
    ffnagent_count: int,
    executor_count: int,
    forward_modes: tuple[XPoolForwardMode, ...],
    dtype: TensorDType,
    quiesce_after_first_request: bool = False,
    repetition_count: int = 3,
    loopback_enabled: bool = True,
    expect_activation_rejection: bool = False,
) -> FabricTopologyReport: ...
```

`controlled_atnagent_arena_process()` and `atnagent_arena_process()` likewise
gain required keyword-only `workdir: Path`; their remaining parameters and
yielded controller/handle semantics do not change. Test-only factories pass
their `tmp_path` into these drivers rather than manufacturing storage.

### Fabric Concurrency Evidence Ownership

Scheduler capacity, distributed protocol correctness, and serving concurrency
are separate claims with separate owners:

- `tests/suites/cext/fabric/scheduler_test.cu::LeasesDistinctExecutorsUntilOneIsReleased`
  deterministically owns Scheduler capacity. It publishes multiple Ready
  models before scheduling and proves simultaneous leases use distinct
  Executors until one lease is released.
- `tests/suites/integration/native/test_fabric_topology.py` owns distributed
  request completion and per-Executor lease exclusivity. The test symbol is
  renamed from
  `test_fabric_topology_executes_concurrent_requests_with_exclusive_executor_leases`
  to `test_fabric_topology_executes_requests_with_exclusive_executor_leases`.
  It removes `assert_cross_model_executor_overlap()`, the associated
  `itertools.combinations` import, and the assertion that observed Executor
  identities equal `set(range(executor_count))`. Reuse of Executor zero after
  release is valid and does not imply lost concurrency capability.
- `tests/harness/support/native/fabric.py::assert_fabric_report()` uses the
  existing `FabricTopologyReport.executor_count` to require every completed
  Coordinator trace to carry an Executor index in `[0, executor_count)`. This
  is geometry validation, not Executor-coverage evidence.
- `tests/suites/e2e/sglang/test_e2e_model_serving.py` continues to own realized
  cross-model concurrency. Its accepted two-model case submits model requests
  concurrently and
  `tests/harness/support/sglang/observer.py::assert_two_model_executor_overlap()`
  requires at least one pair of overlapping active intervals on distinct
  Executors. The predicate does not require a fixed overlap count. On failure,
  the helper reports each model's valid interval count and Executor identities,
  the cross-model candidate-pair count, and the maximum overlap duration; it
  does not dump individual intervals.

No test-only device gate, sleep, retry loop, synthetic oversized workload,
trace field, observer schema, report field, runtime Scheduler policy, native
ABI, or production protocol change is introduced by this repair.

`tests/harness/support/sglang/fakes.py` removes the one-to-one
`type ModelModule = nn.Module` alias. `loaded_model()` changes its type parameter
bound from `[M: ModelModule]` to `[M: nn.Module]`; its arguments, generic return
relationship, and runtime behavior are unchanged.

Repair validation runs, in order:

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
uv run pytest tests/suites/integration/sglang/plugin/test_shutdown.py
uv run pytest tests/suites/integration/native/test_fabric_topology.py
uv run python -m tests --suite cext
uv run pre-commit run --all-files
```

The final pre-commit invocation is run exactly once. It owns the complete suite,
including the two-model E2E overlap gate; no separate E2E run or repeated
topology stress loop precedes it. A newly exposed design defect stops validation
for diagnosis instead of being hidden by retries.

### Accepted Harness Ownership And Endpoint Qualification

This subsection is accepted implementation work, not current validation
evidence. A successful local bind is insufficient endpoint evidence: the
current host contains at least one address that can bind and listen while every
local connection is rejected. A live reservation therefore owns an address
that completed `bind -> listen -> client connect -> listener accept`; the probe
connection is closed and the original listener remains held until spawn
handoff. No Kubernetes or other host-policy port range is hard-coded.

`tests/harness/runner/network.py` defines:

```python
class TcpEndpointUnreachable(RuntimeError):
    address: tuple[str, int]

    def __init__(self, address: tuple[str, int]) -> None: ...


class TcpEndpointAllocationError(RuntimeError):
    host: str
    collision_count: int
    unreachable_count: int

    def __init__(
        self,
        *,
        host: str,
        collision_count: int,
        unreachable_count: int,
    ) -> None: ...


class TcpEndpointConflict(RuntimeError):
    addresses: tuple[tuple[str, int], ...]

    def __init__(self, addresses: tuple[tuple[str, int], ...]) -> None: ...


class TcpEndpointReservation:
    @classmethod
    def reserve(cls, host: str, *, port_space: TcpPortSpace) -> Self: ...

    @classmethod
    def reserve_exact(cls, host: str, port: int) -> Self: ...

    def release_for_spawn(self) -> None: ...
    def reacquire(self) -> None: ...
    def close(self) -> None: ...
```

`reserve_exact()` closes its listener and raises `TcpEndpointUnreachable` from
the concrete socket failure when active qualification fails. `reserve()` skips
only `EADDRINUSE` and `TcpEndpointUnreachable`; all other socket failures
escape. Finite traversal exhaustion raises `TcpEndpointAllocationError` with
aggregate counts and the final candidate error as its cause. It never retains
an unbounded rejected-port list. `reacquire()` restores the same qualified
reservation invariant. Post-cleanup `EADDRINUSE` remains conflict evidence;
post-cleanup unreachability is an infrastructure failure, not occupation.

`SglangEndpointFamilyLease.acquire()` qualifies every HTTP, NCCL, gRPC,
handshake, and ZMQ member while all listeners and abstract namespace locks are
held. One unreachable derived member rejects the whole candidate family.
`reacquire_tcp() -> tuple[int, ...]` aggregates only occupied ports in family
order and propagates endpoint unreachability and unexpected socket failures.
The pinned CLI still cannot inherit listeners, so real listeners close at the
spawn handoff while namespace locks continue coordinating xpool runners.

`tests/harness/sglang/attempt.py` replaces free cleanup coordination with:

```python
@dataclass(slots=True)
class ProbeAttempt:
    manifest: E2eManifest
    case: E2eServingCase
    base_config: XpoolConfig
    graph_settings: SglangGraphSettings
    workdir: Path
    loopback_site: LoopbackSite
    daemon_endpoint: TcpEndpointReservation | None = None
    server_endpoints: list[SglangEndpointFamilyLease] = field(default_factory=list)
    cluster: XpoolCluster | None = None
    servers: list[SglangServerProcess] = field(default_factory=list)
    closed: bool = False

    def run(self) -> ProbeRun: ...
    def diagnostics(self) -> str: ...
    def close_processes(self) -> tuple[str, ...]: ...
    def classify_released_endpoints(self) -> tuple[tuple[str, int], ...]: ...
    def close_endpoints(self) -> None: ...
    def close(self) -> None: ...
```

The attempt owns endpoint allocation, launch materialization, cluster and
server startup, readiness, concurrent inference, diagnostics, process cleanup,
post-cleanup endpoint classification, and final lease release in that order.
No duplicate AttemptState enum is added. Process cleanup must succeed before
occupation inspection; endpoints close last from `run()`'s outer `finally`,
which also sets `closed=True`. `run_probe()` creates a fresh attempt, does not
mutate its state or close its endpoints, and retries at most three times only
for a startup `TcpEndpointConflict` proven after cleanup. Occupation after
inference or success, allocation exhaustion, readiness timeout, process exit,
invalid response, inference failure, cleanup failure, and endpoint
unreachability are not retryable. When classification itself fails, diagnostics
preserve both the primary startup/inference error and the cleanup error. Log
patterns may break a stuck startup but never authorize retry. `close()` remains
an idempotent emergency fallback and does not manufacture classification
evidence.

Focused Unit coverage proves active qualification, listener cleanup, finite
candidate traversal, strong reacquisition, complete family qualification,
readiness evidence, attempt cleanup ordering, retry classification, failure
codes, same-stage cancellation, and generic artifact-adapter dispatch.
Integration coverage proves real parent-to-child endpoint handoff and existing
process/supervisor contracts. The host-specific rejected port is diagnostic
evidence, not a committed constant or race-dependent E2E case.

Resource requirements use `requires_cuda`, `requires_config`, `requires_mps`,
and `requires_model_weights`. Direct pytest skips unavailable resources unless
strict mode is requested; invalid explicit configuration always fails. The
canonical suite retains logs, JUnit, parity artifacts, and observer evidence
under `.xpool-cache/test-runs/`. `XPOOL_TEST_KEEP_RUNS`, when set to a positive
integer, bounds recognized run-directory retention, including interrupted runs;
per-run locks protect concurrent active runs, while unset retention never
removes historical results.

Arena allocations use a 256-byte aligned root, but each internal region carries
its actual requirement: typed metadata and state use `alignof(T)`, cooperative
hidden-state payloads use 16-byte alignment, and remotely signaled Fabric
publications retain their explicit 256-byte alignment. ABI 54 is the first ABI
with this per-region geometry.

### Harness Refactor Implementation Record

Implementation slices 1 through 7 are complete in the current worktree:

1. synchronize this canonical design and `tests/harness/README.md`;
2. mechanically move Runner, Driver, and Support modules and update imports
   without behavioral changes;
3. split Python child ownership, implement canonical failure codes and
   same-stage policy, and introduce generic artifact adapters;
4. implement active endpoint qualification, readiness evidence, and the
   `ProbeAttempt` owner;
5. decompose Native Fabric/Transport drivers and replace temporary logs with
   caller-owned workdirs;
6. finish explicit fixture imports and manifest-owned topology validation;
7. run focused affected suites, then invoke
   `uv run pre-commit run --all-files` exactly once as the complete gate.

No compatibility import aliases or forwarding wrappers preserve old harness
paths. Mechanical moves use `git mv` where a file remains conceptually intact.
A verification failure caused by a new design defect stops implementation for
review; retries and local workarounds must not conceal it.

### Staged-Review Remediation Record

This bounded remediation is implemented and validated. It changes no
runtime config schema, daemon HTTP or Python wire value, native ABI or binding,
Transport/Fabric protocol state, trace schema, E2E manifest schema, model
adapter, graph policy, or FFN request data path. Greenfield policy applies: old
private harness names are removed without compatibility aliases.

#### Interface And Data-Structure Changes

`src/xpool/integrations/sglang/plugin.py` changes the required hook target from
`sglang.srt.utils.common.kill_process_tree` to the actual installed call surface
`sglang.cli.serve.kill_process_tree`. These callable signatures remain:

```python
def around_kill_process_tree(
    original_fn: Callable[[int | None, bool, int | None, float | None], None],
    parent_pid: int | None,
    include_parent: bool = True,
    skip_pid: int | None = None,
    wait_timeout: float | None = None,
) -> None: ...

def around_scheduler_run_event_loop(
    original_fn: Callable[[Scheduler], None],
    scheduler: Scheduler,
) -> None: ...
```

The parent wrapper's eligibility changes from marker plus current parent plus
`include_parent=True` to marker plus current parent only. Scheduler drain is
best-effort and the original SGLang cleanup is called from `finally` with every
argument preserved. Required-hook verification names the serve alias directly.

`tests/harness/sglang/attempt.py` renames only this private helper:

```python
# old
def reacquire_released_endpoints(self) -> tuple[tuple[str, int], ...]: ...

# new
def classify_released_endpoints(self) -> tuple[tuple[str, int], ...]: ...
```

`ProbeAttempt.run() -> ProbeRun`, `close_processes() -> tuple[str, ...]`,
`close_endpoints() -> None`, and `close() -> None` keep their signatures.
`run()` owns process cleanup, successful-cleanup classification, endpoint
release, and `closed=True` on every terminal path. `run_probe()` no longer
closes endpoints or mutates Attempt state. `close()` remains idempotent
emergency cleanup and does not classify. No Attempt state enum, termination
result, aggregate cleanup value, or wire type is added.

`tests/harness/sglang/server.py` keeps this signature:

```python
class SglangServerProcess:
    def healthy(
        self,
        evidence: ReadinessEvidence | None = None,
    ) -> bool: ...
```

Early process exit and TCPStore blocker errors are recorded in the supplied
server evidence before the same exception object is raised. The
`ReadinessEvidence` fields, field order, serialization, and method signatures
do not change.

`tests/harness/native/fabric/topology.py::run_fabric_topology()` keeps the
complete signature recorded above and its `FabricTopologyReport` shape. Its local
`participants` and `instances` lists are built incrementally inside one
topology-owned `try/finally`. `PythonChildProcess.start()` owns rollback of the
currently failing child; topology cleanup terminates and closes every sibling
that started earlier. No native operation or Fabric protocol value changes.

`tests/harness/runner/gpu.py` removes the cross-invocation lock surface:

```python
# removed
GPU_LOCK_DIRECTORY: Path
def acquire_gpu_locks(uuids: tuple[str, ...]) -> tuple[TextIO, ...]: ...

class GpuPool:
    # old
    def __init__(
        self,
        uuids: tuple[str, ...],
        lock_files: tuple[TextIO, ...],
    ) -> None: ...

    # new
    def __init__(self, uuids: tuple[str, ...]) -> None: ...
```

The private `GpuPool.lock_files` field is removed. `uuids`,
`available_uuids`, `active_leases`, and `closed` retain their types and
meanings. `from_environment()`, `try_lease()`, `release()`, and `close()` keep
their signatures; `close()` only rejects active leases and marks the pool
closed. Runner diagnostics no longer claim that OS-level GPU locks are retained.

The fixture moves without behavior change:

```python
# old owner: tests/harness/runner/pytest_plugin.py
# new owner: tests/harness/support/config.py
def reset_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]: ...
```

It continues to clear `xpool.config.global_config`,
`xpool.bootstrap.runtime_role`, and
`xpool.bootstrap.runtime_cuda_device`. Root `tests/conftest.py` activates only
the Runner plugin. Consuming test modules explicitly activate Support Config;
no directory-level implicit fixture import is added.

#### Failure Precedence And Terminal Ownership

- SGLang parent drain failures are logged; the original SGLang cleanup always
  runs and its own exception propagates unchanged.
- Probe process-cleanup failure is primary and suppresses endpoint
  classification, but endpoint leases are still closed. Classification runs
  only after cleanup succeeds.
- Startup plus confirmed occupation raises retryable `TcpEndpointConflict`.
  Occupation after inference or success is an infrastructure cleanup failure.
  Unexpected classification errors are non-retryable and diagnostics preserve
  both an existing startup/inference error and the cleanup error.
- Server-specific readiness evidence owns early-exit or blocker details;
  system evidence may record the propagated failure as the barrier summary.
- Native topology startup failure remains the primary exception after all
  successfully started siblings are terminated and closed.
- `TaskScopeFailure` still fail-stops one test invocation when descendant
  cleanup is unproven; it no longer claims to quarantine an OS-level GPU lock.

#### Documentation And Test Changes

- `AGENTS.md` records that the manifest owns E2E model IDs, placement, and graph
  modes; `XPOOL_CONFIG` supplies the external model root and system settings;
  each task materializes only its selected models.
- `tests/harness/README.md` corrects composition ownership, artifact ownership,
  infrastructure cancellation, explicit visible-GPU exclusivity, and in-run
  leasing. Affected lifecycle docstrings describe non-obvious mutation and
  failure behavior.
- Delete only the whole-run GPU-lock conflict test. Replace the parent shutdown
  wrapper test with real serve-alias production-call coverage. Update existing
  readiness tests for terminal evidence, rename endpoint-classification helper
  tests, add one parameterized Attempt terminalization test for startup,
  inference, and success, and add one pure-Python Native Fabric partial-start
  rollback Unit test.
- Do not add tests for source location, docstring text, or prose. Existing
  pytester-generated `test_example` strings and all other audited tests retain
  distinct behavior contracts.

#### Implementation And Validation Record

1. Repair the SGLang serve hook and focused shutdown Integration coverage.
2. Repair Probe terminalization and readiness evidence with focused Unit
   coverage.
3. Make Native Fabric participant startup transactional and add the pure-Python
   rollback Unit case.
4. Move the config fixture, remove whole-run GPU locks, update explicit fixture
   activation, and synchronize diagnostics/docs.
5. Run formatting, lint, and type checks over changed files; run affected Unit
   and shutdown Integration selectors.
6. Run `uv run python -m tests --suite unit` and
   `uv run python -m tests --suite integration`.
7. Run the representative Qwen3-14B TP1/DP1/F1/E1 manifest case, including its
   complete eager/full/piecewise parity group:
   `uv run python -m tests --suite e2e -k qwen3-14b-tp1-dp1-f1-e1`.
   Escalate to the complete E2E matrix only if focused validation fails or final
   review finds a model-, topology-, or graph-mode-dependent impact.
8. Review implementation, tests, this document, and
   `tests/harness/README.md` against this section. The review closed without an
   unresolved finding, so Phases 6 and 7 are Implemented.

## Current Verification

The 2026-07-27 loopback milestone passed the canonical build, complete
`python -m tests`, and the non-duplicated formatting, lint, type, Doxygen,
native formatting, and pre-commit non-test quality gates. Exact test counts are
runtime collection facts and are intentionally not copied into this
architecture document.

That verification covers DeepSeek-V2-Lite, dense Qwen3, GLM-4.7-Flash, and
Qwen3-30B-A3B. The canonical suite found every declared requirement available,
ran every E2E task without skips, and compared every token-parity artifact
group successfully.

The harness decomposition, active endpoint qualification, readiness artifacts,
generic artifact adapter, revised runner failure policy, and staged-review
remediation are implemented in the current worktree. On 2026-07-29 the repair
gate passed Ruff, ty, focused Unit and Integration selectors, the canonical
Unit and Integration suites, and the complete eager/full/piecewise parity group
for `qwen3-14b-tp1-dp1-f1-e1`. Final implementation, test, and documentation
review found no unresolved lifecycle, ownership, model, topology, or graph-mode
finding, so the complete E2E matrix was not repeated.

## Phase 8: Real FFN Execution

Phase 8 is blocked. Before implementation, a new design pass must decide:

1. who loads weights, where each Dense/MoE layer is placed, and how layer
   implementation identity is bound;
2. the relationship among FfnAgent count, Executor concurrency, FFN tensor
   parallelism, and expert parallelism;
3. real Decode and Prefill kernels, workspace ownership, CUDA graph capture,
   and reuse of the existing Fabric transfer protocol;
4. output contribution semantics, cross-FfnAgent collectives or reduction, and
   rank-local Result Handoff;
5. weight and temporary-memory lifecycle, multi-model isolation, failure
   propagation, quiesce, drain, and shutdown;
6. DeepSeek-V2, dense Qwen3, GLM-4.7-Flash, and Qwen3-MoE numerical oracles
   across supported TP/DP and graph modes, including concurrent two-model
   execution; and
7. performance acceptance criteria and profiler evidence.

The accepted design must be merged into this document with exact new and old
interfaces, data structures, ownership, lifecycle, failure behavior, and
compatibility policy before source implementation begins. Until then,
`src/cext/ffnagent/executor.cu` is an execution extension boundary with debug
loopback only, not a production FFN body. Its current contract writes a valid
output only when FfnAgent loopback returns `FfnResultCode::Ok`;
`ProtocolMismatch` reports invalid Invocation or layer facts, and
`NotImplemented` reports an unsupported execution policy or geometry. The
native declaration and `FfnShimModule.forward` documentation must state this
current capability rather than implying that Dense or MoE weights are
executed. Device-side failure does not synchronize and raise at the Python
call: it publishes sticky canonical failure, poisons the asynchronous output
with NaNs, and is converted into Instance process fail-stop by the native
failure monitor.

## Completed Phase 4 And Phase 6 Remediation

The final R10-R19 review closed the following bounded implementation work.
Phases 4 and 6 are `Implemented`; the remediation changed no wire schema, native
memory layout, public command spelling, or serving topology.

### Daemon Watchdog Fail-Stop

`src/xpool/service/daemon/app.py` keeps
`create_daemon() -> FastAPI` unchanged and adds the module-internal value:

```python
@dataclass(slots=True)
class DaemonFailure:
    exception: BaseException | None = None

    @property
    def failed(self) -> bool: ...

    def record(self, exception: BaseException) -> None: ...
```

`record()` retains only the first non-cancellation exception. The application
stores one instance as `app.state.daemon_failure`. Its lifespan watchdog
re-raises `asyncio.CancelledError`, logs every other exception with traceback,
records it, and returns. It does not restart the watchdog or mutate readiness.

`src/xpool/cli/subcommands/daemon.py` replaces the convenience
`uvicorn.run(app, ...)` call with the private adapter:

```python
class DaemonServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, failure: DaemonFailure) -> None: ...
    async def on_tick(self, counter: int) -> bool: ...
```

`on_tick()` returns `await super().on_tick(counter) or failure.failed`, thereby
using Uvicorn's ordinary shutdown path. `DaemonServeCommand.run(...) -> int`
still owns server construction and returns 1 after a latched watchdog failure,
or 0 after ordinary shutdown. Neither type is exported from
`xpool.service.daemon`; no signal, `os._exit`, restart loop, or production test
injection parameter is added.

Unit coverage in `tests/suites/unit/service/daemon/test_app.py` proves first-error
latching and application-state ownership. A new
`tests/suites/integration/service/daemon/test_watchdog.py` starts the real
Uvicorn server in a child after monkeypatching `ControlPlane.watchdog`, observes
initial HTTP responsiveness, and proves bounded nonzero process exit after the
next watchdog call fails.

### Supervised Task And GPU Ownership

`tests/harness/runner/supervisor.py` adds two parent-side exception types while keeping
`TaskScopeState`, typed Pipe messages, and `TaskCompletion` layouts unchanged:

```python
class TaskStartFailure(RuntimeError): ...
class TaskSupervisionFailure(RuntimeError): ...

@classmethod
def run(
    cls,
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> TaskCompletion: ...
```

`start()` retains its signature. A failed start whose rollback proves emptiness
raises `TaskStartFailure`; rollback that remains unproven raises
`TaskScopeFailure`. After successful start, Pipe loss, premature Supervisor
exit, invalid messages, and a Supervisor failure message raise
`TaskSupervisionFailure` and move the scope to `FAILED`. A Supervisor-local
exception may instead publish `TaskCompletion(INFRASTRUCTURE_FAILED)` only when
its own cleanup proved the domain empty.

`terminate_all(scopes: Sequence[SupervisedTaskScope]) -> None` retains its
signature and fans out normal cancellation before runner fallback. Successful
fallback reaps Supervisors, drains runner-adopted descendants, advances failed
scopes to `DRAINED`, and returns normally. Only fallback that cannot prove
emptiness or reap a Supervisor raises `TaskScopeFailure`. `run()` composes
start, wait, fallback, and close: resource-safe startup rollback or recovered
supervision failure returns `INFRASTRUCTURE_FAILED`; terminal unproven cleanup
propagates `TaskScopeFailure`; every live scope is `CLOSED` before return.

`tests/harness/runner/suite.py::SuiteRunner.start_task()` creates directories and the
command before acquiring a GPU lease. Lease acquisition immediately precedes
`SupervisedTaskScope.start()`: `TaskStartFailure` releases it,
`TaskScopeFailure` appends it to `retained_leases`, and successful start installs
the scope in `active`. A later `TaskSupervisionFailure` stops stage scheduling,
drains all active scopes concurrently, releases every successfully closed
lease, and contributes infrastructure exit code 2. The
`resources_releasable` property additionally requires
`gpu_pool.active_leases` to be empty.

`tests/__main__.py::execute_test_run()` replaces the `runner is None` proxy with
one explicit run-level resource-release fact. It remains true for ordinary
failure, `TaskStartFailure`, and recovered `TaskSupervisionFailure`, but becomes
false on `TaskScopeFailure`. `GpuPool.close()` runs only when that fact and the
runner/pool lease proofs all hold.

Unit and Integration Supervisor tests cover all three exception levels,
Supervisor-local infrastructure completion, recovered fallback, terminal
fallback failure, startup lease release, startup lease quarantine, and the
pool-backed final release proof. Existing signal isolation, subreaper adoption,
timeout, leak, TERM/KILL escalation, and concurrent cancellation tests remain.

### Serial MPS And CTest Scopes

`src/xpool/service/daemon/mps.py` removes `MPS_DEFAULT_PIPE_DIRECTORY` and
changes the private lock helper from `mps_probe_lock_path() -> Path` to:

```python
def mps_probe_lock_path(pipe_directory: Path) -> Path: ...
```

`probe_mps_controller() -> MpsProbeResult` keeps its public signature but first
requires a nonempty `CUDA_MPS_PIPE_DIRECTORY`, returning an offline diagnostic
when absent. It expands and resolves that explicit directory, derives the
per-user serialization lock from it, and invokes the controller with the
process environment. `CUDA_MPS_LOG_DIRECTORY` remains an external deployment
requirement rather than controller identity. Unit tests cover missing/empty
configuration, explicit lock identity, serialization, invalid output, timeout,
and controller failure.

`tests/__main__.py::prove_gpu_pool()` replaces manual `start()`/`wait()`/`close()`
with `SupervisedTaskScope.run(...)`.
`tests/harness/runner/ctest.py::CtestSuite.run()`
retains its signature and result type but launches its complete CTest command
through one serial scope; CTest's resource specification and internal parallel
scheduler remain unchanged. An ordinary nonzero CTest exit with JUnit is code
1; timeout, leak, Supervisor infrastructure completion, missing JUnit, or the
existing infrastructure sentinel is code 2. A terminal `TaskScopeFailure`
escapes to the composition root without claiming that its GPU resources are
releasable. CTest unit tests
mock the scope result rather than `subprocess.run`, and Integration supervision
proves descendant cleanup around one synthetic serial command.

### Native Trace Boundary And Documentation

`src/cext-bindings/fabric.cpp` keeps the three overloaded Python signatures for
each of `FabricTraceRecord.recorded(event) -> bool` and
`FabricTraceRecord.timestamp(event) -> int`. Each binding adapter first checks
that the event enum family matches `record.kind`; mismatch raises
`pybind11::value_error` before calling the existing fail-stop native member.
The native record, enum values, variant, memory layout, and observer JSON remain
unchanged. A dedicated Native Integration case obtains a real trace snapshot in
its participant process and proves every mismatched event family raises
`ValueError` without aborting; it adds no public record constructor or test-only
native binding.

Documentation-only correction touches the owning declarations and bindings:

- `src/cext-include/xpool/{fabric,transport}/trace.hpp` and
  `src/cext-bindings/{fabric,transport}.cpp` define sender `Published` as a local
  publication edge or completed issue of all fan-out calls; only receiver
  `Observed` proves protocol observation. Transport sender events remain causal
  markers immediately before release-store, not completed remote transitions.
- `src/xpool/integrations/sglang/shim.py::FfnShimModule.forward` documents
  asynchronous sticky device failure, NaN poison, and failure-monitor process
  exit instead of promising synchronous `RuntimeError` for an unimplemented
  executor.
- daemon route docstrings state their actual 204 success responses and current
  404/409/503 mappings; Transport declarations state `int32 | int64` DP counts,
  CUDA-only native submit, and local-loopback admission behavior.
- stale descriptor, devagent, two-loopback-site, and csrc wording is replaced
  with Request, Agent, all reachable loopback sites, and CTest/native-test
  terminology. The random Scheduler binding states its nonzero-seed
  precondition.

Generated `xpool.native` stubs are regenerated from the corrected pybind
surface; no `.pyi` file is edited manually. R16 makes no source or test change:
model adapters remain architecture-selected and current manifest model IDs
remain E2E evidence identities rather than production allowlists.

### Remediation Validation

Development uses focused Unit and Integration selectors for the affected daemon,
Supervisor, CTest, MPS, Fabric binding, and shim-failure boundaries. Native
binding changes then rebuild through the canonical uv/scikit-build command.
Closure was validated by one complete `uv run python -m tests` run followed by
Ruff format/check, ty, and Doxygen. The complete suite was not repeated through
pre-commit. Phase 8 remains blocked.

## Validation

Build through the canonical uv/scikit-build path:

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
```

Load optional local test configuration before repository commands:

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
```

Run focused checks while developing. The complete milestone gate is:

```bash
uv run python -m tests
uv run ruff format --check
uv run ruff check
uv run ty check
doxygen Doxyfile
```

Do not immediately run the complete suite a second time through a redundant
pre-commit invocation. Routine commits use the installed hooks.
