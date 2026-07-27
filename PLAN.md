# xpool Architecture And Plan

This document is the canonical architecture and implementation plan for xpool.
It describes the current supported design and the only remaining feature gate.
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
| 5. SGLang integration | Implemented | DeepSeek-V2 and Qwen3 adapters, workload derivation, topology checks, and eager/full/piecewise graph-safe shim paths. | `src/xpool/integrations/sglang/` |
| 6. Test orchestration | Implemented | Resource-aware CTest/pytest scheduling, descendant supervision, GPU lease proof, and explicit MPS-pipe ownership. | `tests/__main__.py`, `tests/harness/` |
| 7. Loopback evidence | Implemented | Installed-command serving, graph, topology, observer, parity, multi-model, and shutdown evidence with debug loopback. | `tests/suites/e2e/`, `tests/harness/sglang/manifest.toml` |
| 8. Real FFN execution | Blocked | No production weight loading or FFN calculation is claimed. | `src/cext/ffnagent/executor.cu` |

Implemented status requires both the code and its validation gate. Symbol
presence alone is not evidence. Phase 8 remains blocked until its independent
design is accepted and merged into this document.

## Mission And Boundaries

xpool separates attention-side serving from FFN execution within one host.
SGLang owns request scheduling, attention, KV cache, graph selection, and output
postprocessing. xpool intercepts supported FFN calls, transports rank-local
requests through CUDA IPC, forms one distributed invocation across AtnAgents,
coordinates execution across FfnAgents, and returns the contribution expected
by SGLang.

The current E2E validation targets are `deepseek-ai/DeepSeek-V2-Lite-Chat` and
`Qwen/Qwen3-14B`. Their IDs are evidence identities, not production adapter
allowlists. SGLang is the sole serving engine.

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

## Implementation Map

- `src/xpool/service/daemon/` and `src/xpool/runtime/` own Python control-plane
  authority and participant lifecycle.
- `src/xpool/fabric.py`, `src/xpool/transport.py`, and
  `src/xpool/service/wire.py` own Python domain and wire values.
- `src/xpool/integrations/sglang/` owns serving-engine integration and model
  adapters; `src/xpool/ops.py` owns the sole Tensor dispatcher facade.
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
- `tests/harness/` owns reusable collection, scheduling, process, GPU, native,
  and SGLang infrastructure and does not import collected test modules.

`tests/README.md` is the user-facing testing guide: it describes suite layers,
placement rules, requirements, artifacts, and canonical commands without
enumerating individual cases. `tests/harness/README.md` is the harness
architecture reference. It documents the collection-to-execution pipeline,
module boundaries, process tree and typed supervision protocol, GPU/MPS lease
ownership, endpoint-family reservations, result/artifact flow, and extension
rules. Superseded research notes are not retained as a second source of truth.

The suite collects concrete pytest items into a typed plan, runs CTest before
Python stages, runs Unit before Integration, and admits E2E only after
Integration succeeds. GPU tasks are sorted by required GPU count and estimated
duration, then backfilled over the idle pool. The complete MPS pool-usability
probe, the complete CTest stage, and every pytest GPU task each run in a
`SupervisedTaskScope`; CTest retains its internal resource-spec scheduling
inside that one stage scope. Whole-run GPU locks are released only after every
scope created by any stage reaches `CLOSED`. An unproven descendant domain
retains its locks and fails closed even when `SuiteRunner` was never created.

The runner-to-supervisor control pipe is also the parent-liveness boundary. If
the runner exits abruptly, EOF makes the dedicated supervisor drain its task
root and all adopted descendants before the runner-level subreaper reaps the
supervisor. This preserves the same empty-domain invariant for graceful
cancellation, timeout, and parent death.

`TaskCompletion` also represents a Supervisor-local infrastructure failure when
the Supervisor has nevertheless drained the task domain and proved it empty;
that task contributes suite exit code 2 without cancelling unrelated scopes.
`TaskStartFailure` represents startup rollback that proved the attempted domain
empty. `TaskSupervisionFailure` represents loss of the normal Supervisor
protocol and triggers runner-wide fallback; it is not yet a final emptiness
result. `TaskScopeFailure` is reserved for fallback that still cannot prove the
descendant domain empty or close its Supervisor, and only that terminal failure
quarantines the affected GPU lease and retains whole-run locks.

Successful runner fallback advances failed scopes to `DRAINED` and returns
normally without manufacturing task completions. Concurrent `SuiteRunner`
closes those scopes, releases their leases, stops scheduling, and exits with
infrastructure code 2. `SupervisedTaskScope.run()` owns the complete serial
start, wait, fallback, and close lifecycle for MPS pool usability and CTest. It
returns `INFRASTRUCTURE_FAILED` after resource-safe startup rollback or a
recovered supervision failure, and otherwise returns only after the scope is
`CLOSED`.

Concurrent task launch performs MPS checks, directory creation, and command
construction before acquiring a task-local GPU lease. Lease acquisition and
`SupervisedTaskScope.start()` are the sole launch transaction:
`TaskStartFailure` returns the lease, while `TaskScopeFailure` quarantines it.
The final resource-release proof also requires `GpuPool.active_leases` to be
empty so runner bookkeeping cannot release whole-run locks while a lease is
still outstanding.
Supervisors are isolated from terminal signals and normal cancellation uses the
typed control pipe. A signal sent directly to a Supervisor is owner loss, not a
local cleanup request; the runner subreaper performs emergency cleanup and stops
the suite.

The GPU pool is derived only from startup `CUDA_VISIBLE_DEVICES`, normalized to
physical UUIDs, and locked for the complete run. Every selected GPU must pass
MPS preflight. `CUDA_MPS_PIPE_DIRECTORY` is mandatory and identifies the
externally managed controller; xpool does not fall back to NVIDIA's shared
`/tmp/nvidia-mps` endpoint. Queries to the same explicit controller pipe are
serialized across processes because its control client does not support
concurrent commands. Each SGLang process receives an owned endpoint family,
including HTTP, NCCL, gRPC, and DP-derived ports when needed.

`tests/harness/sglang/manifest.toml` is the sole source for E2E model IDs,
topologies, graph modes, FfnAgent/Executor counts, observer capacities,
estimates, timeouts, and test-only KV limits. Model paths remain external and
come only from `XPOOL_CONFIG`. `model_serving_cases` exercise the complete
serving contract through FfnAgent loopback until Phase 8; separate
`loopback_serving_cases` sweep all explicit debug sites. Cross-mode token
parity is evaluated only for complete declared groups. Every successful HTTP
attempt retains a versionless `*.inference.json` record containing the exact
model ID, URL, request body, response status/content type, and decoded response;
the record is written before status and token-shape validation so failed
inference remains inspectable.

E2E startup treats daemon health and agent registration as distinct phases.
Daemon HTTP readiness has a 60-second deadline and records its measured startup
duration in the case artifact; agent registration retains a separate 30-second
deadline so a slow daemon cannot consume the participant-registration budget.

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

## Current Verification

The 2026-07-27 loopback milestone passed the canonical build, complete
`python -m tests`, and the non-duplicated formatting, lint, type, Doxygen,
native formatting, and pre-commit non-test quality gates. Exact test counts are
runtime collection facts and are intentionally not copied into this
architecture document.

Implemented status must be re-evaluated when a protocol, ownership, support
boundary, or validation gate changes.

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
6. DeepSeek/Qwen numerical oracles across supported TP/DP and graph modes,
   including concurrent two-model execution; and
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

`tests/harness/supervisor.py` adds two parent-side exception types while keeping
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

`tests/harness/runner.py::SuiteRunner.start_task()` creates directories and the
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
false on `TaskScopeFailure`. Whole-run GPU locks close only when that fact and
the runner/pool lease proofs all hold.

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
with `SupervisedTaskScope.run(...)`. `tests/harness/ctest.py::CtestSuite.run()`
retains its signature and result type but launches its complete CTest command
through one serial scope; CTest's resource specification and internal parallel
scheduler remain unchanged. An ordinary nonzero CTest exit with JUnit is code
1; timeout, leak, Supervisor infrastructure completion, missing JUnit, or the
existing infrastructure sentinel is code 2. A terminal `TaskScopeFailure`
escapes to the composition root and retains whole-run locks. CTest unit tests
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
