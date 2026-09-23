# CrossPool Roadmap

This document maps long-term candidate workstreams and their relationships.
Current architecture lives under `docs/designs/`; accepted implementation
plans live under `docs/plans/<task>/README.md`.

Accepted changes that need a decision-complete target design receive a separate
`docs/plans/<task>/README.md`. Owners, status, priority, schedules, and progress
belong in the team's external tracker rather than this repository.

The [design map](../designs/README.md) describes the implemented and accepted
system. The root [domain glossary](../../CONTEXT.md) owns CrossPool terminology.

## Relationships

The Roadmap uses only two relationships:

- **Requires** identifies evidence or a decision without which the workstream
  cannot proceed.
- **Benefits from** identifies another capability that improves implementation,
  diagnosis, or evaluation without blocking the workstream.

Model Coverage, Serving-engine Coverage, and Accelerator Portability are
orthogonal dimensions. Their presence on this Roadmap does not promise the full
Cartesian product of models, engines, and accelerators. Each supported
combination requires its own accepted plan and qualification evidence.

## Workstream map

| Workstream | Class |
| --- | --- |
| Unified Timeline Observability | Platform Capability |
| Live KV Cache Observability | Platform Capability |
| Model Coverage | Product Capability |
| Context Parallel Serving | Product Capability |
| Serving-engine Coverage | Product Capability |
| Cross-host Fabric | Product Capability |
| Accelerator Portability | Product Capability |
| Code Quality and Taste | Cross-cutting Practice |
| Benchmark Workflows | Cross-cutting Practice |
| Evaluation and Baselines | Cross-cutting Practice |

## Unified Timeline Observability

- **Class:** Platform Capability.
- **Outcome:** One queryable and visualizable causal timeline spanning the
  daemon, serving Instances, AtnAgents, FfnAgents, CPUs, accelerators, and future
  hosts.
- **Current seam:** Typed Transport, Fabric, Graph, and Routing Observer evidence
  plus the identities already propagated across CrossPool protocols.
- **Requires:** An accepted event and clock-domain model, interface research,
  and evidence that Host, accelerator, process, and cross-host clocks can be
  aligned with adequate fidelity.
- **Benefits from:** Existing typed Observer snapshots and Hook Points.

Research must compare suitable collection and export interfaces rather than
preselecting one. Candidates include Perfetto, Chrome Trace, OpenTelemetry,
NVTX/CUPTI, and combinations of complementary interfaces. The result must also
describe collection overhead, capacity, causality, queryability, and portability
to non-CUDA accelerators.

Candidate deliverables are a timeline requirements report, a minimal prototype
over existing Observer evidence, clock-alignment measurements, and an accepted
active plan for Unified Timeline Observability. Continuous streaming, alerting,
and a production monitoring service are outside this workstream.

## Live KV Cache Observability

- **Class:** Platform Capability.
- **Outcome:** Provide `xpool top` to show live KV Cache occupancy by GPU and
  model, with pool totals and each model's share clearly distinguished.
- **Current seam:** Generation-scoped KV capacity accounting and the KV Control
  Channel already track physical pool budgets and partition capacity.
- **Requires:** A read-only live snapshot contract, a clear distinction between
  physically backed bytes and logical token usage, and evidence that collection
  does not disturb serving or capacity coordination.
- **Benefits from:** Unified Timeline Observability, without depending on it.

Research must identify which occupancy values are authoritative and which need
new instrumentation. Candidate deliverables are a telemetry-source audit, a
minimal terminal view, and an accepted active plan for `xpool top`. Operational
logs are not a source for the live view.

## Model Coverage

- **Class:** Product Capability.
- **Outcome:** Support additional model families within the existing FFN
  semantics or through an explicitly accepted extension of those semantics.
- **Current seam:** The engine-neutral FFN Model Adapter, automatic discovery,
  and serving-integration-specific model adapters.
- **Requires:** Per-family ownership of FFN formula, activation, Dense or MoE
  geometry, routing, checkpoint mapping, dtype and operator requirements, and
  reference qualification.
- **Benefits from:** Unified Timeline Observability and the existing numerical
  qualification harness.

A model compatible with the current gated Dense/MoE boundary can proceed as a
focused implementation task. A model requiring Unary FFN, expert parallelism,
quantization, a new Router, or a new operator first needs research for that
missing capability. Candidate deliverables are one compatibility review,
implementation, and qualification package per model family.

## Context Parallel Serving

- **Class:** Product Capability.
- **Outcome:** Support SGLang Prefill and Decode Context Parallelism with a
  context-parallel degree greater than one while preserving correct FFN row
  ownership and Elastic KV Cache capacity semantics.
- **Current seam:** SGLang integration rejects context-parallel execution; the
  attention topology and FFN handoff currently use TP/DP rank geometry.
- **Requires:** Current SGLang model and topology support evidence, a runnable
  prototype, an accepted FFN row and KV Capacity Group contract, and numerical,
  graph, and serving qualification.
- **Benefits from:** Existing model-qualification and installed-serving harnesses.

Research must establish rank-local FFN row shapes, KV Capacity Group membership,
and graph behavior for a runnable Prefill and Decode Context Parallelism setup
before changing CrossPool interfaces. Candidate deliverables are a SGLang
support audit, a minimal installed-serving prototype, and a decision-complete
active plan with per-model qualification.

## Serving-engine Coverage

- **Class:** Product Capability.
- **Outcome:** Support serving engines in addition to SGLang without leaking
  engine-owned types or state into core Plans, Registries, Projections, or the
  native ABI.
- **Current seam:** `xpool.ops.ffn_shim`, engine-neutral core values, and the
  isolation boundary demonstrated by `xpool.integrations.sglang`.
- **Requires:** A compatibility spike covering module replacement, weight
  filtering, TP/DP behavior, eager and graph execution, output ownership,
  process lifecycle, and failure propagation.
- **Benefits from:** A representative qualified model and Unified Timeline
  Observability.

vLLM is the first candidate. Research must determine whether its public
extension surfaces are sufficient, which existing SGLang integration behavior
is truly engine-neutral, and whether the FfnAgent's bounded use of SGLang
low-level Expert kernels remains acceptable. Candidate deliverables are a vLLM
compatibility spike, an integration-boundary report, a representative end-to-end
prototype, and an accepted active plan for vLLM integration.

No multi-engine provider abstraction should be introduced before a second
integration demonstrates a shared seam.

## Cross-host Fabric

- **Class:** Product Capability.
- **Outcome:** Extend AtnAgent-to-FfnAgent and FfnAgent-to-FfnAgent Fabric traffic
  across hosts while retaining the Instance-to-AtnAgent Transport as a
  host-local, rank-local path.
- **Current seam:** The symmetric NVSHMEM Fabric, fixed PE world, Device-side
  publication protocol, and canonical generation failure.
- **Requires:** A real two-host NVSHMEM/IBGDA prototype plus accepted host
  identity, NIC/GPU topology, bootstrap, placement, lifecycle, failure, and
  shutdown contracts.
- **Benefits from:** Unified Timeline Observability and the existing Fabric
  protocol.

Research must verify publication ordering and Graph replay over the remote
transport, compare GPU-initiated and CPU-proxy behavior, measure NIC/QP and
completion costs, and determine whether generation-wide fail-stop remains the
correct failure model. Candidate deliverables are environment qualification,
raw two-host prototype evidence, a host-aware control-plane design, and an
accepted active plan for Cross-host Fabric.

The workstream does not justify a speculative transport-backend registry.

## Accelerator Portability

- **Class:** Product Capability.
- **Outcome:** Port selected CrossPool control-plane, data-plane, and execution
  capabilities to a non-CUDA accelerator platform, with Ascend as the first
  candidate.
- **Current seam:** Engine-neutral core values, model semantics, control-plane
  concepts, and the serving-integration boundary.
- **Requires:** Feasibility evidence for equivalents to CUDA Graph, CUDA IPC,
  NVSHMEM, CCCL synchronization and cooperative primitives, MPS, Device memory
  observation, FFN operators, and timeline data sources.
- **Benefits from:** SGLang Ascend, `sgl-kernel-npu`, Model Coverage, and Unified
  Timeline Observability.

Research should reuse the existing SGLang NPU stack where it is suitable and
concentrate CrossPool work on its own control and data planes. Candidate deliverables
are an SGLang NPU reuse audit, a CrossPool platform-gap report, a minimal end-to-end
Ascend prototype, and an accepted active plan for the Ascend port.

The CUDA implementation must not acquire empty backend abstractions solely for
this candidate workstream.

## Code Quality and Taste

- **Class:** Cross-cutting Practice.
- **Outcome:** Continuously remove unnecessary abstraction, duplicate wrapping,
  excessive checks, and inconsistent ownership without creating a permanent
  broad refactoring project.
- **Requires:** A bounded review scope and concrete evidence for every finding.
- **Benefits from:** Applying the practice to every implementation task rather
  than scheduling it as a separate development phase.

Candidate deliverables are bounded audit reports and small root-cause changes.
Only findings that alter architecture or public contracts receive an active
plan. This workstream does not add a second linter, include sorter, comment
density rule, or source-text quality gate.

## Benchmark Workflows

- **Class:** Cross-cutting Practice.
- **Outcome:** Provide `xbench` as the canonical benchmark runner alongside
  `xtest`, with live progress and metrics and durable results in a consistent,
  machine-readable format.
- **Current seam:** `xtest` already manages test resources and artifacts;
  existing serving and report-only performance runs provide benchmark inputs.
- **Requires:** An accepted benchmark case and result contract covering workload,
  environment, warmup, measurements, failures, and artifact ownership.
- **Benefits from:** Reusing applicable `xtest` resource and process-lifecycle
  mechanisms, and Unified Timeline Observability when available.

Research should identify what can be shared with `xtest` without making
benchmarks depend on test verdicts. Candidate deliverables are a minimal
`xbench` prototype, a stable live-and-stored result shape, and an accepted
active plan for benchmark execution and result retention.

## Evaluation and Baselines

- **Class:** Cross-cutting Practice.
- **Outcome:** Evaluate CrossPool functionality, resource efficiency, throughput,
  latency, and tail latency with fair and reproducible methods.
- **Current seam:** Numerical, topology, serving, and report-only performance
  evidence already owned by the qualification system.
- **Requires:** Explicit hardware, models, workloads, arrival process, memory
  budget, warmup, concurrency, SLOs, metrics, raw artifacts, and treatment of
  failed or incomplete runs.
- **Benefits from:** Benchmark Workflows, Unified Timeline Observability, and
  the Product Capability being evaluated.

Baseline classes include native SGLang and potentially vLLM, multi-model sharing
systems such as MuxServe, KV-elastic systems such as kvcached, and CrossPool
ablations. Candidate deliverables are an accepted benchmark methodology,
workload and baseline compatibility reports, reproducible runs, raw results, and
analysis reports.

The Roadmap defines no performance bound while the system remains incomplete.
An unavailable baseline does not justify widening a production interface.

## Starting work

To start work from this Roadmap:

1. Select one concrete candidate deliverable.
2. When material uncertainty exists, complete source or API research, the
   smallest throwaway prototype that resolves it, and a recommendation backed by
   raw evidence.
3. Create `docs/plans/<task>/README.md` before implementing an accepted change to
   architecture, interfaces, data structures, ownership, lifecycle, or
   validation contracts.
4. After implementation and acceptance, use the repository's `write-design`
   workflow to update the current design and obtain confirmation for plan cleanup.
