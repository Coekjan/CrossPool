# CrossPool

CrossPool separates attention-side serving from FFN weight residency and
computation while retaining one generation-scoped data plane.

## Language

**CrossPool**:
The resource-disaggregated multi-model serving system described by this glossary.

Avoidance notes distinguish domain concepts in discussion and documentation.
They do not prohibit citing existing implementation symbols by their actual
names or require those symbols to be renamed.

Use `KV Cache` or `KV` in documentation, comments, and docstrings. Keep the
phrase as two title-cased words; never hyphenate it or lowercase its second
word. Executable identifiers may use `kv_cache` or `kv`.

**Device**:
A physical AI accelerator used for CrossPool computation, distinct from the
process-local index used to address it. Platform-specific APIs and mechanisms
retain their precise technical names.
_Avoid_: GPU or CUDA device when referring to the platform-neutral domain concept

**Deployment Device View**:
The ordered physical Devices selected for one deployment. Configured Device
indices identify positions in this view, distinct from host inventory indices
and logical role ranks.
_Avoid_: Host device numbering, Instance Rank order

**Instance**:
A configured model-serving deployment whose workers share the deployed model's
Model ID and one Fabric Instance Plan.
_Avoid_: InstanceRankRuntime, model process

**Model ID**:
The `namespace/name` identity of one model, preserving its spelling independently
of its local checkpoint path or particular serving deployment.
_Avoid_: Checkpoint path, model directory

**Benchmark Target**:
One model-serving endpoint measured in a benchmark case, identified by its Model
ID and corresponding to a configured Instance or an externally supplied endpoint.
Each Model ID has one Target within a case; aggregate measurements are derived
across Targets rather than another Target.
_Avoid_: Benchmark Instance, aggregate target

**Serving Benchmark Case**:
A concrete serving-performance experiment fixing its declared models, deployment
conditions, prompt inputs or generation parameters, and arrival process. It is
distinct from its executions, observed environments and retained measurements.
_Avoid_: Workload template, benchmark run

**Isolated Benchmark Deployment**:
A single-model reference for a mixed-model benchmark, retaining the comparison's
resource geometry and remaining target's workload while removing the peer model.
_Avoid_: Mixed deployment with zero peer arrival rate

**Catalogue Case ID**:
The stable identity of one authored test or benchmark scenario, distinct from
its Model IDs, concrete test rows and executions.
_Avoid_: Run ID, pytest node ID, Model ID

**Case Family**:
The scenario category shared by catalogue cases, such as serving or topology,
distinct from a serving case's owned or client execution mode.
_Avoid_: Execution mode, test suite

**Benchmark Measurement**:
Client-observed timing, token progress and request outcomes from one benchmark
workload execution, independent of their presentation.
_Avoid_: Benchmark Report, plotted results

**Benchmark Repetition**:
One requested complete measurement of a Serving Benchmark Case using its
prepared workload, distinct from its identity, execution attempts and requests.
_Avoid_: Case ID, Benchmark Attempt

**Benchmark Attempt**:
One actual execution of a Benchmark Repetition with its own measurements and
outcome; another attempt does not create another logical repetition.
_Avoid_: Additional repetition, request retry

**Benchmark Report**:
An offline presentation derived from retained Benchmark Measurements and their
deployment conditions, without another serving execution.
_Avoid_: Benchmark execution, measurement run

**Instance Rank**:
One SGLang worker process within an Instance, owning rank-local attention
execution and one Transport attachment.
_Avoid_: InstanceRankRuntime rank, Instance process

**Attention World**:
The complete attention-side rank set backed by the shared AtnAgent Fleet. Each
Instance covers this World through attention DP groups of attention TP ranks.
_Avoid_: Attention TP width, per-model rank subset

**FfnAgent Fleet**:
The complete set of FfnAgents available for generation-scoped FFN placement,
distinct from the group selected to execute one layer.
_Avoid_: FFN TP group

**FFN Execution Group**:
The ordered FfnAgent participants assigned to one model layer's tensor-parallel
computation. Its width is that layer's FFN TP width, not the Fleet size.
_Avoid_: FfnAgent Fleet, model replica

**Prefill Context Parallelism**:
The SGLang execution mode that partitions Prefill token rows among attention
context-parallel ranks for attention computation. SGLang may preserve those
rank-local rows or gather them before an FFN boundary, depending on the model,
communicator, and expert-dispatch path.
_Avoid_: Attention TP, Decode Context Parallelism

**Decode Context Parallelism**:
The SGLang execution mode that partitions Decode KV positions within a
tensor-parallel group and combines partial attention results. It is distinct
from Prefill token-row partitioning and does not by itself define a new FFN
row partition.
_Avoid_: Prefill Context Parallelism, Decode TP

**Fabric Executable**:
A Fabric generation whose participants have activated their data-plane
resources, allowing Instance Ranks to attach Transport. It does not imply that
an Instance scheduler or public serving endpoint has finished starting.
_Avoid_: System ready, Serving healthy

**System Ready**:
The daemon verdict that the configured CrossPool processes and generation-scoped
data plane are ready for every Instance scheduler. It is distinct from public
serving-endpoint health.
_Avoid_: Fabric executable, HTTP ready

**Serving Listener**:
The host and port declared by one Instance for its public serving interface.
It is a bind target, not evidence that the socket is accepting requests.
_Avoid_: Instance Rank endpoint, healthy endpoint

**Serving Healthy**:
The one-time post-System-Ready observation that every configured Instance's
public HTTP health check has succeeded for the current listener snapshot. It
is not continuous availability monitoring or a Fabric lifecycle phase.
_Avoid_: System ready, Fabric executable, serving monitor

**Elastic Context State (ECS)**:
Request context retained for later inference whose physical-memory capacity
can grow and be reclaimed. It includes token histories, active recurrent state,
reusable checkpoints and auxiliary indices. Capacity sharing preserves each
Instance's state isolation and logical cache semantics.
_Avoid_: KV Cache when referring to non-KV retained state, model weights,
temporary execution workspace, Transport buffers

**Active Recurrent State**:
The mutable retained state advanced by one request's recurrent attention,
distinct from a reusable prefix snapshot.
_Avoid_: Token KV page, cached recurrent checkpoint

**Recurrent Checkpoint**:
A reusable snapshot of recurrent state at a processed prefix, distinct from
the active state advanced by an executing request.
_Avoid_: Active Recurrent State, token KV page

**Elastic Context State Memory Quota**:
The physical-memory allowance assigned to an Instance on one attention Device
for its managed Elastic Context State, distinct from the memory currently
backed and the logical capacities of its individual pools. Its pools share the allowance
without fixed live shares. Resident and retiring backing remain charged until
their physical release is confirmed.
_Avoid_: Pool bundle count, mapped memory, free device memory

**Elastic Context State Capacity Pool**:
The physical-memory capacity available for context-state backing across
Instance Ranks colocated on one attention Device.
_Avoid_: Shared state contents, token KV tensor pool

**Attention Runtime Headroom**:
One Instance Rank's immutable byte budget for service-time attention memory
kept outside its Device's Elastic Context State Capacity Pool. The daemon sums
these declarations per device. It is distinct from the configured utilization margin
and the one-time device-memory observation.
_Avoid_: KV margin, live memory usage, free-memory report

**Elastic Context State Control Channel**:
The daemon-owned, Fabric-Generation-scoped host-local control surface that
connects per-device capacity pools, logical capacity groups, and their
rank-local partitions. It carries capacity demand, adjustment commands, group
coordination, and physical completion observations.
_Avoid_: AtnAgent channel, TP-group channel, state-data channel

**Elastic Context State Capacity Group**:
The Instance ranks for one Model's attention data-parallel replica that jointly
adopt physical-byte quotas across their tensor-parallel shards.
_Avoid_: TP pool, KV allocation group

**Elastic Context State Quota Target**:
The absolute physical-byte allowance for each rank in a capacity group,
distinct from retained backing and individual pools' logical capacities.
_Avoid_: Bundle target, mapped capacity, token limit

**Elastic Context State Active Quota**:
The physical-byte quotas currently adopted by a capacity group. Its ranks
may retain different sparse backing inventories within their allowances.
_Avoid_: Active bundle prefix, mapped capacity, active requests

**Elastic Context State Quota Command**:
A distinct group capacity adjustment with a fixed target. The daemon authorizes
the target; the group coordinates safe application and reports its outcome.

**Elastic Context State Quota Demand**:
The Capacity Group leader's persistent report of the evaluated grant and
absolute per-rank bytes required by an unresolved admission-failure witness,
paired with its scheduler-local SLO deadline.
_Avoid_: Device utilization, capacity request event, one-bundle request

**Elastic Context State Quota Completion**:
One rank's terminal report that it adopted a Quota Command and its actual
retained physical bytes fit the new allowance.

**KV Page Bundle**:
One rank-local compound VMM mapping unit covering the same row interval across
one KV pool's layers and buffers. Its physical cost follows that pool's
immutable geometry.
_Avoid_: CUDA page, KV block

**KV VMM Backing**:
The stable virtual storage, Tensor views and sparse physical residency for one
rank-local KV pool, distinct from shared byte permission and logical allocation.
_Avoid_: VMM manager, elastic KV pool

**Elastic Context State Capacity Reconciliation**:
The adoption of a Quota Target and safe adjustment of rank-local backing
while preserving logical allocation and in-flight device access.
_Avoid_: KV resize, capacity application, VMM policy

**FFN Execution Installation**:
The generation-scoped assembly of FFN computation and the resources needed to
run it.
_Avoid_: Graph install, backend install

**FFN Semantics**:
The closed vocabulary shared across CrossPool for describing an FFN invocation,
its layout, its required output, and its outcome.
_Avoid_: ABI values, Transport FFN enums, Fabric FFN types

**FFN Model Specification**:
The generation-independent description of a model's FFN geometry, mathematical
behavior and checkpoint identity, distinct from the source model architecture
used to interpret its configuration.
_Avoid_: Source architecture, Execution Signature, Fabric Plan

**Execution Activation**:
The startup transition that makes an installed FFN execution available to the
generation.
_Avoid_: Execution installation, execution selection

**Compute Branch**:
One selectable FFN computation body within a Lane Graph, separate from the
Lane's waiting, delivery, and lifecycle structure.
_Avoid_: Compute Graph, Lane Graph

**Primary Graph**:
A construction-only FFN graph captured from one representative layer and used
to materialize compatible Compute Branches.
_Avoid_: Primary Layer, production GraphExec

**Control Graph**:
A construction-only companion to the Primary Graph used to discover which
captured parameters vary between compatible layers.
_Avoid_: Control Layer, fallback graph

**Control Capture Probe**:
The temporary shape-equivalent resources that distinguish variable parameters
while comparing a Primary Graph with its Control Graph.
_Avoid_: Control weights, Control Layer resources

**Execution Capacity**:
The padded row capacity of one Compute Branch. It is distinct from live rows,
Executor Lane count, and a compute-load budget.
_Avoid_: Batch size, Lane capacity

**Routing Metadata**:
The final per-row Expert identifiers and weights shared by the FFN participants
for one MoE layer invocation.
_Avoid_: Router output, TopK result

**Router Algorithm**:
The scoring rule used to select routed Experts and derive their weights,
distinct from its numerical precision policy.

**Router Precision**:
The element-type policy for Router projection inputs, retained weights and
logits, distinct from the Router Algorithm and payload element type.

**Router Correction Bias**:
A per-Expert adjustment applied to activated scores only for Expert selection.
Selected Expert weights derive from the uncorrected scores under the model's
normalization and scaling rules.
_Avoid_: Router projection bias, logits bias

**Hook Point**:
A neutral extension seam named after the production event it exposes rather
than the observer that consumes it.
_Avoid_: Callback site, plugin branch

**Fabric Coordinator**:
The generation-scoped progress owner that forms Invocations, leases Executor
Lanes, commits outputs, and releases acknowledged work.
_Avoid_: FfnAgent Coordinator, Scheduler kernel

**Executor Lane**:
A generation-scoped distributed slot that can own one FFN Invocation at a time.
Its count bounds simultaneous Invocations, not row capacity.
_Avoid_: FFN concurrency, Executor

**Submission Rendezvous**:
The Fabric Coordinator's agreement check over the AtnAgent Submissions that
form the next Invocation for one Instance.
_Avoid_: Invocation handoff, Submission merge

**Canonical Generation Failure**:
The first failure accepted for a Fabric generation and subsequently observed by
every participant.
_Avoid_: Local result, latest failure

**Payload Element Type**:
The scalar type of hidden-state payload elements, distinct from wider types used
for algorithm state or accumulation.
_Avoid_: PayloadDType, dtype code

**Native ABI Version**:
The stopped-world compatibility identity shared by one native extension and its
Python caller.
_Avoid_: ABI probe, Python ABI source

**Exact Resource Ledger**:
The stage-live sum of CrossPool-owned logical Tensor requests and native device
allocations whose sizes follow stable formulas.
_Avoid_: Exact allocation demand, safe estimate

**Analytic Allocator Allowance**:
A history-independent upper bound for allocator overhead beyond the Exact
Resource Ledger.
_Avoid_: Allocation error tolerance, allocator simulator

**Calibration Sample**:
A synthetic FFN structure with fixed geometry and mathematical behavior, chosen
to exercise resource construction independently of a real checkpoint.
_Avoid_: Source model family

**Calibration Experiment**:
A controlled startup measurement setup combining one or more Calibration Samples
with their placement and execution configuration.
_Avoid_: Model family, FFN structure kind

**Calibrated Overhead Envelope**:
An environment-qualified empirical bound for device-memory overhead not covered
by the Exact Resource Ledger and Analytic Allocator Allowance.
_Avoid_: Universal overhead, default reserve

**Device Observation Quantum**:
The smallest positive unit represented by one run's device-memory observations.
_Avoid_: Pressure quantum, memory margin

**Operator Device-Memory Margin**:
An explicit deployment reserve added after the predicted startup peak.
_Avoid_: Estimator correction, calibration allowance

**Resource Qualification**:
Real-model evidence that the memory estimate admits startup without hiding
systematic underprediction.
_Avoid_: Memory profiling, calibration fitting
