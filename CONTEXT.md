# xpool

xpool separates attention-side serving from FFN weight residency and
computation while retaining one generation-scoped data plane.

## Language

Avoidance notes distinguish domain concepts in discussion and documentation.
They do not prohibit citing existing implementation symbols by their actual
names or require those symbols to be renamed.

**Instance**:
A configured model-serving deployment whose SGLang workers share one identity
and one Fabric Instance Plan.
_Avoid_: InstanceRankRuntime, model process

**Instance Rank**:
One SGLang worker process within an Instance, owning rank-local attention
execution and one Transport attachment.
_Avoid_: InstanceRankRuntime rank, Instance process

**Fabric Executable**:
A Fabric generation whose participants have activated their data-plane
resources, allowing Instance Ranks to attach Transport. It does not imply that
an Instance scheduler or public serving endpoint has finished starting.
_Avoid_: System ready, Serving healthy

**System Ready**:
The daemon verdict that the configured xpool processes and generation-scoped
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

**FFN Execution Installation**:
The generation-scoped assembly of FFN computation and the resources needed to
run it.
_Avoid_: Graph install, backend install

**FFN Semantics**:
The closed vocabulary shared across xpool for describing an FFN invocation,
its layout, its required output, and its outcome.
_Avoid_: ABI values, Transport FFN enums, Fabric FFN types

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
The stage-live sum of xpool-owned logical Tensor requests and native device
allocations whose sizes follow stable formulas.
_Avoid_: Exact allocation demand, safe estimate

**Analytic Allocator Allowance**:
A history-independent upper bound for allocator overhead beyond the Exact
Resource Ledger.
_Avoid_: Allocation error tolerance, allocator simulator

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
