# FFN Execution

FFN execution owns model materialization, routed and dense operators,
GraphTemplates, Executor Lanes, and delivery kernels. See [Fabric](fabric.md)
for admission and distributed invocation lifecycle.

## Model materialization

FfnAgent adapters compile integration-provided architecture evidence into
CrossPool-owned model specs. Adapter discovery is automatic. Each supported model
family lives in its own module and compiles activation, Router Algorithm,
Router Precision, geometry, and checkpoint identities. Source architecture
lookup ends at this boundary; execution consumes the compiled semantics.
Synthetic calibration declares those same semantics directly rather than
selecting a model Adapter.

Shared family-profile validation admits BF16 and FP16 checkpoint declarations
alongside the expected model type and SiLU activation. The Llama adapter reuses
this validation rather than restricting the family to one checkpoint's dtype.
Family admission is distinct from
[qualification of a concrete checkpoint](qualification.md#numerical-and-graph-evidence).

Weights are read from safetensors through explicit checkpoint keys. Loading
verifies tensor presence, logical shape, dtype, and TP divisibility. Each
FfnAgent retains only its planned shard; no complete weight replica is retained
to simplify TP execution. The checkpoint Tensor dtype is source evidence,
whereas the model profile's payload dtype is the effective BF16 or FP16
execution dtype. Conversion occurs only for the selected local shard.

Router Precision determines projection input, retained weight and logits dtypes,
independently of the Router Algorithm. `PAYLOAD` uses the effective payload dtype
for all three. `FP32` uses FP32 for all three, converting projection inputs into
caller-owned storage. `FP32_LOGITS` retains payload-precision inputs and weights
and produces FP32 logits. These policies preserve backend arithmetic settings,
including TF32. Loading, Control Probe construction and memory estimation use
the same declared precision.

Dense layers retain packed gate/up and down shards in two owned storages.
MoE layers additionally retain Router weights, expert tensors, top-k routing
parameters, and any model-defined correction bias. Activation is parameterized.
The admitted execution model implements gated formulas.

Checkpoint loading and synthetic calibration share canonical empty-weight
allocation factories, preserving shape, dtype, allocation order, and ownership.
The loader fills those tensors from selected checkpoint shards; calibration
initializes them without checkpoint I/O.
Checkpoint-key containers own their ordered key projections. Weight owners
construct zero-valued shape-equivalent Control Capture Probes and expose
non-owning native address projections while retaining Tensor ownership.

## Operators, routing, and delivery

The FFN path uses the pinned low-level Expert implementation from the supported
SGLang environment for admitted BF16 and FP16 devices. Only
`xpool.runtime.ffnagent.operators` may import reusable low-level Dense and Expert
operations, including the fused-MoE Triton configuration and Kernel
implementation. Model adapters compile model-specific geometry, validation, and
routing semantics. Startup selects immutable launch configuration and Graph
Capture executes the Expert kernels.

Any temporary adaptation of SGLang global arguments is scoped and
unconditionally restored. Retained execution state is CrossPool-owned. The
current seam calls the selected SGLang operator and configuration modules
directly, without a provider hierarchy or fallback dispatch. Expanding this
dependency seam requires an accepted design change.

The shared [execution geometry](../../src/xpool/runtime/ffnagent/execution.py)
owns Signatures, workspace shapes, alignment, offsets and Graph-storage byte
formulas. Allocation, operator checks, memory estimation and placement consume
those same formulas. Geometry calculation is below routing and operators and
performs no Tensor allocation or kernel launch. MoE capture allocates one
contiguous workspace covering every admitted block size; typed views follow
the selected kernel configuration and preserve shared-region overlap.

The [Router implementation](../../src/xpool/runtime/ffnagent/routing.py) owns
projection, TopK and final Routing Metadata construction. Source compilers select
the admitted Algorithm/Precision pairs: Qwen3-MoE uses Softmax with `PAYLOAD`,
DeepSeek-V2 uses Softmax with `FP32_LOGITS`, and GLM uses Sigmoid with `FP32`.
The Spec validates these combinations and normalization; execution consumes the
admitted semantics directly.

Both algorithms use one local Triton TopK kernel. The Host entry validates
resources, constructs projection logits and launches that kernel directly.
The kernel computes FP32 activated scores and selects Experts using those scores
plus optional per-Expert FP32 correction bias. Selected weights come from the
original activated scores, followed by model-declared normalization and scaling.
The admitted Sigmoid path requires normalization. Bias presence is independent
of the algorithm.

Softmax uses stable max/exp/sum scoring. Equal selection scores prefer lower
Expert IDs for Softmax and higher IDs for Sigmoid. These are explicit local tie
policies rather than guarantees of Torch TopK ordering. Padding is excluded from
selection, and normalization preserves zero weights when their sum is zero.
Sigmoid reuses its FP32 logits workspace for a stored-selection-score rounding
boundary while keeping original scores separate for weight calculation. The
preceding projection regenerates logits on every execution. Independent
numerical and Graph qualification establishes the admitted implementation.

The `FP32` policy converts Router inputs into caller-owned workspace before its
matrix multiplication. Only this conversion uses local `torch.compile`
during pre-capture warmup, producing a kernel compatible with declared-address
Graph parameterization. Serving-engine whole-model compilation remains outside this
seam, and native parameter checks remain in force. Input conversion and logits
share the accounted Router workspace; payload and Expert arithmetic retain
their admitted precision.

The Router Owner finalizes Routing Metadata in Lane storage, appending Shared
Expert routes and invalidating Capacity tails. A successor Kernel
publishes the fixed-capacity payload and `RoutingMetadataReady` identity to the
selected FFN TP ranks. Consumers validate the Invocation Key and Executor Lease
before using their copies. Routing publication is required protocol behavior;
the FFN Routing Observer records optional evidence from the same immutable data.
Readiness and Lane Graph topology remain production responsibilities.

Dense, MoE, Router workspace, captured input and output, and resident weights
use the effective BF16 or FP16 dtype except where model mathematics requires
FP32 Router or accumulation state. TP1 delivers its complete local result
directly. Wider TP groups combine each output element by a fixed PE-order FP32
fold, cast once to payload dtype, synchronize the CTA, and then publish the
result. Completion publication owns the remote-completion ordering for all
nonblocking delivery writes. Fast math requires separate numerical and
performance evidence.

## GraphTemplate and Executor Lanes

A GraphTemplate is one reusable captured compute structure. Its Execution
Signature contains execution semantics, tensor geometry, Capacity, TP
realization, and delivery behavior. It omits model IDs and layer ordinals when
multiple layers can safely reuse the same structure.
Router Algorithm, Precision, normalization and bias presence specialize capture.
Replay consumes current Tensor contents and binds layer addresses within that
fixed resource schema; optional ownership remains constant within a Signature.
Registry installation, memory estimation, and placement use one shared required
Signature inventory rather than deriving their own capture domains.

Each request selects the smallest installed Capacity that contains its live row
count. Capacity tails are storage only and carry no semantic payload. Capacity
derivation and validation belong to FfnAgent installation; the Fabric
Coordinator publishes live rows and the selected Layer, not a Graph Capacity.

Each Signature Capture produces topology-identical Primary and Control Graphs.
Primary Capture uses execution resources and is embedded into each Lane Graph.
Control Capture uses probe resources only to discover and validate the concrete
Kernel argument schema. Both source Graphs and temporary capture resources are
construction resources and are released after synchronous installation.

Graph parameterization discovers supported resource-address sites and converts
the embedded Primary Kernel Nodes into Device-updatable nodes. Before each
execution, the Lane Graph binds the selected Layer's resident addresses without
mutating Graph topology. Generic Host-side CUDA Graph construction,
conditional-node creation, topology comparison, Kernel parameter storage, and
dependency splicing belong to `xpool::utils::graph`. Kernel parameter storage
normalizes CUDA pointer-array and packed-buffer transports into the same owned
logical arguments. FFN resource matching, Routing Publication placement, and
Lane topology remain in `xpool::ffnagent`.

Different Executor Lanes own different GraphExec instances even when they share
one GraphTemplate. A GraphExec is never launched concurrently with itself.
Every Lane Graph contains the resident wait, request binding, selected embedded
compute branch, routing publication where required, delivery branch,
completion publication, and lease re-arm needed for Device-side progress.

`ExecutionProjection` owns intrinsic Signature, Capacity, Layer, resource
schema, topology, ordering, and cross-Capacity weight-geometry validation.
Runtime installation separately validates live CUDA allocations, current-device
accessibility, Graph handles, and external runtime results. An established
Projection invariant is not checked again in private materialization helpers.

The native Execution Runtime owns installed GraphExecs, Lane state, discovered
schemas, shared immutable lookup tables, workspaces, streams, and activation.
The Python `FfnExecutionRegistry` owns the tensors and modules whose addresses
remain referenced after installation. It validates each local weight owner
against its co-indexed Spec and Plan before per-Capacity capture. Dense/MoE
dispatch preserves that admitted relationship through the Signature and Control
Capture Probe cache; private capture consumers rely on that correspondence.
Installation either publishes the whole runtime or rolls back construction
resources; activation never performs lazy materialization.

`FfnAgentControl` owns the process-local Device allocation shared by Lane
Graph activation edges and, on the Coordinator PE, the Fabric Coordinator
Scheduler. The Fabric Runtime creates it, passes the activation counter to the
Execution Runtime, waits for all resident owners to start, and destroys it
during Fabric finalization.
