# FFN Execution

FFN execution owns model materialization, routed and dense operators,
GraphTemplates, Executor Lanes, and delivery kernels. See [Fabric](fabric.md)
for admission and distributed invocation lifecycle.

## Model materialization

FfnAgent adapters compile integration-provided architecture evidence into
CrossPool-owned model specs. Adapter discovery is automatic. Each supported model
family lives in its own module and directly selects its activation, Router, and
checkpoint mapping functions.

Weights are read from safetensors through explicit checkpoint keys. Loading
verifies tensor presence, logical shape, dtype, and TP divisibility. Each
FfnAgent retains only its planned shard; no complete weight replica is retained
to simplify TP execution. The checkpoint Tensor dtype is source evidence,
whereas the model profile's payload dtype is the effective BF16 or FP16
execution dtype. Conversion occurs only for the selected local shard.

Router weight precision is an adapter-owned policy, independent of payload
and Expert precision. GLM-4 MoE Lite retains FP32 Router weights; weight
loading, Control Probe resources, and memory estimation preserve that dtype.

Dense layers retain gate, up, and down shards. MoE layers additionally retain
Router weights, expert tensors, top-k and group-routing parameters, and any
model-defined correction bias. Activation is parameterized. The admitted
execution model implements gated formulas.

## Operators, routing, and delivery

The FFN path uses the pinned low-level Expert implementation from the supported
SGLang environment for admitted BF16 and FP16 devices. Only
`xpool.runtime.ffnagent.operators` may import reusable low-level Expert and
TopK operations, including the fused-MoE Triton configuration and Kernel
implementation. Model adapters retain model-specific geometry, validation, and
routing semantics. Startup selects immutable launch configuration and Graph
Capture executes the Expert kernels.

Any temporary adaptation of SGLang global arguments is scoped and
unconditionally restored. No SGLang runtime object, callable, or state is
retained by an Execution Registry, GraphTemplate, Plan, native Projection, or
CrossPool global. Expanding this dependency seam requires an accepted design change;
the current implementation has no provider hierarchy, backend registry,
fallback, copied upstream kernel, or request-time selection.

Model adapters select Router functions directly. Qwen3-MoE,
DeepSeek-V2-Lite, and GLM preserve their upstream routing mathematics.
CrossPool-owned Triton routing is used only where upstream exposes no reusable
standalone operation with the required semantics.

GLM owns a local biased-sigmoid TopK kernel with configuration-driven routed K
and Expert count. Padded Experts are excluded from selection; equal biased
scores prefer higher Expert IDs. This is an explicit local tie policy, not a
guarantee of Torch TopK ordering. Numerical acceptance against the independent
reference, rather than kernel identity with the serving engine, qualifies it.
The kernel reuses the FP32 logits workspace to preserve stored-score rounding
and normalizes selected weights without dividing by zero. The preceding gate
GEMM regenerates logits on every execution. Shared-Expert finalization remains
separate, and no capture-owned temporary allocation is introduced.

GLM converts Router inputs into caller-owned FP32 workspace before its FP32
matrix multiplication. Only this conversion uses local `torch.compile` during
pre-capture warmup, producing a kernel compatible with declared-address Graph
parameterization without maintaining a handwritten conversion kernel. This
does not enable serving-engine whole-model compilation or relax native
parameter checks. Input conversion and logits share the accounted Router
workspace; payload and Expert arithmetic retain their admitted precision.

The Router Owner finalizes Routing Metadata in Lane storage. A successor Kernel
publishes the fixed-capacity payload and `RoutingMetadataReady` identity to the
selected FFN TP ranks. Consumers validate the Invocation Key and Executor Lease
before using their copies. Routing publication is required protocol behavior;
the FFN Routing Observer records optional evidence from the same immutable data
and never provides readiness or rewires the Lane Graph.

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
Signature contains compute-function identity, tensor geometry, Capacity, TP
realization, and delivery behavior. It omits model IDs and layer ordinals when
multiple layers can safely reuse the same structure.

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
dependency splicing belong to `xpool::utils::graph`; FFN resource matching,
Routing Publication placement, and Lane topology remain in
`xpool::ffnagent`.

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
remain referenced after installation. Installation either publishes the whole
runtime or rolls back construction resources; activation never performs lazy
materialization.

`FfnAgentControl` owns the process-local Device allocation shared by Lane
Graph activation edges and, on the Coordinator PE, the Fabric Coordinator
Scheduler. The Fabric Runtime creates it, passes the activation counter to the
Execution Runtime, waits for all resident owners to start, and destroys it
during Fabric finalization.
