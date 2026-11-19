# FFN Execution

FFN execution owns model materialization, routed and dense operators,
GraphTemplates, Executor Lanes, and delivery kernels. See [Fabric](fabric.md)
for admission and distributed invocation lifecycle.

## Model materialization

FfnAgent adapters compile integration-provided architecture evidence into
CrossPool-owned model specs. Adapter discovery is automatic. Each supported model
family lives in its own module and compiles activation, Router Algorithm,
Router Projection Precision, geometry, and checkpoint identities. Source
architecture lookup ends at this boundary; execution consumes the compiled
semantics.
Synthetic calibration declares those same semantics directly rather than
selecting a model Adapter.

The immutable FFN Spec declares each projection's encoding, values-source layout
and physical keys, with separate per-Expert or stacked routed sources and
explicit Expert cardinality. A fused or stacked source key is counted once. Activation parameters,
additive projection bias and selection-only Router correction bias have separate
declarations. A representable declaration does not establish operator support
or qualify a concrete model.

Family adapters check the model type and declared activation. Where the adapter
names a source dtype field, the checkpoint must declare BF16 or FP16. Original
GPT-OSS configs omit that field. `InstanceFfnProfile` independently requires
BF16 or FP16 payloads. See [checkpoint qualification](qualification.md#numerical-and-graph-evidence).

Compilation validates exact FFN key coverage for each Spec-declared layer
within family-owned checkpoint prefixes. Before materializing weight tensors,
the loader checks Model/Instance identity
and ordered layer geometry. Weights are read from explicit safetensors keys.
The bounded reader verifies each original source shape, forms its declared
canonical view, then applies the selected TP slice. Matrix and scale-blocked
views are declared per projection, allowing gate/up and down to use different
source-axis layouts.
Floating sources may convert to another floating dtype; encoded values and
scales require matching dtypes and are
copied unchanged. Layer loading also checks key coverage and TP geometry. Each
FfnAgent retains only its planned shard; no complete weight replica is retained
to simplify TP execution. The checkpoint Tensor dtype is source evidence,
whereas the model profile's payload dtype is the effective BF16 or FP16
execution dtype. Conversion occurs only for the selected local shard.

Additive projection biases use FP32 storage. During Host staging, Dense and
`PAYLOAD` Router bias values first round to the payload dtype and then widen to
FP32. Expert and other Router biases use FP32 source conversion. Placement and
Control Probes account for the same physical layout as the loaded weights.

Checkpoint Gate/Up order describes source rows; the selected operation declares
its required execution order. The loader normalizes either concatenated or
interleaved sources into that order with matched row selections for values,
scales and bias. TP slicing preserves encoding-block constraints, and target
padding retains its neutral values. Row order remains independent of activation
and weight encoding. Separate Gate/Up and single Up sources share projection
descriptor construction; stacked Expert sources follow their declared layout.

Bounded pinned slots stage those selections on the Host. Contiguous destinations
use the ordinary blocking copy; strided rows use direct two-dimensional CUDA
copies on the destination's current stream. Each span is synchronized before
advancing or returning the slot, preserving initialization and slot lifetime
without a temporary device Tensor. Splitting Expert spans may increase startup
copy-call count; the transfer protocol and device-error ownership are unchanged.

Router Projection Precision determines input, retained weight and logits dtypes,
independently of the Router Algorithm. `PAYLOAD` uses the effective payload dtype
for all three. `FP32` uses FP32 for all three, converting projection inputs into
caller-owned storage. `FP32_LOGITS` retains payload-precision inputs and weights
and produces FP32 logits. Ordinary FP32 matrix multiplication uses IEEE FP32.
Loading, Control Probe construction and memory estimation use
the same declared precision.

TopK Weight Rounding is independent of projection precision. `NONE` retains
FP32 selected weights. `LOGITS_DTYPE` rounds them through the logits dtype after
optional normalization, then publishes FP32 Routing Metadata. Shared-Expert
assembly and model-declared scaling follow that boundary.

Dense and MoE execution resources name their first and second weight projections
`i_proj` and `o_proj` across prepared layouts, weight owners, execution signatures
and native bindings. A gated input projection contains packed Gate/Up rows;
an ungated Dense input projection contains one projection. Checkpoint sources
retain their mathematical Gate/Up/Down roles. Dense input bias follows the TP
slice; output bias is loaded on rank zero and remains zero on other ranks, so
TP reduction adds it once.
MoE layers additionally retain Router weights, expert tensors, top-k routing
parameters, and any model-defined correction bias. The model declares SiLU
(with slope `alpha`), ReLU or GELU independently of a layer's gated structure. A model
has a `GateTransform` for its gated layers, with an optional positive clipping
`limit` and `up_offset`; its default performs ordinary gating. The projection
sources determine whether a layer is gated. An ungated execution uses
`Activation.gate=None`. Projection bias is applied before clipping.
A limit bounds Gate above and Up on both sides; the result is
`activation(g) * (u + up_offset)`.
Ordinary gated SiLU uses slope 1, no clipping and zero offset. GPT-OSS resolves
its slope and clipping from the source configuration, using 1.702 as the slope
fallback, and declares an Up offset of 1. The qualified checkpoints use a
clipping limit of 7. Up offset is independent of projection bias.

Checkpoint loading and synthetic calibration share canonical empty-weight
allocation factories, preserving shape, dtype, allocation order, and ownership.
The loader fills those tensors from selected checkpoint shards; calibration
initializes them without checkpoint I/O.
Layer and source declarations own their ordered physical-key enumeration.
Weight owners construct zero-valued shape-equivalent Control Capture Probes and
expose non-owning native binding views while retaining Tensor ownership.

## Operators, routing, and delivery

The [operator package](../../src/xpool/runtime/ffnagent/operators/) owns the
pinned matrix, activation and routed Expert dependency seam. Dense, Router and
Experts compose the same `GemmPlan` and `Gemm` in `linear.py`. A plan retains
operand layouts, weight encoding and Gate/Up order, output precision, optional
activation fusion and local scratch. Quantization owns encoded storage and TP
slices; matrix execution owns representation compatibility, kernel selection and
scratch. Model adapters declare source geometry and mathematical semantics.

Small ordinary unquantized projections use a local row-reduction kernel.
Larger ordinary matrices with actual FP32 inputs and weights use local IEEE FP32
SIMT kernels and an explicit Split-K reduction. Remaining ordinary payload and
grouped floating/MXFP4 matrices use the bundled Triton-Kernels primitive.
FP32 logits from payload inputs retain payload multiplication. Prepared encoding
determines values and optional scales; adding an encoding requires its
representation and arithmetic implementation.

The [activation module](../../src/xpool/runtime/ffnagent/operators/activation.py)
writes SiLU, ReLU or GELU, with or without gating, into caller-owned outputs.
Ungated operations may reuse their input; gated operations use separate output
storage. The standalone combinations share a local Triton kernel and their
scalar activation expressions. Ungated ReLU preserves
Torch's FP16/BF16 bit patterns. Floating matrices store their payload-precision
projection before standalone activation, preserving that rounding boundary.

Activation also supplies matrix epilogues. Gated fusion requires interleaved
Gate/Up rows; other physical layouts retain a materialized projection and
standalone activation. Floating fusion rounds the projection to its declared
dtype before activation. Encoded execution retains accumulator precision;
interleaved SiLU with unit Up offset uses upstream `swiglu_fn`. The other fused
combinations use the same scalar expressions as standalone activation. Matrix
plans account for materialized intermediate storage whenever fusion is absent.

`DenseExecutionPlan`, `RouterExecutionPlan` and `ExpertExecutionPlan` compose
matrix plans and stored workspace views. `MoeExecutionPlan` places Expert and
optional Router backing in one root. Preparation retains concrete compute plans
by Signature index for budgeting, allocation and capture. `WorkspaceLayout`
stores aligned offsets and creates views without recomputing byte formulas.
Each root receives its own allocator-reservation estimate, multiplied by the
Lane count. Mutable backing remains independent per Lane.

The common Expert pipeline constructs route histograms, gather/scatter indices
and the pinned upstream ragged metadata, executes both matrices and reduces the
weighted routes. Projection temporaries and route outputs reuse storage after
their last consumer. Metadata covers the pinned matrix library's encoded block
sizes, including its schedule-index limit. Matrix warmup selects public launch
flags from actual geometry and device properties, with measured tile overrides
for selected ordinary BF16 shapes on Hopper. Bundled matrices use single-split,
non-persistent launches. Floating and MXFP4 ragged matrices use at least four
warps. The root plan accounts for local FP32 Split-K partials. The final
reduction combines those partials before bias and activation. Temporary library
configuration is restored on success and failure. Runtime
objects borrow current layer tensors and own no checkpoint weights.

[Execution geometry](../../src/xpool/runtime/ffnagent/execution.py) defines
Signatures and capture placeholder layouts. [Preparation](../../src/xpool/runtime/ffnagent/preparation.py)
derives each TP shard's physical requests and Capacity-specific plans without
allocating tensors or querying a device. Loading, capture and placement retain
the same representation metadata. The Registry validates layer owners against
those layouts before capture, then native replay supplies current resources
without Python operation selection per invocation.

The [Router implementation](../../src/xpool/runtime/ffnagent/operators/routing.py) owns
projection, TopK and final Routing Metadata construction. Source compilers select
the admitted Algorithm/Precision pairs: Qwen3-MoE uses Softmax with `PAYLOAD`,
DeepSeek-V2 uses Softmax with `FP32_LOGITS`, and GLM uses Sigmoid with `FP32`.
GPT-OSS uses Softmax with `PAYLOAD`, additive projection bias and
`LOGITS_DTYPE` weight rounding; the existing families use `NONE`. The Spec
validates these combinations and normalization; execution consumes the admitted
semantics directly.

Dense, Router and Expert projections share the [linear entry](../../src/xpool/runtime/ffnagent/operators/linear.py)
and write caller-owned outputs. Pure preparation fixes local geometry and
scratch before allocation; warmup fixes bundled launch flags without changing
that scratch. Primary and Control use the same selection, and replay performs
no Python operation selection. Each invocation supplies current weight, scale
and FP32 bias addresses directly. Runtime module loading and compilation
finish before capture. Pure preparation runs without a device context.

Both algorithms use one local Triton TopK kernel. The Router uses its prepared workspace to construct projection logits and
launches that kernel directly.
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
matrix multiplication. Input conversion and logits share the accounted Router
workspace. Payload and Expert arithmetic retain their admitted precision.

The Router Owner finalizes Routing Metadata in Lane storage, appending Shared
Expert routes and invalidating Capacity tails. A successor Kernel
publishes the fixed-capacity payload and `RoutingMetadataReady` identity to the
selected FFN TP ranks. Consumers validate the Invocation Key and Executor Lease
before using their copies. Routing publication is required protocol behavior;
the FFN Routing Observer records optional evidence from the same immutable data.
Readiness and Lane Graph topology remain production responsibilities.

Captured payload input and output use effective BF16 or FP16. Prepared layouts
separately determine floating or encoded weight storage, biases and all
Router/Expert workspace dtypes. TP1 delivers its complete local result
directly. Wider TP groups combine each output element by a fixed PE-order FP32
fold, cast once to payload dtype, synchronize the CTA, and then publish the
result. Completion publication owns the remote-completion ordering for all
nonblocking delivery writes. Fast math requires separate numerical and
performance evidence.

## GraphTemplate and Executor Lanes

A GraphTemplate describes compute shared by compatible layers. Each Lane
captures its own instance for each Execution Signature. The Signature fixes
geometry, Capacity, TP and delivery, operator semantics and physical layouts.
Layers can share a Signature when only their weight addresses differ. Capture,
placement and memory estimation use the same Signature inventory.

Each request selects the smallest installed Capacity that contains its live row
count. Capacity tails are storage only and carry no semantic payload. Capacity
derivation and validation belong to FfnAgent installation; the Fabric
Coordinator publishes live rows and the selected Layer, not a Graph Capacity.

For each required Signature the Lane captures two graphs with the same
topology. Primary supplies the compute nodes embedded in the Lane Graph.
Control uses probe weights so installation can identify which kernel arguments
must change when the Lane selects another layer. Compatible captures may reuse
read-only probes. Replay uses the current layer's resident weights.

The FfnAgent measures its clean free-memory baseline before loading weights.
`FfnExecutionRegistry` retains weights and owns each Lane's capture backing
through confirmed native retirement. Concrete Signature plans determine that
backing before capture.

Each active capture Lane owns an exclusive stream and a private Torch memory
pool with `use_on_oom=False`. Compute workspace stays at
their Primary addresses. IO, routing and live-row placeholders use the
default allocator. The Signature's placeholder layout supplies the allocation
shapes and request sizes; budgeting separately counts temporary initialization
and Torch's per-graph RNG state. The Registry retains initialized tensors
while construction can still reuse their subblocks.

The Lane owner captures Primary and Control against one root workspace.
A recursive scalar Graph query checks whether each graph references that range
and validates typed access bounds throughout the graph. Both captures must
agree on usage. A used range must have complete retained pool coverage; address
observation alone cannot establish ownership or discover opaque nested pointers.
Native installation checks the optional workspace view against IO, both weight
inventories and other Lanes, including different Signatures. Each Lane keeps
its mutable scratch private. Predicted versus actual pool reservation is reported
separately from ownership failures.

The Registry checks real Tensor layouts against the Layer Plan. Native
installation checks all Lanes against the same Layer mapping, verifies graph
topology and device accessibility, and binds Fabric IO, routing and live rows.
Weight arguments match exact projection bases, including optional scales and
bias; compute ranges allow verified interior offsets. Supported memory nodes
include bounded pitched Memset and contiguous one-dimensional device copies
with both endpoints checked. Unsupported nodes or inconsistent access fail
installation.

The native runtime owns each Lane's GraphExec, execution stream and binding
tables. It borrows compute storage. Fixed-table sizing counts native requests;
driver graph storage, variable binding arrays and allocator reservation can
add physical cost. Each Lane runs its own wait, binding, compute, delivery
and lease re-arm sequence. A GraphExec never runs concurrently with itself.
Lane terminal branches follow the block-wide decision contract in
[Fabric](fabric.md#scheduler-and-failures).

Installation is synchronous and consumes one attempt. On success the Registry
synchronizes capture streams, resets the source graphs and releases
construction tensors and probes. It then trims
releasable default cache. Each private pool and its stream stay sealed against
new allocations until retirement; replay can still write scratch there.
[Memory admission](control-plane.md#memory-admission) uses this point for the
retained budget.

On partial failure the native runtime drains streams before destroying graphs
or their storage. If cleanup also fails, it preserves the remaining handles.
The Registry holds weights and Lane backing until native
retirement is confirmed. Activation never retries installation.

`FfnAgentControl` owns the process-local Device allocation shared by Lane
Graph activation edges and, on the Coordinator PE, the Fabric Coordinator
Scheduler. The Fabric Runtime creates it, passes the activation counter to the
Execution Runtime, waits for all resident owners to start, and destroys it
during Fabric finalization.
