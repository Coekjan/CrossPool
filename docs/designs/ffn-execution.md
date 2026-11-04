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

Family adapters check the model type, SiLU activation and BF16 or FP16 source
dtype. `InstanceFfnProfile` independently requires BF16 or FP16 payloads. See [checkpoint qualification](qualification.md#numerical-and-graph-evidence).

Before checkpoint I/O the loader checks Model/Instance identity and ordered
layer geometry. Weights are read from explicit safetensors keys. The bounded
reader verifies each original source shape, forms its declared canonical view,
then applies the selected TP slice. Matrix and scale-blocked views are declared
per projection, allowing gate/up and down to use different source-axis layouts.
Floating sources may convert to another floating dtype; encoded values and
scales require matching dtypes and are
copied unchanged. Layer loading also checks key coverage and TP geometry. Each
FfnAgent retains only its planned shard; no complete weight replica is retained
to simplify TP execution. The checkpoint Tensor dtype is source evidence,
whereas the model profile's payload dtype is the effective BF16 or FP16
execution dtype. Conversion occurs only for the selected local shard.

Checkpoint Gate/Up order describes source rows; the selected operation declares
its required execution order. The loader normalizes either concatenated or
interleaved sources into that order with matched row selections for values,
scales and bias. TP slicing preserves encoding-block constraints, and target
padding retains its neutral values. Row order remains independent of activation
and weight encoding.

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
and produces FP32 logits. These policies preserve backend arithmetic settings,
including TF32. Loading, Control Probe construction and memory estimation use
the same declared precision.

TopK Weight Rounding is independent of projection precision. `NONE` retains
FP32 selected weights. `LOGITS_DTYPE` rounds them through the logits dtype after
optional normalization, then publishes FP32 Routing Metadata. Shared-Expert
assembly and model-declared scaling follow that boundary.

Dense layers retain packed gate/up and down shards in two owned storages.
MoE layers additionally retain Router weights, expert tensors, top-k routing
parameters, and any model-defined correction bias. `SiluActivation` declares
the positive slope `alpha`, optional positive clipping `limit`, and `up_offset`.
Projection bias is applied first. Clipping bounds Gate above and Up on both
sides before computing `g * sigmoid(alpha * g) * (u + up_offset)`. Ordinary
gated SiLU uses `(1.0, None, 0.0)`. Model adapters resolve source parameters; execution supplies
no model-specific defaults. Up offset is part of activation, not projection bias.

Checkpoint loading and synthetic calibration share canonical empty-weight
allocation factories, preserving shape, dtype, allocation order, and ownership.
The loader fills those tensors from selected checkpoint shards; calibration
initializes them without checkpoint I/O.
Layer and source declarations own their ordered physical-key enumeration.
Weight owners construct zero-valued shape-equivalent Control Capture Probes and
expose non-owning native binding views while retaining Tensor ownership.

## Operators, routing, and delivery

The [operator package](../../src/xpool/runtime/ffnagent/operators/) owns the
pinned low-level projection, activation and Expert dependency seam. Ordinary
BF16/FP16 Experts use SGLang's Triton configuration and launchers through
`SglangTritonExperts`; packed MXFP4 Experts with BF16 payloads use the upstream
Triton-Kernels operations through `TritonKernelsExperts`. Model adapters declare
source geometry and mathematical semantics, not concrete operator classes.
Pure preparation selects one Expert operation and checks its representation
before weight reads or device work. The same closed operation mapping serves
preparation, workspace sizing and capture.

The shared [activation module](../../src/xpool/runtime/ffnagent/operators/activation.py)
supplies `apply_into` for caller-owned Dense and floating-Expert outputs and
`fused` descriptors for packed Expert GEMMs. Both consume the same activation
declaration while preserving each operation's qualified fusion and
payload-rounding boundaries. Selection follows parameters and row layout,
not model identity.
The unit Up-offset path retains upstream `swiglu_fn`; other offsets use the
local `fused_gated_silu`. Fused descriptors preserve each implementation's
function identity so specialization caches distinguish them.

Any temporary adaptation of SGLang global arguments is scoped and
unconditionally restored. Retained execution state is CrossPool-owned. The
concrete operations call their pinned libraries and configuration modules
directly, without a provider hierarchy or service-time fallback. Expanding this
dependency seam requires an accepted design change.

[Execution geometry](../../src/xpool/runtime/ffnagent/execution.py) defines
Signatures and capture placeholder layouts. [Preparation](../../src/xpool/runtime/ffnagent/preparation.py)
derives each TP shard's weight requests and Capacity-specific Signatures
without allocating tensors. Loading, capture and placement use the same layout.
An MoE workspace has one root large enough for every admitted block size.
The Expert implementation lays out its slice; Router storage has independent
offsets. Estimation uses the same byte extents as view construction.

The [Router implementation](../../src/xpool/runtime/ffnagent/operators/routing.py) owns
projection, TopK and final Routing Metadata construction. Source compilers select
the admitted Algorithm/Precision pairs: Qwen3-MoE uses Softmax with `PAYLOAD`,
DeepSeek-V2 uses Softmax with `FP32_LOGITS`, and GLM uses Sigmoid with `FP32`.
The Spec
validates these combinations and normalization; execution consumes the admitted
semantics directly.

Dense and Router projections share the [linear entry](../../src/xpool/runtime/ffnagent/operators/linear.py)
and write caller-owned outputs. Torch `mm`/`addmm` preserves the existing
floating paths. Startup selects FlashInfer TinyGemm for biased BF16 `PAYLOAD`
Router projections with Capacity at most 128, aligned geometry and a device
meeting its SM90+ TMA requirement; other projections keep Torch. This choice
is fixed for the capture environment, and replay performs no operation
selection. TinyGemm uses `use_pdl=False` and the same Primary/Control resource
discovery and Device updates as other captured operations. Runtime module
loading and compilation finish before capture; pure preparation does not
initialize a device context.

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

The FfnAgent prepares source declarations before the free-memory baseline.
It keeps any external owners locally and transfers them with the weights to
`FfnExecutionRegistry` before materialization. A Signature borrows only its
declared shared ranges. Production currently declares stream-local BLAS
scratch; it supplies no baseline external Tensor. BLAS settings must stay
fixed from preparation through capture.

Each active capture Lane owns an exclusive stream and a private Torch memory
pool with `use_on_oom=False`. Compute workspace and library scratch stay at
their Primary addresses. IO, routing and live-row placeholders use the
default allocator. The Signature's placeholder layout supplies the allocation
shapes and request sizes; budgeting separately counts temporary initialization
and Torch's per-graph RNG state. The Registry retains initialized tensors
while construction can still reuse their subblocks.

A source declares the identity and logical extent of each range. Pool coverage
proves the backing is retained; comparing graph addresses alone cannot prove
ownership or discover opaque nested pointers. Installation checks Primary and
Control offsets and access bounds. Writable storage is disjoint across Lanes;
explicit read-only ranges may be shared. A difference between predicted and
actual pool reservation is reported separately from an ownership failure.

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

Installation is synchronous and consumes one attempt. On success the Registry
synchronizes capture streams, resets the source graphs, clears stream-local
library caches and releases construction tensors and probes. It then trims
releasable default cache. Each private pool and its stream stay sealed against
new allocations until retirement; replay can still write scratch there.
[Memory admission](control-plane.md#memory-admission) uses this point for the
retained budget.

On partial failure the native runtime drains streams before destroying graphs
or their storage. If cleanup also fails, it preserves the remaining handles.
The Registry holds weights, external owners and Lane backing until native
retirement is confirmed. Activation never retries installation.

`FfnAgentControl` owns the process-local Device allocation shared by Lane
Graph activation edges and, on the Coordinator PE, the Fabric Coordinator
Scheduler. The Fabric Runtime creates it, passes the activation counter to the
Execution Runtime, waits for all resident owners to start, and destroys it
during Fabric finalization.
