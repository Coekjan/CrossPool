# GPT-OSS with resident MXFP4 Experts

Implementation and qualification are complete; the
[qualification report](qualification.md) retains measured evidence. Current
designs linked below own the implemented contracts. This completed target
document, including its original baseline and acceptance materials, is retained
for PR review at the user's request.

## Goal

Support both official `openai/gpt-oss-20b` and `openai/gpt-oss-120b` models on
six NVIDIA A100 40GB devices within CrossPool's SGLang/CUDA boundary.
FfnAgents retain their local Expert
weight shards in packed MXFP4 form in Device memory and execute the model's
quantization-aware FFN. Hidden-state transport remains BF16 initially.

The accepted implementation uses the A100 W4A16 path described below. Both
model sizes and the initial ordinary Elastic KV storage policy are required.
The operator experiment and qualification gates below define the accepted
validation scope.

The independently useful [SGLang research](research.md) records upstream source
evidence, numerical semantics, and backend differences. Current owners are
[FFN Execution](../../designs/ffn-execution.md),
[Control Plane](../../designs/control-plane.md),
[Elastic KV Cache Pooling](../../designs/elastic-kv-cache.md), and
[Qualification](../../designs/qualification.md).

## Baseline

The repository pins SGLang 0.5.20, sglang-kernel 0.4.7, Torch 2.13.0, and
Triton 3.7.1. Implementation targets the locked environment, not an unpinned
upstream branch.

- `src/xpool/ffn.py` represents ordinary SiLU-gated Dense/MoE layers. Expert
  count derives from a tuple of per-Expert projection keys. GPT-OSS instead
  stores all Experts in fused packed tensors and requires a different activation.
- `runtime/ffnagent/loader.py` accepts floating-point source tensors and converts
  local shards to payload dtype. `weights.py` requires BF16/FP16 Expert storage.
- `operators.py` captures unquantized low-level SGLang Expert kernels with
  caller-owned resources. Its current launch configuration disables TMA.
- `execution.py`, `registry.py`, and native FFN Projections describe two Expert
  weight pointers, Router weights, and an optional Router correction bias.
  They have no Expert scale or projection-bias resources.
- `device_memory.py` derives Expert bytes from payload dtype; placement uses
  this estimate before loading. A loader-only change would misprice memory.
- `integrations/sglang/server_args.py` rejects quantization globally. Family
  shims replace entire FFN modules and filter their checkpoint namespace.
- The installed Elastic KV lifecycle expects a single pool with one
  `KvVmmBacking`; GPT-OSS's upstream hybrid sliding-window pool requires an
  explicit compatibility choice before claiming installed serving support.

## Accepted Constraints

- Hardware qualification uses six A100 40GB devices (SM80), selected with
  `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`. Physical device 0 is excluded.
  Blackwell and Hopper paths are outside this task.
- Both 20b and 120b are mandatory for first acceptance. Implementing one family
  adapter or qualifying only 20b is insufficient.
- Expert matrices remain packed in Device memory; BF16 hidden states are the
  proposed initial arithmetic boundary.
- The first version uses the existing Elastic KV pool with upstream hybrid
  SWA memory optimization disabled, while preserving sliding attention semantics.

## Accepted Changes

### Storage and computation contract

MXFP4 means E2M1 values packed two per byte with one UE8M0 scale for each
32-value block. It is distinct from integer INT4 and NVIDIA NVFP4. Checkpoint
metadata, tensor keys, shapes, and dtypes jointly establish the format;
`torch_dtype=bfloat16` alone does not describe Expert storage.

The loader retains only the assigned TP shard of each assigned layer. It may
repack packed values and transform scales to a selected kernel layout during
startup. The final Expert matrix representation remains four bits per value;
scale dtype, padding, permutation, and auxiliary allocations are explicit parts
of its physical layout. A persistent BF16 expansion, including one hidden in
an upstream post-load hook, violates this task's residency contract.

Software execution decodes packed values and applies scales inside a bounded
kernel computation, using BF16 activations and higher-precision accumulation
where required. Native block-scaled execution uses a hardware-supported operand
combination; if it quantizes activations to MXFP8 or MXFP4, that is a separate
arithmetic contract requiring its own numerical evidence. Weight format does
not by itself authorize activation quantization.

The proposed implementation uses W4A16 with CrossPool-owned Triton kernels:
decode E2M1 values and UE8M0 scales per tile, convert that tile to BF16, and
use A100 BF16 Tensor Cores with FP32 accumulation. Only tile-local decoded
weights live in registers/shared memory. Activations and bounded intermediate
outputs may use BF16 Device storage.

Pinned SGLang's MXFP4 Marlin method explicitly rejects SM80. Its lower-level
kernel has SM80 code, but that is not evidence that the complete model path is
supported. A low-level Marlin experiment may inform a later optimization;
it is not a dependency or fallback for this first implementation. Kernel choice
is immutable for an installed generation. Unsupported combinations fail during
preparation; there is no service-time backend switch or silent BF16 expansion.

### GPT-OSS mathematics

For each row `x`, compute Router logits `l = x W_router^T + b_router`, select
the configured top-k Experts, and normalize the selected logits with softmax.
The official 20b and 120b configurations use top-4. The Router bias contributes
to both selection and probabilities; it is not the existing selection-only
Router correction bias.

The A100 Router uses cuBLASLt's biased BF16 GEMM with an explicit 1-MiB
workspace, matching the pinned original SGLang `Linear` path. Its selected
algorithm may round Split-K partials to BF16 before reduction; replacing that
operation with one FP32 dot product or an ordinary workspace-free GEMM can
change selected Experts. The same contract applies to 32 and 128 Experts and
to single-row and multi-row inputs. The surrounding FfnAgent retains its
zero-workspace policy for ordinary Torch cuBLAS calls.

The native `ffnagent.compute_bf16_router_logits` entry accepts BF16 input,
Router weight, bias and output tensors plus one caller-owned 1-MiB byte
workspace. It validates shapes, dtype, device, contiguity and workspace
alignment, selects the bounded cuBLASLt algorithm during preparation, and
enqueues work on the current stream. CUDA Graph capture freezes that choice;
replay allocates no storage and uses no host callback. Library failure is a
preparation error, with no backend switch. Host descriptors are scoped to the
preparation call; all device intermediates reside in the supplied workspace.

The native declaration exports the workspace byte count and 256-byte alignment.
`GptOssAdapter.router_workspace_bytes` accounts for that storage, alignment
padding and BF16 logits in the existing per-Lane Router workspace. Each Lane
has independent storage, and native graph parameterization relocates every
scratch pointer through the existing workspace range. The additional native
entry advances the ABI to 86 and invalidates existing calibration profiles.
Ordinary warmup retains the selected host algorithm value by device, matrix
geometry and alignment. Capture requires a warmed combination and reuses that
value; the cache owns no device allocation, layer pointer or per-Lane scratch.
Builds select the cuBLASLt shared library from the pinned `nvidia-cublas`
wheel used by Torch, while retaining the Toolkit headers paired with NVCC
and CCCL. The installed native extension searches
that sibling wheel directory so importing native code before Torch cannot
mix cuBLAS handles with another Toolkit cuBLASLt release.

For selected Expert `e`, using the decoded checkpoint weights:

```text
[g_raw, u_raw] = x W_gate_up[e]^T + b_gate_up[e]
g = min(g_raw, limit)
u = clamp(u_raw, -limit, limit)
h = g * sigmoid(alpha * g) * (u + 1)
y_e = h W_down[e]^T + b_down[e]
y = sum_e routing_weight[e] * y_e
```

The checkpoint gate/up rows are interleaved: `g0, u0, g1, u1, ...`.
The official activation uses `alpha=1.702` and `limit=7`; the gate has an upper
clamp only. Compilation validates the source activation parameters and
records them in the model digest. The adapter must not map this to ordinary
`SiLU(g) * u` or apply the routing weight before the nonlinear activation.

Both official sizes have hidden and Expert intermediate width 2880. The 20b
configuration has 24 layers and 32 Experts per layer; 120b has 36 layers and
128 Experts. These values are checkpoint validation evidence, not switches on
the model name. One family adapter compiles both geometries.

### TP and bias ownership

Continue tensor parallelism over the intermediate dimension, with every
participant retaining all Experts' local intermediate slices. Require exact
divisibility, 32-element block alignment, and the selected kernel's additional
tile alignment. Do not adopt upstream ceil-based TP padding implicitly.

- Slice gate/up rows in complete interleaved pairs together with their bias.
- Slice down-projection packed K blocks and their scales using the same local
  intermediate range. Hidden width remains full on each participant.
- Router weights and Router bias reside only on the existing Router Owner.
- Retain full `[E, H]` down bias only on FFN TP rank zero. That rank adds it
  to its per-Expert partial before routing-weight multiplication; other ranks
  contribute no down bias. The existing ordered FP32 TP fold then includes
  each selected bias exactly once. Backend-required zero placeholders on other
  ranks are allowed only as explicitly accounted resources.

The FFN payload, Routing Metadata, delivery mode, lane leasing, and native
Transport/Fabric wire formats keep their current meanings. Native execution
resource structures change, so the Native ABI Version and generated bindings
must follow the repository's normal compatibility rules.

## Interface Changes

All paths below are relative to `src/xpool/` unless stated otherwise.

| Owner | Target delta |
| --- | --- |
| `ffn.py` | Add `ActivationKind.SWIGLU_OAI`, finite positive `activation_alpha: float = 1.0` and optional positive `activation_limit: float \| None = None` to `FfnModelSpec`; ordinary SiLU requires the defaults, GPT-OSS requires its explicit values. Include all fields in the existing digest. |
| `ffn.py` | Add `Mxfp4MoeCheckpointKeys` and admit it alongside `MoeFfnCheckpointKeys` in `MoeFfnSpec.checkpoint`. Its discriminator is `format: Literal["mxfp4"]`; its `expert_count: int` and eight named keys represent Router weight/bias and gate/up plus down blocks/scales/bias. Derive `routed_expert_count` from the selected checkpoint variant. Existing per-Expert keys retain their current representation. |
| `runtime/ffnagent/models/gpt_oss.py` | Add `GptOssAdapter(MoeFfnModelAdapter)` for `GptOssForCausalLM`; compile config and exact namespace, select Router mathematics, validate BF16 payload and source quantization metadata. Reject unsupported checkpoints explicitly. |
| `runtime/ffnagent/weights.py` | Add separate Router `bias: Tensor \| None` with default `None`; retain `correction_bias` as a distinct semantic field. Add `Mxfp4MoeFfnWeights` to the closed `FfnLayerWeights` union. |
| `runtime/ffnagent/loader.py` | Keep `materialize_layer_weights` as the loading entry; dispatch on the typed checkpoint variant. Extend read descriptors to validate packed `uint8` sources separately from floating sources, preserving strict shape/dtype checks and bounded staging. |
| `runtime/ffnagent/execution.py` | Extend `MoeFfnExecutionSignature` with optional immutable `Mxfp4ExecutionConfig(alpha, limit, down_bias_present)` and Router additive-bias presence. The packed variant fixes canonical layout and SM80 W4A16 arithmetic; ordinary signatures retain `mxfp4=None`. |
| `runtime/ffnagent/operators.py` | Own fixed startup kernel configuration and a quantized Expert launch accepting retained weights, fixed Routing Metadata, caller-owned workspace, and output. Put the tiled MXFP4 kernels in a directly imported `mxfp4.py` sibling; reusable low-level SGLang routing/alignment imports stay in `operators.py`. |
| `runtime/ffnagent/registry.py` | Validate and retain quantized weight owners; create shape/layout-equivalent Primary/Control resources and project every scale/bias pointer for native binding. |
| Native `ffnagent/projection.hpp`, parameterization, runtime, and bindings | Extend the existing MoE projection with optional typed `Mxfp4ExpertBindingResourceProjection` and Router additive bias. The packed schema adds both scales and Expert biases to binding values; validate schema presence, geometry, TP bias ownership, address access, and cross-Capacity consistency. |
| Native `ffnagent/router.hpp`, implementation, and bindings | Add `compute_bf16_router_logits(hidden_states, weight, bias, output, workspace)` with explicit bounded cuBLASLt scratch and exported workspace geometry; retain the existing graph resource schema. |
| `integrations/sglang/models/gpt_oss.py` | Add a parameter-free `GptOssSparseMoeBlock` replacement, exact upstream constructor/forward compatibility, FFN namespace filtering, and full-boundary validation. |
| `integrations/sglang/server_args.py` | Replace blanket quantization rejection with the narrow admitted GPT-OSS/MXFP4 combination and verify effective model quantization again after config resolution. Other quantizers remain unsupported. |

The operator entry is `operators.compute_mxfp4_partial`. Like the ordinary
MoE entry, it receives explicit caller-owned alignment buffers, gate/up and
activation views, per-route outputs, final output, and immutable arithmetic
config. Its authoritative signature lives at the source declaration. It writes
the complete fixed-Capacity partial output on the current stream and keeps
routing state on the Device. Live-row tails have invalid Expert IDs and zero
route weights, matching the existing publication contract.
Do not add a general quantization provider API or reuse SGLang module objects
as core state.

## Data-Structure Changes

### Checkpoint representation

For `H=hidden_size`, `I=expert_intermediate_size`, and `E=expert_count`, each
layer consumes precisely these keys below `model.layers.<layer>.mlp`:

| Key suffix | Logical checkpoint shape | Meaning |
| --- | --- | --- |
| `router.weight` | `[E, H]` | Unquantized Router projection |
| `router.bias` | `[E]` | Additive Router bias |
| `experts.gate_up_proj_blocks` | `[E, 2I, H/32, 16]` | Packed interleaved gate/up E2M1 |
| `experts.gate_up_proj_scales` | `[E, 2I, H/32]` | UE8M0 scales |
| `experts.gate_up_proj_bias` | `[E, 2I]` | Interleaved projection bias |
| `experts.down_proj_blocks` | `[E, H, I/32, 16]` | Packed down E2M1 |
| `experts.down_proj_scales` | `[E, H, I/32]` | UE8M0 scales |
| `experts.down_proj_bias` | `[E, H]` | Down projection bias |

Validate this mapping against actual selected checkpoint headers before loading.
Only the verified official schema is admitted initially; alternate flattened
or third-party layouts require an explicit mapping, not a shape guess.
`checkpoint_keys_for_layer` enumerates each fused tensor once and existing
exact-namespace checks remain authoritative.

### Retained resources and execution identity

`Mxfp4MoeFfnWeights` owns `gate_up_packed`, `gate_up_scales`, `down_packed`,
`down_scales`, `gate_up_bias`, optional `down_bias`, and optional `router`.
The weight type defines one canonical row-major layout with interleaved
gate/up and uint8 E2M1/UE8M0 storage; a separate layout selector is unnecessary
for this single implementation. All tensor resources are CUDA-resident.
Byte counts use physical allocations, not inferred payload dtype.

The initial proposal preserves Router storage/rounding at the BF16 reference
boundary, with FP32 routing probabilities. Retain Expert biases in BF16, as
the pinned GPT-OSS MXFP4 loader does; add them to FP32 GEMM accumulators before
the output conversion. Use FP32 activation intermediates and BF16 activation
output. Down GEMM adds bias and multiplies the route probability in FP32 before
storing BF16 per-route output. The route fold accumulates these in FP32 and
casts the local partial once to BF16. Independent SGLang comparison retains
the existing numerical qualification policy.
A blanket dtype conversion must not silently alter Router selection.

The execution signature identifies software W4A16, its resource layout, and
bias roles. Native activation-quantizing execution remains outside scope. Compatible
layers share signatures; their IDs and weight contents do not enter the key.
Layer-specific pointers enter the closed native resource projection in stable
semantic order: packed gate/up, gate/up scales, packed down, down scales,
gate/up bias, optional down bias, then optional Router weight and additive bias.

Registry-owned immutable weight tensors live until execution retirement.
Workspaces belong to individual Executor Lanes. Primary/Control construction
buffers are released after installation. Probe scales use valid finite encoded
values; different addresses, not invalid scale data, distinguish the captures.

### A100 operator and workspace

Use canonical row-major `uint8` packed values and `uint8` UE8M0 scales with
the source interleaving retained. For local width `I_r`, flatten the trailing
block dimensions to `[E, 2I_r, H/2]` and `[E, H, I_r/2]`; scales have shapes
`[E, 2I_r, H/32]` and `[E, H, I_r/32]`. No permanent matrix padding or repacking
is needed for this proposed Triton representation. Kernel loads mask tile tails.

Use a fixed startup-selected tile configuration, initially `M=16, N=64, K=32`;
changing it requires regenerating the exact alignment/workspace geometry and
requalifying numerical/graph behavior. Avoid TMA and opaque tensor descriptors.
The execution pipeline is:

1. Align the published routing entries by Expert into caller-owned indices.
   The fixed launch grid covers the maximum padded route count; kernels read
   the actual padded extent from Device storage and mask inactive work.
2. Gate/up GEMM gathers each route's input row, decodes only the current packed
   weight tile and scale block, accumulates in FP32, adds bias, and stores the
   BF16 interleaved gate/up result in original route order.
3. A pointwise kernel applies the GPT-OSS activation and writes BF16 intermediate
   activations. Invalid routes produce zero output.
4. Down GEMM decodes local weight tiles, accumulates in FP32, and adds down bias
   on the designated rank. Apply routing weights at the reference-qualified
   rounding boundary, then reduce selected Expert outputs into the local partial.
5. Existing native delivery folds TP partials and casts to payload dtype.

For `C` rows and `R=C*topk`, use gate/up `[R, 2I_r]`, activated `[R, I_r]`,
and per-route down `[R, H]` BF16 views. Gate/up and down output share the
existing overlap region because down starts only after activation consumes
gate/up. Their combined allocation is `2R(I_r+max(2I_r,H))` bytes, in addition
to input/output and Router storage.
Account for route-alignment indices, block Expert IDs, padded-count scalar,
and any required reduction workspace from the exact selected alignment routine.
Subregions use the existing alignment convention. No replay-time allocations,
full-layer weight expansion, atomic accumulation across invocations, or host
readback of route counts are allowed.

### Memory admission and preparation

For an unpadded local shard with `N=3 E H I_local` Expert elements, checkpoint-
style packed values and scales use `N/2 + N/32 = 17N/32` bytes. This is the
baseline representation cost, not a complete Device-memory estimate. Any later
operator retaining wider scales or padded matrices needs a new layout ledger.

`device_memory.py` must account separately for final packed weights, final
scales, all bias/Router storage, layout padding, bounded repack overlap, temporary
Device staging, probe resources, per-Capacity capture resources, and per-Lane
operator scratch/locks. CPU pinned staging remains bounded by loader policy.
Do not keep both source and final Device layouts after preparation.

The sole selected representation is canonical MXFP4 with the fixed SM80 W4A16
operator, so all participants and the estimator share one physical-layout
contract before placement. Validate SM80 in FFN preparation before weight
allocation. No public backend selector or new automatic fallback is proposed.

The model-neutral memory corpus adds a packed MoE fit coordinate `C-Q4` and
includes it in the mixed held-out world. It covers scale/bias allocation count,
packed Control Probes, and quantized Graph Capture topology. Existing profiles
are invalidated by the Native ABI Version change.

### Six-device deployment and capacity

Start with two attention devices and four FFN devices. Both models use
attention TP2/DP1 and explicit FFN TP1. The existing placement solver assigns
whole layers to FFN devices; it may use a subset of the Fleet according to its
existing group-minimization and memory-admission objectives. Do not confuse
four FFN devices with FFN TP4. TP2 is a separate component/topology acceptance
case for bias/sharding correctness.

Ignoring biases and other resources, the official Expert matrices require:

| Model | Packed E2M1 + UE8M0 | BF16 expansion, for comparison |
| --- | --- | --- |
| 20b | 9.455 GiB | 35.596 GiB |
| 120b | 56.731 GiB | 213.574 GiB |
| Both | 66.186 GiB | 249.170 GiB |

These are derived from `3 * layers * Experts * H * I` elements at `17/32`
bytes per element for packed values/scales. One 120b layer occupies about
1.576 GiB before bias/Router resources. Thus the weight-only estimate fits
within the aggregate capacity of four 40GB FFN devices, including both models;
installed readiness still depends on the complete ledger and observed free memory.

Use a bounded initial serving context, proposed at 8192 tokens, and small
explicit concurrency for qualification. Exercise a prompt longer than the
configured sliding window; the retained workload uses 4097 input tokens.
Ordinary MHA storage allocates both K and V for every retained token/layer:
for 120b BF16 at attention TP2 the ideal KV storage
is `2 * 36 * (8/2) * 64 * 2 = 36,864` bytes per token per attention rank,
or 288 MiB for one 8192-token sequence, before allocator/page rounding.
Include separate per-model KV pools, unquantized model weights, graph storage,
and runtime headroom; six-device aggregate capacity is not an admission proof.

## Implementation

1. **Resolve the operator and serving seams.** On the chosen hardware, inspect
   real checkpoint headers and run a throwaway two-layer quantized MoE capture.
   Start with the raw-pointer Triton tiled decoder and BF16 GEMM. Establish packing, bias dtype,
   workspace geometry, supported TP alignments, and kernel argument ownership.
   Alternate two layers whose packed weights, scales, and biases all differ
   through one native GraphTemplate. Verify concurrent Lanes, capacity tails,
   routing publication placement, and no replay allocation or host callback.
   Freeze the selected API, registration needs, numerical policy, and resulting
   exact memory formulas in this plan before production changes.
2. **Implement semantics and materialization.** Add the typed model/checkpoint
   variant, GPT-OSS compiler, quantization validation, packed TP reads, final
   layout preparation, and resource ledger. Verify packed values/scales against
   an independent decoder and prove local-shard ownership. Preserve existing
   model behavior and reject malformed or unsupported input before execution.
3. **Install quantized execution.** Implement model-owned Router semantics,
   Expert activation/bias handling, allocation-free operator launches, native
   pointer binding, Control Probes, and GraphTemplate/Lane installation. Verify
   complete TP results against an unsharded reference, especially down bias.
4. **Integrate and qualify serving.** Add the SGLang shim, narrow quantization
   admission, independent reference binding, portable model cases, and actual
   installed serving. Resolve the hybrid KV compatibility described below.
   Qualify 20b first, then 120b; sharing an adapter does not substitute for each
   model's evidence. Update current design documents only after acceptance.

If a candidate kernel hides layer-varying addresses in opaque TMA descriptors
or cannot expose its scratch/output allocations, it fails the current execution
contract. Reconsider the candidate or obtain an explicit graph-design revision;
do not bypass parameter validation or capture a separate permanent graph per
layer as an undocumented workaround.

### Attention and Elastic KV compatibility

GPT-OSS retains SGLang's attention sinks and alternating full/sliding attention.
The accepted initial serving configuration uses upstream
`disable_hybrid_swa_memory=true` so all attention layers use the existing single
elastic MHA pool. This changes cache storage policy, not the sliding attention
mask. Use the pinned Triton attention backend on A100. Its actual pool type,
backend compatibility, memory demand, eviction,
growth, and graph replay must be verified on the pinned version.

The shared installed-serving launcher derives GPT-OSS architecture from the
checkpoint config and supplies Triton attention, ordinary SWA cache storage,
an 8192-token context, and request concurrency four. Manual launches supply
the same engine flags; the integration validates cache and backend compatibility
before loading. These launch bounds support qualification and do not claim
the model's full advertised context length.

If this setting cannot preserve the existing Elastic KV contract, installed
GPT-OSS serving remains blocked until a separately accepted composite-pool
design is available. A successful isolated FFN experiment must not be reported
as complete model support. Supporting independently sized full/SWA pools is
outside the proposed first quantization change.

## Validation

The [manual qualification guide](testing.md) provides the six-A100 deployment,
checkpoint preflight, and ordered numerical/serving commands for both models.
The [qualification report](qualification.md) records the measured environment,
retained verdicts, and accepted scope.

- **Source and loader:** strict metadata/key/header checks; independent E2M1
  nibble decoding and UE8M0 scale application, including signs, zeros, scale
  boundaries, interleaving, TP block boundaries, and malformed source cases.
  Inspect retained allocations to prove packed residency and bounded startup
  repack peak; numerical equivalence alone cannot prove this contract.
- **Mathematics:** biased Router logits, selected IDs/probabilities, asymmetric
  clipping, alpha, `up + 1`, both Expert biases, and route-weight placement.
  Use synthetic boundary cases plus real checkpoints. Tie handling must be
  explicit and compared with the chosen independent reference.
  Router regression covers both 32 and 128 Experts, single-row and multi-row
  inputs, original default-workspace SGLang/Torch routing, workspace-free
  FfnAgent construction, allocation-free replay, and actual native rebinding.
- **Operators:** compare an independently decoded FP32 formula for diagnosis
  and original SGLang FFN execution of the MXFP4 checkpoint for qualification.
  The A100 reference may expand one isolated layer using the upstream generic
  Triton fallback, keeping that expansion outside production. Cover
  small Decode rows, Prefill, tails, inactive Experts, and TP1/TP>1. Freeze
  backend-specific tolerances from baseline evidence before acceptance; do not
  relax them after observing CrossPool failures.
- **Native graphs:** Primary/Control topology agreement, scale/bias pointer
  coverage, repeated A/B layer rebinding, per-Lane scratch isolation, real TP
  delivery, routing-publication order, and cleanup after installation failure.
- **Serving:** no FFN parameters or Expert post-load allocations in Instance
  workers; independent native-SGLang reference; Eager, Decode Full, and combined
  Decode Full plus Prefill Breakable. Preserve the existing Decode token and
  Prefill KL/Graph Observer criteria. Verify GPT-OSS attention/KV behavior over
  a sliding-window boundary, including prefix reuse and capacity adjustment.
- **Resources:** rerun model-neutral memory calibration where affected and
  real-model baseline/pressure qualification with zero underprediction under
  the existing contract. Record model/checkpoint, backend/layout, hardware,
  toolchain, TP geometry, graph mode, and exact retained/staging/workspace bytes.
- **Final acceptance:** canonical native build/stubs, complete strict suite,
  affected model and resource qualifications, and documentation/code-quality
  review on the final source identity. Performance is report-only; report
  resident bytes, startup peak, FFN latency, and serving throughput/latency.

Use existing Unit, Integration, native CTest, and explicit Models suites under
`tests/`. Extend `xtest.harness.sglang.reference` with original GPT-OSS loading
and backend identity; load one layer at a time rather than a complete BF16
120b replica. It must not call CrossPool's production loader or decoder to
construct its expected result. Real-model cases resolve paths
through `xpool.config` and declare checkpoint/Device requirements.

## Out of Scope

Quantization-aware training, generating or requantizing checkpoints, general
AWQ/GPTQ/NVFP4 support, FP4 hidden-state transport, CPU/offloaded production
Expert weights, Expert Parallelism, another serving engine or platform,
heterogeneous FFN hardware, and performance guarantees. BF16/FP16 support for
existing models remains intact; GPT-OSS FP16 payload qualification is separate
from the proposed BF16 initial scope.

## Feasibility gates

Validate the exact tiled kernel rounding, fixed workspace layout, TP1/TP2
sharding, and A/B layer resource binding on SM80 before installed serving.
The implementation authorization covers these experiments and ordinary fixes
inside the accepted contract. A change to packed residency, graph ownership,
or the required two-model acceptance needs a new design decision.
