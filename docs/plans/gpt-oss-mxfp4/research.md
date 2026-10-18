# GPT-OSS MXFP4 research

This note records source inspection on 2026-10-15 for the [GPT-OSS
plan](README.md). It distinguishes upstream capabilities from CrossPool
qualification. No model execution, GPU benchmark, or graph compatibility test
was performed for this research.

The agreed deployment constraint is **seven A100 40 GB GPUs**, with **both
20B and 120B required in initial acceptance**. Initial deployment may disable
hybrid SWA memory allocation. Hopper/Blackwell paths below explain upstream
behavior; they do not satisfy this hardware constraint.

## Evidence baseline

The repository pins SGLang **0.5.20**, sglang-kernel **0.4.7**, PyTorch
**2.13.0**, and Triton **3.7.1** in [pyproject.toml](../../../pyproject.toml).
[uv.lock](../../../uv.lock) resolves FlashInfer **0.6.18**. Installed Python
sources were inspected directly; the GPT-OSS model, MXFP4 quantization module,
and low-level Marlin Python wrapper match their wheel `RECORD` hashes.

The upstream release page currently identifies **v0.5.20** as the latest stable
release; its release commit is `94602c9`. There is therefore no newer stable
SGLang version to adopt for this investigation. Upstream `main` has additional
commits, but was not treated as an interchangeable implementation baseline.
Release-tag links below identify source owners; local wheel source is the
authority for the exact installed implementation. [SGLang releases](https://github.com/sgl-project/sglang/releases),
[v0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20).

## Detection and loading

SGLang does not infer quantization from the model's marketing name or from the
requested activation dtype. `ModelConfig._verify_quantization()` reads the HF
quantization configuration, resolves `quant_method`, and checks any explicit
quantization request for compatibility. The official GPT-OSS configuration
contains `quantization_config.quant_method = "mxfp4"`; its excluded modules are
attention, router, embedding, and output head. The 20B and 120B configurations
share hidden/intermediate width 2880 and top-k 4. They differ in layers/experts:
24/32 for 20B, 36/128 for 120B. [Model configuration source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/model_config.py),
[20B configuration](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json),
[120B configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json).

`Mxfp4Config.from_config()` recognizes serialized MXFP4. For a `FusedMoE`
layer, `get_quant_method()` returns `Mxfp4MoEMethod`; the dynamic method is a
different path for quantizing nonserialized weights. `GptOssForCausalLM`
dispatches loading to `_load_weights_mxfp4()`, which separates expert tensors
from ordinary model tensors. `_load_mxfp4_experts_weights()` maps the following
HF names, using `model.layers.<layer>.mlp.experts.` as their prefix:

| Checkpoint suffix | SGLang parameter | Logical shape before backend padding |
| --- | --- | --- |
| `gate_up_proj_blocks` | `w13_weight` | `[E, 2I, H/32, 16]`, flattened to `[E, 2I, H/2]` bytes |
| `gate_up_proj_scales` | `w13_weight_scale` | `[E, 2I, H/32]` bytes |
| `gate_up_proj_bias` | `w13_weight_bias` | `[E, 2I]` |
| `down_proj_blocks` | `w2_weight` | `[E, H, I/32, 16]`, flattened to `[E, H, I/2]` bytes |
| `down_proj_scales` | `w2_weight_scale` | `[E, H, I/32]` bytes |
| `down_proj_bias` | `w2_weight_bias` | `[E, H]` |

The loader handles EP expert slicing and TP intermediate slicing in whole
32-value blocks. Its down-projection bias is populated only on TP rank zero
and zeroed elsewhere, so the later sum includes that bias once. Its current
implementation moves a yielded tensor to the weight device before narrowing
the shard; CrossPool should not copy this staging order without accounting
for the extra peak GPU memory. [GPT-OSS loading source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/gpt_oss.py).

MXFP4 here means packed **E2M1** values, two per byte, with one **UE8M0** scale
per 32 logical values. A valid finite scale byte `s` represents `2**(s-127)`;
127 is the neutral scale. Low nibble precedes high nibble. The positive E2M1
values are 0, 0.5, 1, 1.5, 2, 3, 4, 6, with corresponding sign-bit variants.
Ignoring padding and biases, storage is 17 bytes per 32 weights, or **4.25
bits/weight**. This differs from NVFP4's scale conventions. OpenAI's Torch
reference expands the checkpoint into BF16 for simplicity; it is a numerical
reference, not evidence of packed GPU residency. [OpenAI checkpoint decoder](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py).

Backend preparation may transpose, interleave, pad, or repack the bytes.
`torch.uint8`, `torch.int8`, `torch.int32`, or a custom FP4 tensor wrapper can
all represent packed weights; dtype names alone cannot establish effective
precision or residency. Conversely, activation BF16 does not imply BF16
resident weights. Inspect both projection storage, their scales, and actual
kernel arguments. [MXFP4 preparation source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/quantization/mxfp4.py).

## Exact FFN semantics

Let `x` be the normalized FFN input. The router is an ordinary biased linear
layer. For the selected top-four experts, weights are normalized softmax
probabilities. For each selected expert `e`:

```text
router_logits = x @ router_weight.T + router_bias
ids, p = normalized_topk_softmax(router_logits, k=4)
[gate, up] = split_interleaved(x @ dequant(W13[e]).T + b13[e])
g = min(gate, 7)
u = clamp(up, -7, 7)
a = g * sigmoid(1.702 * g) * (u + 1)
expert_output[e] = a @ dequant(W2[e]).T + b2[e]
y = sum(p[e] * expert_output[e])
```

This is a definition of the math, not a proposal to materialize `dequant(W)`.
The gate has only an upper clamp. Biases occur before activation and before
weighted combination. The interleaved row order is
`[gate0, up0, gate1, up1, ...]`; a generic contiguous-half SwiGLU loader would
silently compute a different function. RMSNorm/residual ownership must stay at
the existing integration boundary. OpenAI's reference confirms this formula.
[OpenAI FFN reference](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py).

In the pinned SGLang implementation, `GptOssSparseMoeBlock` configures
`TopK(renormalize=True)`, `with_bias=True`, `gemm1_alpha=1.702` by default, and
the configured `swiglu_limit`. The router uses the model activation dtype;
the standard CUDA top-k path exposes **FP32 weights and int32 IDs**. Specialized
FlashInfer routing can instead pack IDs and BF16 probability bits, so parity
must name the concrete routing path, not merely “SGLang.” [GPT-OSS block](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/gpt_oss.py),
[top-k implementation](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/topk.py).

## GPU compute paths in the installed stack

The selection is jointly determined by checkpoint format, architecture,
requested runner, and available kernel packages. `_gpt_oss_overrides()` forces
BF16 model activations for the native MXFP4 checkpoint. Its automatic choices
include FlashInfer on SM100/SM120; other branches have extra conditions.
Explicit runner selection is preferable for reproducible qualification.
[GPT-OSS configuration overrides](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/arg_groups/model_overrides/gpt_oss.py).

| Path | Resident weights and arithmetic | Relevance to CrossPool |
| --- | --- | --- |
| Marlin MXFP4 | Repacked FP4 weights and E8M0 scales; BF16 activations; FP4 decode/scaling within GEMM | Stock SGLang rejects SM80; low-level reuse is an optional unqualified optimization |
| FlashInfer CUTLASS, SM90 | Packed FP4 with group scaling; default W4A16 mixed-input GEMM | Existing Hopper option; no native FP4 MMA assumption; inspect descriptor/workspace behavior |
| FlashInfer TRT-LLM, SM100 | Packed FP4 with native block-scaled path and MXFP8 activation preparation | Hardware-accelerated follow-on candidate; activation quantization is an additional numerical change |
| FlashInfer CUTLASS, SM120 | Packed MXFP4 plus MXFP8 activation scaling | Separate hardware qualification; not automatically interchangeable with SM100 |
| OpenAI `triton_kernel` | Custom FP4 tensor and scale layouts; BF16 external input; device-specific matmul implementation | TMA descriptor construction complicates parameterized graph reuse |
| Generic fallback under `Mxfp4MoEMethod` | `upcast_from_mxfp()` creates persistent BF16 expert parameters | Does not satisfy packed-residency requirements |

These paths are present in `Mxfp4MoEMethod.__init__`,
`process_weights_after_loading`, and `apply`. Marlin's current MXFP4 path
requires BF16 activations and accepts SM90/SM100/SM120 in SGLang's guard;
`get_min_capability() == 80` on the quantization config alone is not proof that
a specific backend is supported. DeepGEMM and AMD/XPU methods also appear in
the source, but their presence does not qualify GPT-OSS with CrossPool.
[MXFP4 source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/quantization/mxfp4.py),
[Marlin MoE implementation](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py).

SM90 Humming W4A8 is opt-in and has version-sensitive FlashInfer APIs. The
SGLang wrapper explicitly notes that FlashInfer 0.6.17 reverted that API;
availability and numerical behavior on the pinned 0.6.18 installation are
not established here. It should not be the initial fallback. The generic
BF16-upcast fallback must be rejected explicitly rather than silently used
when an intended packed backend is unavailable. [FlashInfer runner source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py).

## Parameterized graph compatibility

Ordinary SGLang CUDA-graph support does not prove compatibility with a
CrossPool graph whose layer resources are rebound. The following findings
come from the installed source and motivate a bounded implementation probe:

1. **Marlin has a promising low-level interface, but no qualified A100 path.**
   `sglang/kernels/ops/moe/moe_wna16_marlin.py` calls the JIT module's
   `moe_wna16_marlin_gemm` with explicit input/output, packed weights, biases,
   scales, routing buffers, lock workspace, and temporary buffers. The
   corresponding `gemm/marlin_moe/kernel.h` ABI exposes raw pointers, and the
   inspected implementation uses shared-memory staging rather than embedded
   TMA descriptors. However, the Python wrapper allocates `c_tmp` for FP32
   reduction, optional empty tensors, and sometimes output. The higher-level
   runner additionally allocates per-call locks, sorting buffers, activation
   caches, and output. A reusable adapter must provide all those buffers,
   prepare optional empty arguments once, and replace the Torch activation
   expression with an explicitly bounded implementation. Passing a workspace
   to the high-level function alone is insufficient. The JIT native source
   includes an SM80+ implementation and BF16/E2M1/E8M0 instantiations; the
   higher-level MXFP4 method nevertheless explicitly rejects SM80. This
   mismatch is a reason for a bounded future probe, not permission to remove
   the guard. Generic Marlin helpers also disable their BF16 atomic-add
   optimization below SM90. MXFP4's fused MoE path already requests FP32
   scratch reduction with atomic adds disabled.
2. **Bias and probability order are visible in Marlin.**
   `marlin_template.h` adds expert bias in the final reduction and multiplies
   the down-projection output by top-k weight afterward. Both GEMMs receive
   biases; the second has `mul_topk_weights=True`. Its output rounding and
   probability multiplication still need numerical comparison against the
   selected reference.
3. **FlashInfer exposes reusable workspace, but needs a native audit.**
   The installed `flashinfer/fused_moe/core.py::cutlass_fused_moe` accepts
   `output` and `workspace_buffer`; `cutlass_fused_moe_workspace_size` sizes
   scratch ahead of execution. SGLang's GPT-OSS wrapper does not pass that
   workspace and allocates output/padding itself. The included CUTLASS MoE
   headers contain TMA-specialized grouped-GEMM structures. Whether the
   selected tactic embeds resource addresses in descriptors or refreshes them
   from exposed pointers must be established on actual captured nodes.
4. **OpenAI Triton native MXFP requires persistent/TMA execution.**
   Installed `triton_kernels/matmul.py` constructs descriptors for weights and
   scales in persistent mode; it rejects native-MXFP configurations without
   persistent/TMA support. This is a concrete reason not to assume that the
   stock wrapper can be rebound by replacing a few raw pointers.

Source owners: [Marlin Python GEMM wrapper](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/kernels/ops/moe/moe_wna16_marlin.py),
[Marlin native kernel](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h),
[Marlin runner](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/moe_runner/marlin.py),
[FlashInfer core](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/fused_moe/core.py),
[OpenAI Triton integration](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py).

Optional optimization probe: compile the low-level Marlin path for SM80;
execute one complete
biased/clamped top-four FFN at fixed capacity; enumerate captured nodes and
resource-pointer occurrences; rebind every projection, scale, router, and bias
between two deliberately different layers; replay A/B/A; verify scratch
ownership and concurrent-lane isolation. Success requires no undeclared
resource address, no opaque layer-bound descriptor, no hidden resident BF16
expert copy, and no allocation during replay. This recommendation is an
inference from the interfaces above, not completed qualification.

## Recommended A100 compute contract

The initial implementation should use a narrowly scoped raw-pointer Triton
W4A16 path owned by CrossPool: decode E2M1 nibbles and apply their UE8M0 block
scales **inside each GEMM tile**, convert the tile to BF16, and use BF16
`tl.dot` with FP32 accumulation. Both expert projections remain packed in
global GPU memory. Only tile values and bounded activation/intermediate
buffers are expanded; no full layer or full expert BF16 weight cache belongs
to the serving implementation. A100 supports BF16 Tensor Core arithmetic
with FP32 accumulation; it does not provide Blackwell's native FP4 path.
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/),
[Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

The recommendation is a new implementation design, not a claim that a stock
SGLang operator already provides this contract. Use fixed launch capacities,
explicit routing buffers and output masks, and declared raw pointers for
packed weights, scales, biases, and all scratch. Avoid TMA descriptors and
backend autotuning during capture/replay. Apply GPT-OSS's activation after
the first biased projection and routing weights after the second biased
projection. Test tail tiles independently: H=I=2880 is divisible by 32 and 64
but not by 128; TP shards must retain complete scale blocks and explicitly
zero any padded lanes.

For independent numerical checks on A100, decode only one layer/shard at a
time into a BF16 SGLang FFN reference, plus a small explicit Torch reference
for byte decoding and the biased/clamped formula. Do not expand the entire
120B model into BF16 merely to obtain a comparator. A whole-model SGLang
BF16 fallback has a separate, much larger capacity requirement and would not
prove packed residency. The current normal GPT-OSS
loader and BF16 fused-MoE path supply the reference semantics; exact
constructor/runtime-context setup still belongs to the test implementation.
[Normal GPT-OSS loading](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/gpt_oss.py),
[SGLang fused-MoE source](https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/layers/moe/fused_moe_triton).

## Serving integration gate

GPT-OSS alternates full and sliding-window attention and uses attention sinks.
The installed `ModelConfig._derive_hybrid_model()` selects hybrid SWA storage
unless `disable_hybrid_swa_memory` is set. Disabling that storage policy does
not remove `layer.sliding_window_size`: attention forward paths still derive
the attention window from each layer. On A100 the GPT-OSS override defaults
to **Triton attention**. Its standard extend path forwards both `sinks` and
`sliding_window_size`; its decode path selects window KV indices per layer
and forwards sinks. The initial candidate is therefore `attention_backend =
"triton"`, DCP size one, and `disable_hybrid_swa_memory=True`, retaining the
existing single-pool Elastic KV boundary while preserving attention math.
Do not use FA3, which is a Hopper choice, or assume FA2 supports the required
sink variant. This is a source-supported integration hypothesis; actual pool
selection, sink handling, sliding-window correctness, and memory accounting
still need serving qualification.
[Model configuration](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/model_config.py),
[Triton attention backend](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/attention/triton_backend.py).

## Remaining evidence

- Exact configured checkpoint revision, safetensors headers, and available
  devices must be recorded in qualification evidence; the public HF links
  above are mutable model references, not pinned local checkpoints.
- The chosen kernel's numerical tolerance must cover routing ties, both
  biases, clipping boundaries, padding, and TP reduction. Backend equivalence
  cannot be inferred from a shared quantization label.
- Peak load memory includes source tensors, repacked destination, scales,
  padding, temporary conversion storage, and workspace. Packed steady-state
  storage alone is not a load-capacity guarantee.
- 120B requires its own memory and serving evidence even though its FFN
  formula matches 20B. Both are mandatory for initial acceptance on the seven
  A100 40 GB cards; missing 120B evidence means acceptance is incomplete.
- Native Blackwell arithmetic and Hopper software-decoded arithmetic require
  separate numerical/performance evidence and graph-node inspection.
