# Gemma 4 E2B and 12B on SGLang

## Goal

Qualify `google/gemma-4-E2B-it` and `google/gemma-4-12B-it` as separate
Model IDs on the existing CrossPool CUDA and SGLang deployment. Serve the
original BF16 checkpoints through their native wrappers for text, image,
audio, and video input with text output. Disaggregate the decoder Dense
MLP in every layer.

## Baseline

The baseline is CrossPool `0d3267e` with SGLang `0.5.20`,
`sglang-kernel` `0.4.7`, and Transformers `5.12.1`. The existing
[model catalogue](../../supported-models.md) lists both Models as coverage
candidates. [FFN Execution](../../designs/ffn-execution.md) defines the
existing gated Dense preparation, weight slicing, and Graph contracts.
[Elastic Context State](../../designs/elastic-context-state.md) defines
ownership of Full and SWA storage. [Research evidence](research.md)
documents the source revisions, checkpoint headers, and pinned serving paths.

| Checkpoint property | `google/gemma-4-E2B-it` | `google/gemma-4-12B-it` |
| --- | --- | --- |
| Wrapper | `Gemma4ForConditionalGeneration` | `Gemma4UnifiedForConditionalGeneration` |
| Root / text `model_type` | `gemma4` / `gemma4_text` | `gemma4_unified` / `gemma4_unified_text` |
| Decoder layers / hidden width | 35 / 1536 | 48 / 3840 |
| Dense intermediate width | 6144 at layers 0–14, 12288 at layers 15–34 | 15360 at all layers |
| FFN activation | Gated `gelu_pytorch_tanh` | Gated `gelu_pytorch_tanh` |
| Per-Layer Embedding (PLE) | Width 256 | Disabled |
| Allocated Full / SWA layer slots in pinned SGLang | 7 / 28 | 8 / 40 |
| Full / SWA KV heads × head width at attention TP1 | 1 × 512 / 1 × 256 | 1 × 512 / 8 × 256 |
| SWA window | 512 tokens | 1024 tokens |
| Layer-level KV semantics | Layers 15–34 read from layer 13 (SWA) or 14 (Full) | Every layer owns its KV. Full layers use tied K/V projection weights |

Both wrappers instantiate `Gemma4TextModel` and call
`Gemma4DecoderLayer.forward`. The Dense implementation is
`gemma4_causal.Gemma4MLP`, an alias of `Gemma3MLP` in the pinned engine.
The FFN checkpoint keys and MLP constructor prefixes are rooted at
`model.language_model.layers`. SGLang exposes the loaded text module at
`wrapper.language_model` and its weight loaders strip the leading `model.`.

CrossPool's current FFN activation declaration holds only SiLU. Checkpoint
validation and the common SGLang weight filter currently select
`model.layers.<id>.mlp`. The existing Dense loader and planner already
consume each layer's declared intermediate size independently. The ECS layout
builder already mirrors SGLang's asymmetric Full/SWA pool geometry through
`is_hybrid_swa_compress`.

SGLang's automatic backend selection disables Prefill Breakable for both
multimodal Gemma architectures. The current three-mode qualification
explicitly requests Breakable, which bypasses that automatic rule. The
pinned runner then supplies both dummy token IDs and embeddings to the
inner decoder during capture. Gemma's text forward requires exactly one
input carrier. Replay also needs to carry E2B's `per_layer_inputs`. These
source findings require CUDA reproduction. [Research evidence](research.md).

## Accepted changes

- Share one Gemma 4 Dense FFN compiler implementation across the two outer
  architectures. Scope admission to the two Model IDs and their verified
  unquantized checkpoint profiles. Keep one SGLang shim adapter that owns the
  shared decoder seam and two wrapper-specific loader hooks.
- Use the existing `DenseFfnSpec` per-layer widths, TP slicing, Dense
  activation workspace, and FFN delivery path. Use the upstream Dense gated
  tanh GELU activation. Preserve the current placement of attention, multimodal
  embedding, PLE, post-FFN norms, residuals, and layer scaling on the
  Instance Rank.
- Reuse the existing ECS Full/SWA pools, allocator, VMM backing, and quota.
  Qualify Gemma layouts and cache behavior with the real upstream factory.
- Set the first support boundary to Eager prefill plus Eager or Decode
  Full decode for these two Model IDs. Reject an explicitly resolved
  Prefill Breakable configuration before load, with a message identifying
  the expected SGLang backend. Preserve the repository's combined graph
  qualification for its already-qualified Models.
- Qualify E2B before 12B. First prove one attention TP1/DP1 and FFN TP1
  deployment per Model. Add attention TP2/DP1 and FFN TP2 evidence through
  the portable deployments. Keep attention DP greater than one outside
  initial admission until the decoder row and collective contract is proven.

### Implementation constraints

- Use the installed ECS, KV Pool, and native ABI contracts for both Models.
- Keep tanh GELU a model-neutral activation of the existing Dense FFN
  specification and execution layout.
- Scope hook changes to the Gemma 4 adapter. Extend the existing compiler and
  test harness only at the checkpoint namespace, activation, reference
  selection, and serving qualification seams used by these Models.

## Interface and data-structure changes

**Model activation.** Use the upstream `GeluActivation(approximate="tanh")`
with the existing default `GateTransform`. The compiled activation and
projection layout flow through preparation, execution signatures, and Dense
Graph capture.

**Checkpoint validation.** Use the upstream `FfnModelAdapter.checkpoint_prefixes`
contract. Both Gemma 4 compilers declare the exact
`model.language_model.layers.<layer_id>.mlp.` subtree for each layer. The loader
checks exact key coverage and shapes through the existing checkpoint path.

**SGLang boundary.** In `src/xpool/integrations/sglang/models/gemma4.py`,
preserve the pinned decoder constructor signature. Create an
`FfnShimModule` with the Dense kind, decoder layer ID, and hidden width
directly in that constructor. Use its existing
`forward(hidden_states, forward_batch=...)` API. The two wrapper loader
hooks remove only the three original decoder projection keys for each layer
and forward the remaining `Iterable[tuple[str, torch.Tensor]]` to the
corresponding original loader. Preserve its return value and upstream
loaded-parameter accounting.

**Qualification tools.** Extend the independent SGLang FFN reference to
select `wrapper.language_model.layers[i].mlp` for these two concrete
wrappers. Use the existing subprocess and reference result format. Extend
the Dense representative-layer rule to include both sides of any adjacent
intermediate-width transition. Keep first and last layers and existing
Dense/MoE transition evidence. Supply model-owned media requests through
the existing `run_probe(..., workload=...)` callback. Let
`E2eServingCase` and `SglangLaunchModel` carry optional
`attention_backend: Literal["triton"] | None` and
`chunked_prefill_size: Literal[-1] | None` test launch settings, both
defaulting to `None`. The serving command appends the corresponding flags
when configured. Retain per-request media payload identity, response,
resolved settings, and matching observer evidence.

## Implementation

### 1. Compile and execute the Dense FFN

Resolve each intended checkpoint with `XpoolConfig.model_path_of(model_id)`.
Require a configured Model ID, local config, and complete original
checkpoint. Confirm the effective BF16 dtype from the published root
`dtype`. If the text config declares a dtype, require agreement. Read
`hidden_activation`, widths, layer count, `num_kv_shared_layers`, and
`use_double_wide_mlp` from `text_config`. The two architecture entries in
`src/xpool/runtime/ffnagent/models/gemma4.py` can share Dense compilation.
Reject an architecture/Model ID mismatch, invalid nested fields,
quantization, or enabled MoE before the generation loads weights.

Use the ordered three separate weight sources at
`model.language_model.layers.<id>.mlp.{gate_proj,up_proj,down_proj}.weight`.
Gate and Up each have source shape `[intermediate, hidden]`. Down has
`[hidden, intermediate]`. Double the configured width for the shared suffix
only when the pinned constructor enables double-wide MLP for that layer.
The Dense loader packs local Gate rows then local Up rows and slices the
matching Down columns using the existing FFN Execution Group geometry.

The upstream gated GELU operation owns its execution row order and writes
to caller-owned storage. Qualify its projection and activation rounding
through the installed Dense operator and Graph replay.
Run SGLang's independently loaded original MLP at the same dtype and FFN
TP width. The MLP owns its native TP reduction there. CrossPool's FfnAgents
deliver the complete result through existing Fabric/Transport semantics.

### 2. Adapt the native text decoder

Target `sglang.srt.models.gemma4_causal.Gemma4DecoderLayer.__init__`
and `.forward` with their pinned signatures. SGLang's hook registry
propagates class replacements by identity through loaded modules, so replacing
the `Gemma4MLP` alias would also replace the original `Gemma3MLP` binding.
Port the pinned Dense constructor branch at the Gemma 4 decoder boundary.
Instantiate the upstream attention, norms, PLE projections and persistent
`layer_scalar` in their original order, and install the parameter-free shim
at `self.mlp` before allocating any local FFN projection. With
`supports_dp_attention=False`, the existing rank binding rejects attention
DP greater than one. Keep the Dense forward's attention calls, fused and
unfused post-FFN paths, PLE, and `(hidden_states, None)` return. The FFN
substitution passes the actual
`ForwardBatch` at the normalized MLP input and consumes the full returned
`[num_tokens, hidden_size]` tensor at the existing post-FFN normalization.
Fabric delivery owns the MLP TP reduction. The attention output projection
retains its own reduction.

After load, require the exact native wrapper and all expected parameter-free
Dense shims in `language_model.layers`, each with the expected ordinal,
type, and hidden width. Verify the complete FFN boundary from the pinned
decoder call sites and real TP evidence. Gemma's decoder does not publish
the `LayerScatterModes` metadata used by other architectures. Reject
unsupported runner modes via the existing plugin lifecycle and model-profile
checks before serving readiness. Use the existing attach and retirement
lifecycle.

For the Unified wrapper, derive the missing `lm_head_is_tied` flag from
the head and embedding module identities before CUDA Graph capture. Apply
the same pinned-version compatibility fix to the independent 12B media
reference. [SGLang fix](https://github.com/sgl-project/sglang/pull/35809).

### 3. Prove ECS compatibility and native input

Keep the existing `SWAKVPool` and `SWATokenToKVPoolAllocator` with elastic
children. Check the ordered Full/SWA child descriptors, per-rank head
counts, VMM row bytes and mapping to the actual upstream layer IDs.
E2B's donor aliases and disabled writes for the shared suffix stay with
SGLang's attention implementation. Size VMM and quota from the pinned pool's
35 allocated layer slots. In 12B, preserve the loader's duplication of the
Full K projection into V and the independently sized Full and SWA buffers.
Test with the Python Unified Radix cache and BF16 KV storage.

Use SGLang's native `/v1/chat/completions` endpoint and small deterministic
fixtures for each media type. Send an `image_url` part for images,
`input_audio` with Base64 WAV for audio, and a `video_url` part for video.
Send local fixture bytes inline. Use the Triton attention backend
and `--chunked-prefill-size -1` in these bounded media cases. This preserves
the full image span for the native wrapper's bidirectional image mask.
Use the same settings in the pinned native SGLang reference and the
CrossPool run. Explicitly resolve and record BF16 and the native multimodal
wrapper in both. Verify real media reaches the processor, generates decoder
rows, and influences the result using paired distinguishable media with
the same prompt. Require the first greedy token to match the native
reference. When two media inputs produce the same leading token, require
the change in its confidence to agree in direction. Include a
capture-capable Decode Full run with Eager prefill and retain observed
graph and FFN activity. Keep media collection and parity at the model-suite
boundary.

### 4. Qualify the graph boundary

Use the existing Eager and Decode Full model qualification for each Model. Disable
Prefill Graph capture explicitly in both configurations. Require the server's resolved
graph settings and Graph Observer evidence, and run the same native
wrapper for text and media. For these Models, run the real ECS prefix and
reclamation cases with Decode Full plus Eager prefill. Record the actual
graph settings alongside retained cache evidence. Preserve the mandatory
combined-mode ECS regression with its existing portable deployments.

A later Prefill Breakable extension can use the same SGLang wrappers once
the runner supports one legal input carrier during capture, replay-stable
per-layer inputs for E2B, and the native image-mask path or explicit media
eager fallback. That extension needs its own runner and graph validation
before it can expand the support boundary.

## Validation

| Boundary | Evidence required |
| --- | --- |
| FFN source and pure preparation | Verify exact checkpoint roots and ordered layer keys for both wrappers, PLE and tower keys preserved by filtering, E2B 14/15 width change, 12B uniform width, untouched Gemma 3 MLP binding, unchanged SiLU digest and default decoding, unsupported activation/encoding combinations and invalid profile failures. |
| Dense operator | BF16 and FP16 caller-owned output, reference arithmetic and Graph replay with the existing qualification tolerances. Distinguish GELU and SiLU signatures while reusing Dense storage shapes. |
| Real checkpoint FFN | E2B layers 0, 14, 15, 34 and 12B layers 0, 47. Compare deterministic input rows in original BF16 against the unpatched SGLang MLP. Record every attention rank's complete results. Test FFN TP2 slicing and reduction in a portable second geometry. |
| Installed serving | Text through the original wrappers, Eager vs Decode Full token ID parity with Eager prefill, observed Decode Full execution, and an early diagnostic for an explicitly requested Prefill Breakable backend. Retain the native TP1 and TP2 evidence and shutdown proof. |
| Native media | For each Model, run image, audio, and video cases with local paired fixtures and text output. Compare CrossPool and pinned native SGLang with identical prompt, media, and sampling settings. Verify the native image mask and actual Decode Graph replay in their respective paths. |
| ECS | Real Full/SWA pool and VMM descriptor/byte agreement, distinct paired locations, and prefix quality/retention with prompts beyond the SWA window. Exercise E2B donor reads across prefill and decode, 12B Full head replication at attention TP2, prefix hit and reclaim under pressure, and checked retirement. Compare final prefill distributions after reclamation with the existing KL policy. |

Add model-owned suites under
`tests/suites/models/google/{gemma-4-E2B-it,gemma-4-12B-it}/test_sglang_model_qualification.py`
and their portable deployments under `configs/deployments/` using the
existing Model ID encoding. At FFN TP1 the raw Dense BF16 weights account
for approximately 2.90 GiB (E2B) and 15.82 GiB (12B). Use runtime memory
admission and real resource evidence for the additional Graph, workspace,
and attention-side memory. On the frozen implementation and native build,
run the canonical strict suite, each explicit Model suite, and the shared
qualification invalidated by activation, loader, or ECS changes.

```bash
uv run xtest run --strict-requirements
uv run xtest run --suite google/gemma-4-E2B-it --strict-requirements
uv run xtest run --suite google/gemma-4-12B-it --strict-requirements
```

Record model revision, effective serving settings, and evidence with each
verdict. Update the two catalogue rows after their respective evidence
meets the accepted boundary.

## Out of scope

Other Gemma variants, quantized checkpoints, MTP, speculative decoding,
LoRA, new serving engines, Prefill Breakable for these two Models, and changes
to the Transport or native Fabric ABI are outside this proposal. Donor-only
KV packing and moving PLE to FfnAgents are separate optimizations.
