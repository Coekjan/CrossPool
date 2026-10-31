# OPT Family Support

## Goal

Add CrossPool support for the `OPTForCausalLM` family through one shared
biased-unary Dense FFN capability, then qualify the eight concrete OPT
checkpoints (`facebook/opt-125m`, `opt-350m`, `opt-1.3b`, `opt-2.7b`,
`opt-6.7b`, `opt-13b`, `opt-30b`, `opt-66b`). Each concrete checkpoint receives
its own deployment scenes, SGLang FFN numerical and serving-graph suite, and
support-table row after acceptance.

## Baseline

At `a3467ca`, the FFN execution boundary already carries per-projection
optional biases:

- `ProjectionLayout` / `ProjectionWeights` own independent `values`, `scales`
  and `bias` storages.
- Native Dense resources already bind `gate_up_bias_address` and
  `down_bias_address`; no ABI change is required.
- The loader reads a Dense output-projection bias only on FFN TP rank zero and
  leaves every non-owner candidate neutrally zero.

What is missing is Dense *semantics*: `ActivationKind` has only `SILU`,
`DenseFfnSpec` accepts only 3-projection gated sources, `DenseFfnWeights`
assumes a fused `2 x I` gate/up row extent, the Dense operator executes gated
SiLU only, and checkpoint coverage hard-codes `model.layers.N.mlp`. SGLang's
pinned `OPTDecoderLayer` inlines `fc1 -> activation_fn -> fc2` with biases and
has no MLP module. No `OPTForCausalLM` adapter exists.

## Accepted Changes

- Add `ActivationKind.RELU` and `ReluActivation`.
- Add `UnaryProjectionSpec` (`fc1`, `fc2`) and let `DenseFfnSpec.projections`
  be either gated or unary. The model-wide activation selects exactly one
  projection form.
- Add `FfnModelSpec.checkpoint_namespace` so model families own their
  checkpoint prefix; the default preserves the current gated namespace.
- Reuse `DenseFfnWeights` and `DenseFfnExecutionSignature`: unary layers map
  the input projection to the `gate_up` resource slot and the output projection
  to the `down` slot. Existing bias ownership and native binding remain
  unchanged.
- Size and execute the unary Dense operator as
  `fc1 -> +fc1_bias -> ReLU -> fc2 -> +fc2_bias`, with exact `2*I_r` workspace.
- Extend preparation, checkpoint coverage, and the loader for the unary
  projection source and Dense namespace.
- Add the engine-neutral `OPTForCausalLM` adapter and SGLang OPT decoder/FFN
  adapter. Preserve the pre-LN and opt-350m post-LN branches, projected
  embeddings, learned positions, tied output embeddings, and final-norm
  policy.
- Extend the isolated SGLang FFN reference and qualification harness to OPT's
  `model.decoder.layers` and inline `fc1/activation_fn/fc2` path with FP16
  payloads.
- Convert official `pytorch_model.bin` checkpoints to safetensors offline before
  qualification; do not extend the runtime reader in this plan.

## Interface Changes

```python
class ActivationKind(IntEnum):
    SILU = 1
    RELU = 2

class ReluActivation(FfnModel):
    kind: Literal[ActivationKind.RELU]

type ActivationSpec = SiluActivation | ReluActivation

class UnaryProjectionSpec(FfnModel):
    kind: Literal["unary"]
    fc1: ProjectionSpec
    fc2: ProjectionSpec

class DenseFfnSpec(FfnModel):
    # existing fields unchanged
    projections: GatedProjectionSpec | UnaryProjectionSpec

class FfnModelSpec(FfnModel):
    # existing fields unchanged
    checkpoint_namespace: str | None = None

def dense_workspace_bytes(
    *, payload_dtype, row_capacity, local_intermediate_size,
    activation: ActivationSpec,
) -> int: ...

def compute_dense_partial(
    *, hidden_states, layer_weights: DenseFfnWeights,
    workspace, output, activation: ActivationSpec,
) -> torch.Tensor: ...
```

`FfnSourceConfig.validate_family_profile` gains optional `activation_field`,
`allowed_activations`, and `allowed_dtypes` parameters. Existing callers retain
the `hidden_act`/`silu`/BF16-FP16 defaults. `unary_projections(prefix)` builds
the biased `fc1`/`fc2` sources.

`filter_decoder_ffn_weights` gains an optional `pattern` argument. The SGLang
OPT adapter replaces `OPTDecoderLayer` with `XpoolOPTDecoderLayer` and filters
`model.decoder.layers.N.fc1/fc2` weights and biases. `require_mlp_ownership` is
extracted from `require_full_mlp_boundaries` for adapters without SGLang scatter
metadata.

## Data-Structure Changes

- Unary `DenseFfnWeights.gate_up` holds `fc1` values `[I_r, H]` and its local
  bias `[I_r]`; `down` holds `fc2` values `[H, I_r]` and its output bias `[H]`.
- Every rank reads its local `fc1` value and bias slice. Only FFN TP rank zero
  reads `fc2` bias; every other rank keeps the prepared neutral zero. Partial
  summation therefore contributes the output bias exactly once.
- `DenseFfnExecutionSignature` needs no layout or bias-presence field: the
  activation identifies the operator body, and both owner and non-owner retain
  a full-shape output-bias storage.
- Unary workspace is
  `row_capacity * 2 * local_intermediate_size * dtype.itemsize` bytes. Gated
  workspace remains `3 x`.
- Checkpoint coverage for OPT restricts the FFN namespace to `fc1/fc2` stems so
  attention and normalization keys are not reported as extra FFN keys.
- Native Dense resource structures, the loader bias policy, and the ABI remain
  unchanged.

## Implementation

1. Add this plan and the `Biased Unary FFN` glossary term; no source changes.
2. Implement the schema, activation, preparation, workspace, operator,
   checkpoint namespace, and loader changes with focused unit coverage.
3. Add the engine-neutral OPT adapter and SGLang OPT decoder/FFN adapter with
   integration tests.
4. Extend the isolated SGLang FFN reference for OPT and FP16.
5. Qualify `opt-125m` first, then `opt-350m` (post-LN and projected embeddings),
   then `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-30b`, and
   `opt-66b`. Each model adds its deployment scenes, suite, and support row
   only after its suite passes.
6. Run the complete strict suite on one frozen source and native-module
   identity before merge.

## Validation

- Unit tests cover the projection union and activation validators, unary
  workspace and signature accounting, checkpoint namespace coverage, and the
  OPT source-config profile.
- Integration tests cover unary preparation and true-TP loading, the biased
  operator against a payload-typed reference, rank-owner output-bias semantics,
  graph capture and replay without steady allocation.
- SGLang integration tests cover hook installation, weight filtering,
  parameter-free shim construction, both LayerNorm branches, and the packed
  OPT weight loader.
- Each model suite runs `uv run xtest run --suite facebook/opt-<size>
  --strict-requirements` and retains numerical samples, serving graph
  evidence, cross-mode token/logit comparisons, Graph Observer evidence, and
  cleanup evidence.
- Final acceptance runs the complete strict-requirements suite with no
  unexpected skip, plus Ruff format/lint, ty, native build/CTest,
  clang-format, and Doxygen.

## Out of Scope

- Runtime `.bin` ingestion; this plan converts checkpoints offline.
- BF16 OPT checkpoints and `enable_bias=false` checkpoints.
- Quantized OPT attention or FFN execution.
- New `LayerKind` values or native Dense operator-kind enums.
- Falcon qualification; the biased-unary capability is family-neutral, but a
  separate concrete-model qualification is required.
- 66b execution when the required devices or conversion scratch space are not
  available; the code and suite are still added.

## Open Questions

None blocking design or implementation. The following are validation/resource
prerequisites tracked by the concrete model issues:

- `opt-30b` uses `ffn_tp=2`; `ffn_tp=3` is rejected because 28672 is not
  divisible by three.
- `opt-66b` uses `ffn_tp=4`; the serving scene is `atn2-ffn4` (six devices)
  because `ffn_tp=3` exceeds a 40 GB device's FFN residency plus execution
  overhead.
- The official OPT Crater repository publishes PyTorch, Flax and TensorFlow
  copies together; only the 132.4 GB PyTorch subset is eligible for offline
  conversion.
- opt-66b conversion and qualification require approximately 145 GB of
  available scratch space in addition to the converted checkpoint.
