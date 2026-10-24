# OPT Family Support

## Goal

Add CrossPool support for the `OPTForCausalLM` family through one shared
biased-unary Dense FFN capability, then qualify the eight concrete OPT
checkpoints (`facebook/opt-125m`, `opt-350m`, `opt-1.3b`, `opt-2.7b`,
`opt-6.7b`, `opt-13b`, `opt-30b`, `opt-66b`). Each concrete checkpoint receives
its own deployment scenes, SGLang FFN numerical and serving-graph suite, and
support-table row after acceptance.

## Baseline

At `5d22ecf`, CrossPool admits only gated Dense and MoE FFNs. `ActivationKind`
has only `SILU`; `DenseFfnSpec` carries `GatedFfnCheckpointKeys`;
`DenseFfnWeights` packs gate/up/down without biases; the Dense operator uses
`torch.mm` + `sgl_kernel.silu_and_mul`; checkpoint coverage hard-codes
`model.layers.N.mlp`; the checkpoint reader accepts only safetensors. SGLang's
pinned `OPTDecoderLayer` inlines `fc1 -> activation_fn -> fc2` with biases and
has no MLP module. No `OPTForCausalLM` adapter exists.

## Accepted Changes

- Add `ActivationKind.RELU` and `DenseFfnLayout` (`GATED_SILU`, `BIASED_RELU`)
  while retaining `LayerKind.DENSE`.
- Add `BiasedFfnCheckpointKeys` (`fc1`/`fc2` weight and bias keys) and a
  `DenseFfnSpec.layout` discriminator over the gated/biased checkpoint union.
- Add `FfnModelSpec.checkpoint_namespace` so model families own their
  checkpoint prefix; the default preserves the current gated namespace.
- Add `BiasedDenseFfnWeights` with `fc1 [I_r,H]`, `fc1_bias [I_r]`,
  `fc2 [H,I_r]`, and owner-only `fc2_bias [H]`.
- Add a biased Dense execution signature and operator:
  `fc1 -> +fc1_bias -> ReLU -> fc2 -> +fc2_bias`, with exact `2*I_r` workspace.
- Extend the loader, registry, memory accounting, and native resource binding
  for the biased layout and rank-zero `fc2` bias ownership.
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

class DenseFfnLayout(IntEnum):
    GATED_SILU = 1
    BIASED_RELU = 2

class BiasedFfnCheckpointKeys(FfnModel):
    fc1_weight_key: str
    fc1_bias_key: str
    fc2_weight_key: str
    fc2_bias_key: str

class DenseFfnSpec(FfnModel):
    kind: Literal[LayerKind.DENSE]
    layout: DenseFfnLayout = DenseFfnLayout.GATED_SILU
    layer_id: int
    intermediate_size: int
    checkpoint: GatedFfnCheckpointKeys | BiasedFfnCheckpointKeys

class FfnModelSpec(FfnModel):
    # existing fields unchanged
    checkpoint_namespace: str | None = None

class DenseFfnExecutionSignature:
    # existing fields unchanged
    layout: DenseFfnLayout = DenseFfnLayout.GATED_SILU
    fc2_bias_present: bool = False

def dense_workspace_bytes(
    *, payload_dtype, row_capacity, local_intermediate_size,
    layout: DenseFfnLayout,
) -> int: ...

def compute_biased_dense_partial(
    *, hidden_states, layer_weights: BiasedDenseFfnWeights,
    workspace, output, activation: ActivationKind,
) -> torch.Tensor: ...
```

`FfnSourceConfig.validate_family_profile` gains optional `activation_field`,
`allowed_activations`, and `allowed_dtypes` parameters. Existing callers retain
the `hidden_act`/`silu`/BF16-FP16 defaults.

`filter_decoder_ffn_weights` gains an optional `pattern` argument. The SGLang
OPT adapter replaces `OPTDecoderLayer` with `XpoolOPTDecoderLayer` and filters
`model.decoder.layers.N.fc1/fc2` weights and biases. `require_mlp_ownership` is
extracted from `require_full_mlp_boundaries` for adapters without SGLang scatter
metadata.

## Data-Structure Changes

- `BiasedDenseFfnWeights.fc2_bias` is non-`None` only on FFN TP rank zero.
  Every rank owns a local `fc1_bias` slice; rank zero owns the full `fc2_bias`.
- `DenseFfnExecutionSignature` includes layout and `fc2_bias_present` so owner
  and non-owner ranks never share a captured graph.
- Biased workspace is `row_capacity * 2 * local_intermediate_size * dtype`
  bytes. The control probe reports `(H*I_r, I_r, H*I_r)` payload elements plus
  `H` for an owner's `fc2_bias`.
- `DenseBindingResourceProjection` and `LayerBindingValues` gain optional
  `fc1_bias_address` and `fc2_bias_address`. The native ABI advances from 84 to
  85. Existing two-address Dense resources remain valid as `None` biases.
- Checkpoint coverage for OPT restricts the FFN namespace to `fc1/fc2` stems so
  attention and normalization keys are not reported as extra FFN keys.

## Implementation

1. Add this plan and the `Biased Unary FFN` glossary term; no source changes.
2. Implement the schema, weights, execution signature, operator, loader,
   checkpoint namespace, registry, memory accounting, and native binding with
   focused unit and CTest coverage.
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

- Unit tests cover schema union/validators, biased weights and signatures,
  rank-zero bias ownership, workspace/probe accounting, checkpoint coverage,
  loader slicing, and memory placement.
- Integration/CTest cover the biased operator against a FP32 reference, rank
  partial summation with a single `fc2` bias contribution, graph capture and
  replay without steady allocation, native bias resource replacement, ABI 85,
  and gated/unary cross-binding rejection.
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
- `opt-66b` uses `ffn_tp=3`; the default serving scene is `atn3-ffn3` (six
  devices). A five-device `atn2-ffn3` variant remains unqualified.
- `opt-66b` conversion needs approximately 263 GB of temporary space.
- The native ABI bump invalidates prior memory-calibration profiles; resource
  evidence must be regenerated when the qualified environment changes.
