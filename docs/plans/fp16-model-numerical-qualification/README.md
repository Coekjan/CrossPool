# FP16 Model Numerical Qualification

## Goal

Allow the model-owned SGLang FFN numerical qualification to exercise an
explicitly declared FP16 payload while preserving the existing BF16 cases and
their numerical policy. This unblocks qualification of concrete FP16 models,
including `huggyllama/llama-7b`, without converting their reference inputs or
production execution to another dtype.

## Baseline

`E2eFfnInputMatrix` declares only a seed and row counts. The numerical harness
therefore generates BF16 inputs, constructs an `InstanceFfnProfile` with a BF16
payload, and requires BF16 reference and production outputs. This works for the
currently qualified BF16 models but rejects an FP16 SGLang reference before
numerical execution.

The runtime FFN operators, checkpoint loader, weight owners, and execution
signatures already admit both BF16 and FP16. The missing capability is limited
to qualification input declaration, propagation, comparison, and retained
evidence.

## Proposed Changes

- Make the numerical case's payload dtype explicit rather than deriving or
  guessing it from a checkpoint.
- Admit exactly `bfloat16` and `float16`, matching the production FFN payload
  boundary.
- Preserve BF16 as the default for existing source-owned model cases.
- Require one dtype to flow unchanged through generated inputs, the independent
  SGLang reference, the production `InstanceFfnProfile`, saved invocation
  inputs, output comparison, and the retained numerical evidence.
- Keep the existing error metrics and tolerance constants unchanged for both
  admitted dtypes. Qualification evidence, rather than a dtype-specific relaxed
  threshold, determines acceptance.

## Interface Changes

In `xtest.harness.support.ffn`, add the public harness type alias:

```python
type FfnPayloadDtypeName = Literal["bfloat16", "float16"]
```

Add a resolver that maps those names to `torch.bfloat16` and `torch.float16`
and fails for any value outside the closed vocabulary.

Change `generate_ffn_hidden_states` from:

```python
generate_ffn_hidden_states(*, hidden_size, seed, row_counts)
```

to:

```python
generate_ffn_hidden_states(*, hidden_size, seed, row_counts, dtype)
```

where `dtype` is an admitted Torch dtype and every returned tensor retains it.

`compare_ffn_outputs` continues to require equal reference and actual dtypes,
but accepts either admitted payload dtype instead of requiring BF16.

## Data-Structure Changes

Extend `E2eFfnInputMatrix` with:

```python
payload_dtype: FfnPayloadDtypeName = "bfloat16"
```

The default preserves every existing source-owned case. FP16 model suites must
declare `payload_dtype="float16"` explicitly.

Extend `FfnNumericalEvidence` with the same `payload_dtype` name. The runner
writes the declared dtype after reference and production outputs have both
passed exact dtype validation. Existing stored artifacts are offline results,
not inputs consumed by a compatibility reader; no migration or fallback field
is added.

## Implementation

1. Add the closed dtype name, resolver, dtype-parameterized deterministic input
   generation, and dual-dtype output validation to the shared FFN numerical
   support module.
2. Add `payload_dtype` to `E2eFfnInputMatrix`, resolve it once in
   `run_numerical_case`, and use the resolved Torch dtype for generated inputs
   and `InstanceFfnProfile.payload_dtype`.
3. Record the declared dtype in `FfnNumericalEvidence`.
4. Mark the `huggyllama/llama-7b` numerical matrix as `float16`; leave existing
   model suites on their default BF16 declaration.

## Validation

- Unit tests cover deterministic BF16 and FP16 input generation, rejection of
  unsupported Torch dtypes, equal-dtype comparison for both admitted dtypes,
  and rejection of mixed or unsupported output dtypes.
- Catalogue/model validation proves the default remains BF16 and accepts an
  explicit FP16 matrix.
- Existing BF16 focused numerical-support and reference-runner tests remain
  passing.
- Run the complete strict `huggyllama/llama-7b` SGLang qualification on one
  frozen source and native-module identity. It must pass independent FFN
  numerical comparison and all declared serving graph modes without skips.
- Final repository acceptance follows `docs/designs/qualification.md`, including
  the complete strict suite and affected static quality checks before support is
  claimed.

## Out of Scope

- Inferring payload dtype from checkpoint metadata or SGLang runtime state.
- Adding FP32, FP8, quantized, or mixed-payload qualification.
- Changing production dtype admission, numerical tolerances, operators, Router
  behavior, serving graph criteria, or checkpoint conversion policy.
