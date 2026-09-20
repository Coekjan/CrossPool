---
name: add-model-support
description: Add and qualify CrossPool support for a concrete locally configured model, reusing an existing SGLang adapter when its FFN architecture permits.
---

# Add Model Support

Add one concrete Model ID without broadening the architecture speculatively.

## Preconditions

1. Read `AGENTS.md`, `docs/supported-models.md`, `docs/designs/ffn-execution.md`,
   `docs/designs/qualification.md`, and `tests/README.md`.
2. Require the requested Model ID to exist in `[[models]]` in the effective
   CrossPool configuration.
3. Resolve its local path only with `XpoolConfig.model_path_of(model_id)`. Stop
   if the config, resolved directory, model config, or checkpoints are absent.
   Do not download weights, guess a path, or edit configuration to bypass this
   gate.

## Workflow

1. Inspect the local model config and checkpoints, the pinned SGLang model
   implementation, and the existing adapters under
   `xpool.integrations.sglang.models`.
2. Classify the model as one of:
   - supported by an existing adapter without source changes;
   - supported by a small model-specific adapter;
   - blocked by a missing Router, operator, quantization, expert-parallel, or
     FFN architecture capability.
3. Reuse an existing adapter whenever its architecture and checkpoint contract
   match. Add the smallest model-specific adapter only when behavior differs.
4. If support requires a new architecture capability, stop ordinary
   implementation and use `write-plan` to design that capability first.
5. Add the model-owned SGLang qualification at
   `tests/suites/models/<model-id>/test_sglang_model_qualification.py`. Keep
   model constants and model-specific cases in that suite.
6. Run the focused checks and then:

   ```bash
   uv run xtest run --integration=sglang --suite <model-id> --strict-requirements
   ```

7. Add the Model ID to `docs/supported-models.md` only after the strict model
   suite passes.

## Completion Report

Report these outcomes separately:

- adapter implementation or confirmed reuse;
- qualification result;
- supported-model entry added or withheld.
