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
   - compatible with an existing adapter without source changes;
   - requiring a small model-specific adapter within accepted contracts;
   - requiring a missing Router, operator, quantization, expert-parallel, or
     FFN capability, or a new cache, wrapper, or lifecycle contract.
3. Reuse an existing adapter whenever its architecture and checkpoint contract
   match. Add the smallest model-specific adapter only when behavior differs.
4. If support requires a new architecture capability or cache, wrapper, or
   lifecycle contract, pause affected implementation and use `write-plan` to
   define the scoped extension. Obtain acceptance before implementing it.
5. Add the model-owned SGLang qualification at
   `tests/suites/models/<model-id>/test_sglang_model_qualification.py`. Keep
   model constants and model-specific cases in that suite.
6. Run the focused checks and then:

   ```bash
   uv run xtest run --integration=sglang --suite <model-id> --strict-requirements
   ```

   Complete the task's applicable acceptance and invalidated qualification as
   defined in `docs/designs/qualification.md`. When the task includes native
   multimodal input acceptance, decoder/text-only success completes a phase;
   finish the required native-input acceptance before claiming support.

7. After required qualification and task acceptance pass, update
   `docs/supported-models.md` by the complete Model ID. Update the existing
   row's Family from the verified full checkpoint architecture and set Support
   to `✅ Supported`; add a row only when that ID is absent. Preserve the
   Model ID / Family / Support columns and unique Model IDs. If acceptance
   fails or remains incomplete, retain the candidate's unsupported status and
   report the failed or missing evidence.

## Completion Report

Report these outcomes separately:

- adapter implementation or confirmed reuse;
- qualification result and remaining required evidence;
- support-table row updated or added, or update withheld with its reason.
