---
name: add-benchmark-case
description: Add or refine concrete serving benchmark comparisons for supported models using the catalogue and existing portable deployments.
---

# Add Benchmark Case

Define a comparison whose fixed conditions, changed dimension and resource needs
are explicit. A new serving architecture belongs to a separately accepted plan;
model qualification belongs to `add-model-support`.

## Workflow

1. Read [the tooling tutorial](../../../docs/tutorials/tooling.md) and inspect
   `uv run xbench list`. Consult [benchmark design](../../../docs/designs/benchmark.md),
   [shared tooling](../../../docs/designs/tooling.md),
   [supported models](../../../docs/supported-models.md) and
   [qualification](../../../docs/designs/qualification.md) for the chosen scope.
2. Confirm the requested models are supported, their local checkpoints resolve
   through runtime configuration, and the proposed layout fits the available
   device budget. Establish the baseline and keep other conditions fixed.
3. Clone a suitable case with
   `uv run xbench case-gen --type serving --from CASE_PREFIX`. Edit the printed
   line range, including an English description. The new UUID is permanent;
   presentation changes retain it, while a distinct fixed experiment gets a new
   case. Preserve catalogue-relative inputs and deployment basename references.
4. Reuse a matching portable deployment or add a complete one under the existing
   Model ID directory convention. Match its descriptive basename to explicit
   TP/DP and Lane values; the name does not supply configuration. For sampled
   lengths, inspect both legal intervals and the output/input ratio together
   with the per-model offered rates. Validate the edited declarations with
   `uv run xbench list`; inventory does not establish deployment feasibility.
5. Execute only the experiments authorized for the task, using `run --case`.
   Inspect retained outcomes and generate reports by the listed artifact ID.
   A request to add declarations alone does not authorize expensive runs, weight
   downloads, runtime changes, commits or publication.

## Completion

Report source changes, execution/cleanup evidence and observed performance
separately. State comparisons not executed or invalidated; measurements are
report-only and do not establish a performance pass/fail threshold.
