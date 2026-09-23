# Test Architecture

`xtest run` is the canonical composition root. It collects pytest cases,
derives their resource requirements, runs CTest and Python stages in order, and
keeps one GPU pool locked until every supervised process scope is reaped.
Direct pytest and CTest commands are focused debugging interfaces only.
The process tree, scheduler, GPU lease, endpoint, and artifact internals are
documented in [`tests/harness/README.md`](harness/README.md).

Tests prove behavior visible at public boundaries. Avoid tests that mirror
registry internals, config table structure, or implementation text. Source-text
inspection is reserved for explicit quality gates such as documentation coverage
and allowlisted environment references. If a test still passes when its claimed
public behavior is broken, replace it with a behavior test.

## Placement

- `tests/suites/unit/` mirrors Python modules and owns deterministic behavior.
  Unit tests do not execute native operations, initialize CUDA, launch
  subprocesses, or load weights. The mandatory session native/dispatcher
  preflight still runs.
  Unit tests may construct immutable bound values when native operations are
  mocked and the subject is a pure Python projection; importing a bound value
  type alone does not require Integration placement.
- `tests/suites/integration/` owns cross-module, serving-engine, native binding,
  component CUDA, daemon, CLI, and real process-management contracts.
- `tests/suites/e2e/` owns routine installed `xpool` and `sglang serve`
  workflows, small-model weights, HTTP inference, observed graph structure,
  observer traces, multi-model concurrency, and shutdown.
- `tests/suites/cext/` owns C++/CUDA value, layout, protocol, scheduler,
  resident-kernel, trace, and utility behavior through CTest/GTest.
- `tests/suites/models/<model-id>/` owns optional real-checkpoint numerical and
  cross-graph qualification, including token or logit comparison. Engine-specific
  files use names such as `test_sglang_model_qualification.py`.
- `tests/harness/` contains reusable collection, scheduling, process, GPU,
  native, and SGLang infrastructure. Harness modules are not test suites and
  must not import collected test modules.

Place a test at the lowest layer that can observe its public behavior. Shared
setup belongs in a focused harness module or an explicitly imported fixture;
do not create implicit fixture dependencies through directory `conftest.py`
imports. Put engine-owned Integration and E2E files under an engine-named
directory; do not mix their cases with engine-neutral files. E2E files use
`test_e2e_*.py` names. Shared model IDs, topology matrices, graph modes, and
test-only KV limits belong in `tests/harness/sglang/manifest.toml`. Observer
record capacity belongs to the serving harness. Concrete-model numerical and
serving qualification cases belong as typed constants in their model suite
modules.

Keep common and subsystem-specific fixtures separate so each fixture owns one
coherent reset boundary. Use pinned SGLang concrete types, such as `ServerArgs`,
rather than handwritten substitutes when those types are available.

One layer should own each expensive behavioral verdict. Higher layers assert
only their integration seam instead of replaying lower-level protocol details.
Use a Cartesian matrix only when its dimensions interact; otherwise cover each
independent dimension once at its lowest observable boundary.

## Requirements

Pytest resource markers are executable metadata:

- `requires_cuda(min_devices=N)` requests `N` GPUs.
- `requires_config` requires `XPOOL_CONFIG`.
- `requires_mps` requires the externally managed CUDA MPS controller.
- `requires_model_weights(model_id)` resolves weights through `XPOOL_CONFIG`.

Unavailable resources skip by default and fail with
`--strict-requirements`; malformed explicit configuration always fails. Every
pytest session preflights `xpool.native` and the sole `xpool.ops.ffn_shim`
dispatcher registration. CTest CUDA cases declare one CTest GPU resource and
perform their own MPS preflight.

A missing or ABI-incompatible native extension fails the session, including
Unit-only sessions. Resource markers handle unavailable external resources.
E2E tests run without strict mode
when all declared and derived requirements are available. Each task materializes
a private config from `XPOOL_CONFIG` and its selected case models, without
waiting for unrelated configured models. All serving E2E cases use the shared
manifest's `serving_slo` rather than external scheduler or model SLO values.

Graph-mode acceptance criteria belong to
[Qualification](../docs/designs/qualification.md#numerical-and-graph-evidence).
Keep Eager, Decode Full, and combined Decode Full plus Prefill Breakable
evidence distinct.

## Execution Flow

The package runner performs these steps:

1. Collect selected Python suites in an isolated worker and compile a typed
   test plan from pytest metadata.
2. Acquire one pool from startup `CUDA_VISIBLE_DEVICES` only when selected
   cases require GPUs, then prove every visible device works through MPS.
3. Run CTest, Unit, Integration, E2E, and any explicitly selected Models in
   canonical order. GPU work is sorted
   by resource count and estimated duration and backfilled across idle GPUs.
4. Run each Python GPU task in a `SupervisedTaskScope`; release its lease only
   after the complete descendant process domain is reaped.
5. Parse JUnit and E2E artifacts, evaluate declared serving-graph groups, and
   retain logs under `.xpool-cache/test-runs/`.

Task `RUNNING` and terminal `PASSED` or `FAILED` lines identify leased physical
GPU indices and UUIDs. Terminal lines report task elapsed time and available
per-case JUnit durations on subsequent lines. CTest records each native case's
GPU assignment and duration separately.

`xtest clean` explicitly removes inactive test results. It keeps the
newest 20 inactive entries by default; use `--keep N`, `--all`, and
`--dry-run` to select or preview another cleanup. Concurrent active runs are
never removed.

Each E2E SGLang server writes a versionless `*.inference.json` beside its log.
It contains the exact public `/generate` request and response and is written
before HTTP-status and token-shape validation. JUnit describes case outcome,
`*.duration.json` records timing and resolved graph mode. Explicit Models
qualification compares Eager and Decode Full token output plus Eager and
combined-mode first-prefill logits. Model suites own numerical parity; routine
E2E owns installed serving, observed graph structure, and Transport/Fabric
behavior.

## Commands

Before canonical commands, conditionally set `UV_ENV_FILE` as below so uv loads
optional local configuration. Shell-exported values retain precedence; pytest
does not parse dotenv files.

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi

# Routine resource-eligible repository suite; model qualification is opt-in.
uv run xtest run

# One or more canonical stages.
uv run xtest run --suite unit
uv run xtest run --suite cext --suite integration
uv run xtest run --suite e2e --strict-requirements
uv run xtest run --suite integration --suite e2e --integration=sglang
uv run xtest run --suite Qwen/Qwen3-0.6B --integration=sglang
uv run xtest run --suite models --strict-requirements

# Explicit durable-result cleanup; default is --keep 20.
uv run xtest clean --dry-run
uv run xtest clean --keep 5
uv run xtest clean --all

# Focused debugging without cross-task scheduling or parity aggregation.
uv run pytest tests/suites/unit/config/test_schema.py
uv run pytest tests/suites/integration/native/test_transport_lifecycle.py -s
```

Use `pytest --collect-only` to inspect concrete parameterized cases. Build and
install the native extension with the repository's canonical uv/scikit-build
command before running native or E2E tests.

`--integration` filters engine-owned files in the selected Integration, E2E,
and Models suites before pytest imports them; engine-neutral tests remain.
Omitting it selects all integrations. Currently `sglang` is the only accepted
value. The file-level filter cannot be combined with explicit pytest selectors.

## Verification And Commit Hooks

Use affected focused checks during iteration. The installed hooks run normally
during commit; their complete-suite run is resource-eligible and is not proof
of strict final acceptance. Follow
[Qualification](../docs/designs/qualification.md#acceptance-and-invalidation)
for final evidence and reuse rules.

Keep hook definitions directly in `.pre-commit-config.yaml`. File-scoped hooks
check the paths supplied by pre-commit; whole-project checks may use
`pass_filenames: false` but must not recursively scan ignored environments such
as `.venv/`. Do not add a separate hook wrapper script.
