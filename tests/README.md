# Test Architecture

`python -m tests` is the canonical composition root. It collects pytest cases,
derives their resource requirements, runs CTest and Python stages in order, and
keeps one GPU pool locked until every supervised process scope is reaped.
Direct pytest and CTest commands are focused debugging interfaces only.
The process tree, scheduler, GPU lease, endpoint, and artifact internals are
documented in [`tests/harness/README.md`](harness/README.md).

## Placement

- `tests/suites/unit/` mirrors Python modules and owns deterministic behavior.
  Unit tests do not execute native operations, initialize CUDA, launch
  subprocesses, or load weights. The mandatory session native/dispatcher
  preflight still runs.
- `tests/suites/integration/` owns cross-module, pinned-SGLang, native binding,
  component CUDA, daemon, CLI, and real process-management contracts.
- `tests/suites/e2e/` owns installed `xpool` and `sglang serve` workflows,
  model weights, HTTP inference, graph evidence, observer traces, token parity,
  multi-model concurrency, and shutdown.
- `tests/suites/cext/` owns C++/CUDA value, layout, protocol, scheduler,
  resident-kernel, trace, and utility behavior through CTest/GTest.
- `tests/harness/` contains reusable collection, scheduling, process, GPU,
  native, and SGLang infrastructure. Harness modules are not test suites and
  must not import collected test modules.

Place a test at the lowest layer that can observe its public behavior. Shared
setup belongs in a focused harness module or an explicitly imported fixture;
do not create implicit fixture dependencies through directory `conftest.py`
imports. E2E files use `test_e2e_*.py` names. Model IDs, topology matrices,
trace capacities, graph modes, and test-only KV limits belong in
`tests/harness/sglang/manifest.toml`, not Python test code.

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

## Execution Flow

The package runner performs these steps:

1. Collect selected Python suites in an isolated worker and compile a typed
   test plan from pytest metadata.
2. Acquire one pool from startup `CUDA_VISIBLE_DEVICES` only when selected
   cases require GPUs, then prove every visible device works through MPS.
3. Run CTest, Unit, Integration, and E2E in canonical order. GPU work is sorted
   by resource count and estimated duration and backfilled across idle GPUs.
4. Run each Python GPU task in a `SupervisedTaskScope`; release its lease only
   after the complete descendant process domain is reaped.
5. Parse JUnit and E2E artifacts, evaluate declared token-parity groups, and
   retain logs under `.xpool-cache/test-runs/`.

Set `XPOOL_TEST_KEEP_RUNS` to a positive integer to retain only that many
recognized runs, including interrupted runs. When unset, the runner does not
remove old results. Concurrent active runs and unrecognized directories are
never cleanup candidates.

Each E2E SGLang server writes a versionless `*.inference.json` beside its log.
It contains the exact public `/generate` request and response and is written
before HTTP-status and token-shape validation. JUnit describes case outcome,
`*.duration.json` records timing and resolved graph mode, token-parity artifacts
compare declared cross-mode groups, and observer files prove internal graph and
transport/fabric behavior.

## Commands

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi

# Complete resource-eligible repository suite.
uv run python -m tests

# One or more canonical stages.
uv run python -m tests --suite unit
uv run python -m tests --suite cext --suite integration
uv run python -m tests --suite e2e --strict-requirements

# Focused debugging without cross-task scheduling or parity aggregation.
uv run pytest tests/suites/unit/config/test_schema.py
uv run pytest tests/suites/integration/native/test_transport_lifecycle.py -s
```

Use `pytest --collect-only` to inspect concrete parameterized cases. Build and
install the native extension with the repository's canonical uv/scikit-build
command before running native or E2E tests.
