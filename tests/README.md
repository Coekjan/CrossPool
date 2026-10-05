# Test Architecture

`xtest run` is the canonical composition root. It collects pytest cases,
derives their resource requirements, runs CTest and Python stages in order, and
retains each tool-local device allocation until resource cleanup is verified and
every supervised process scope is reaped.
Direct pytest and CTest commands are focused debugging interfaces only.
The process tree, scheduler, device allocation, endpoint, and artifact internals are
documented in [Test and Benchmark Tooling](../docs/designs/tooling.md).

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
- `tests/suites/e2e/` owns routine installed `xpool`, including `xpool exec -- sglang serve`,
  workflows, small-model weights, HTTP inference, observed graph structure,
  observer traces, multi-model concurrency, and shutdown.
- `tests/suites/cext/` owns C++/CUDA value, layout, protocol, scheduler,
  resident-kernel, trace, and utility behavior through CTest/GTest.
  Host-only C++ cases use `.cpp` translation units; device cases use `.cu`
  and receive device resources through the existing CMake classification.
- `tests/suites/models/<model-id>/` owns optional real-checkpoint numerical and
  cross-graph qualification, including token or logit comparison. Engine-specific
  files use names such as `test_sglang_model_qualification.py`.
- `src/xpool-dev/xkit/` owns shared process, device, endpoint, run-store and serving
  lifecycle mechanisms. `src/xpool-dev/xtest/harness/` owns test collection,
  scheduling, verdicts, fixtures, native support and qualification;
  `src/xpool-dev/xbench/harness/` owns
  benchmark workloads, measurements and reports. Harness modules are installed
  tooling, not test suites, and must not import collected test modules.

Place a test at the lowest layer that can observe its public behavior. Shared
setup belongs in a focused harness module or an explicitly imported fixture;
do not create implicit fixture dependencies through directory `conftest.py`
imports. Put engine-owned Integration and E2E files under an engine-named
directory; do not mix their cases with engine-neutral files. E2E files use
`test_e2e_*.py` names. Named cases in `tests/tests.toml` select models, source
modules, graph modes and test-only KV limits, with English descriptions.
Catalogue cases reference portable scenes under `configs/deployments/` for
complete device placement, model TP/DP geometry, Executor Lane count and SLO;
owned benchmark cases reuse those scenes from `benches/benches.toml`.
The common runtime assembly and model-path contract belong to
[Shared deployment configuration](../docs/designs/tooling.md#shared-deployment-configuration).
Observer record capacity belongs to the serving harness. Concrete-model
numerical and serving qualification cases belong as typed constants in their
model suite modules.

Tool self-tests use `tests/suites/<layer>/{xkit,xtest,xbench}/`. Unit paths
mirror the installed modules, including each tool's `harness/`; Integration
paths identify the owning interface or workflow. Shared mechanisms are tested
once under `xkit`; tool tests prove CLI wiring, retained outcomes and
finalization. Real sockets, subprocesses, locks and cross-module execution
belong to Integration. Product tests retain their subsystem ownership even
when they use a tool fixture.

Each tool retains a minimal real CPU execution through the editable `xpool-dev`
development entry from outside the checkout. Real SIGTERM cancellation and
worker loss remain separate: cancellation has an observed outcome, while worker
loss lacks complete execution evidence. Fresh-process collection retains its
process boundary. Ordinary command wiring calls `xtest.cli.main` or
`xbench.cli.main` directly; collection and worker boundaries remain real.
Client execution, report projection
and RunStore lifecycle checks own their behavior without repeating a complete
list/run/report/clean cycle for each verdict. Private HTTP peers and case data
shared by tool tests belong in a non-collected test-local support module.

The owned benchmark regression lives under
`tests/suites/e2e/xbench/sglang/` and uses
the existing two-Qwen, two-device deployment with a short deterministic workload.
It proves the prepared workload, successful output from both models, actual
worker inventory within its assigned UUID lease, attention/FFN placement,
complete evidence and verified cleanup. Focused Integration tests own metadata
field projections and offline report rendering and export.
Its performance measurements are report-only. Source-owned benchmark programs
and prompt/trace inputs belong under `benches/suites/<family>/`; these are
measurement scenarios, distinct from pytest self-tests.

Keep common and subsystem-specific fixtures separate so each fixture owns one
coherent reset boundary. Use pinned SGLang concrete types, such as `ServerArgs`,
rather than handwritten substitutes when those types are available.

One layer should own each expensive behavioral verdict. Higher layers assert
only their integration seam instead of replaying lower-level protocol details.
Use a Cartesian matrix only when its dimensions interact; otherwise cover each
independent dimension once at its lowest observable boundary.

Report tests construct real Figures for layout and data checks and capture them
at the save boundary. One real multi-format export verifies file signatures and
embedded fonts. Rendering failure, active-lock protection and source-preserving
publication retain separate checks without re-exporting every format.

## Requirements

Repository-owned Python tests declare external resources on each function with
`xtest.requirements`. Static declarations describe the function's complete needs:

```python
import xtest


@xtest.requirements(device_count=2)
def test_component() -> None:
    ...
```

Use a callback returning `xkit.ResourceRequirements` when resources depend on
concrete parameter values. Callbacks receive all collected parameter values as
keyword arguments, including those supplied by native `pytest.mark.parametrize`;
fixtures remain ordinary pytest fixtures. Local checkpoint declarations use typed
`ModelId` values with `requires_config=True`. Derive case-dependent resources from
the case owner. The [owned benchmark E2E](suites/e2e/xbench/sglang/test_e2e_owned.py)
binds its selected case to the benchmark suite's existing requirements producer.

Tests without external-resource needs require no empty declaration. Module-level
`pytestmark` owns shared fixture, timeout and other native pytest policy;
each test function declares its own complete resources. Ordinary input matrices
use `pytest.mark.parametrize`. `xtest.parameterize` binds catalogue cases or
expands source-owned case/graph inputs; optional row callbacks return native
pytest parameters.
See the [serving test](suites/e2e/sglang/test_e2e_model_serving.py) and
[shared declaration contract](../docs/designs/tooling.md#catalogue-and-source-declarations).

The pytest adapter evaluates each concrete item's declaration once before marker
deselection and retains its typed resource value for preflight and test-plan
construction. It generates marker metadata for native `-m` selectors:

- `requires_device(min_devices=N)` selects tests requesting `N` devices.
- `requires_config` selects tests requiring `XPOOL_CONFIG`.
- `requires_model_weights(model_id)` selects tests requiring a local checkpoint.

Tests using `e2e_base_config` must declare `requires_config=True`. The fixture and
that item's checkpoint checks share the base resolved during resource setup.
[Shared deployment configuration](../docs/designs/tooling.md#shared-deployment-configuration)
owns its lifetime, environment inputs and assembly boundary.

Synthetic tests reuse the typed `TEST_MODEL_ID` from
`xtest.harness.support.config` unless a distinct identity is part of the behavior
under test.

Unavailable resources skip by default and fail with
`--strict-requirements`; malformed explicit configuration always fails. Every
pytest session preflights `xpool.native` and the sole `xpool.ops.ffn_shim`
dispatcher registration and loads the complete portable test catalogue once
before collection, including Unit-only sessions. Collection does not resolve
machine configuration, read checkpoints or probe devices. CTest device cases
declare a `devices` resource and execute directly. Role-aware deployment and
topology owners prepare MPS when their actual execution needs it.

A missing or ABI-incompatible native extension fails the session, including
Unit-only sessions. Resource preflight handles unavailable external resources.
E2E tests run without strict mode
when all declared and derived requirements are available. Each task materializes
a private config from `XPOOL_CONFIG` and its selected case models, without
waiting for unrelated configured models. Product serving E2E cases use the
selected deployment's SLO and model geometry rather than external scheduler or
model overrides. The owned benchmark regression follows the same configuration
assembly contract.

Graph-mode acceptance criteria belong to
[Qualification](../docs/designs/qualification.md#numerical-and-graph-evidence).
Keep Eager, Decode Full, and combined Decode Full plus Prefill Breakable
evidence distinct.

## Execution Flow

The package runner performs these steps:

1. Collect selected Python suites in an isolated worker and compile a typed
   test plan from pytest metadata.
2. Acquire one pool from startup `CUDA_VISIBLE_DEVICES` only when selected
   cases require devices. Normalize physical visibility without creating contexts.
3. Run CTest, Unit, Integration, E2E, and any explicitly selected Models in
   canonical order. Device work is sorted
   by resource count and estimated duration and backfilled across idle devices.
4. Run each Python device task in a `SupervisedTaskScope`; managed owners protect
   their clients from generic descendant signals. Return the allocation only
   after resource cleanup proof and complete descendant-domain reaping.
5. Parse JUnit and E2E artifacts, evaluate declared serving-graph groups, and
   retain logs under `.xpool-cache/test-runs/`.

Task `RUNNING` and terminal `PASSED` or `FAILED` lines identify leased physical
device indices and UUIDs. Terminal lines report task elapsed time and available
per-case JUnit durations on subsequent lines. CTest records each native case's
device assignment and duration separately.

`xtest clean` explicitly removes inactive test results. It keeps the
newest 20 inactive entries by default; use `--keep N`, `--all`, and
`--dry-run` to select or preview another cleanup. Concurrent active runs are
never removed.

Each E2E SGLang server writes `models/<model-id.uri_encode()>/inference.json` beside
its server log within the attempt directory.
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

# Inventory concrete cases without running fixtures or probing resources.
uv run xtest list --suite unit
uv run xtest list --suite integration --integration=sglang

# Offline reporting preserves the original test verdict and strictness.
uv run xtest report .xpool-cache/test-runs/RUN_ID --output /tmp/test-report

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
