# Codex Instructions

## Scope

This file defines repository-level development conventions only. Do not use it
to preserve transient branch status, current task boundaries, or the detailed
procedure of any repo-local skill.

Repo-local skills live under `.codex/skills/<name>/SKILL.md` and own reusable
task workflows. `AGENTS.md` may require that a skill be used, but must not
duplicate that skill's internal procedure. Keep repo-local skills self-contained:
use skill-relative paths for bundled resources, and put reusable scripts under
the relevant skill's `scripts/` directory. Repo-local skills must follow the
`skill-creator` structure: required `SKILL.md`, recommended
`agents/openai.yaml` UI metadata, and only task-relevant optional
`scripts/`, `references/`, or `assets/` resources.

## Code Style

All Python, C++, CUDA, binding, test, and build-definition code must follow
[`docs/code-style.md`](docs/code-style.md). That document is the canonical
source for code-level naming, typing, ownership, abstraction, documentation,
formatting, and native-language conventions. Do not duplicate its rules in
this file, repo-local skills, or reviewer instructions.

## Design And Documentation

Keep design documents self-contained. A new engineer should be able to implement
from the repository without relying on prior chat context, a separate worktree,
or host-specific paths.

[`docs/designs/README.md`](docs/designs/README.md) maps the current implemented
and accepted architecture. [`docs/plans/README.md`](docs/plans/README.md) maps
long-term candidate workstreams and their relationships; it is not an active
plan and does not record owners, status, or schedules. Active target changes
live under `docs/plans/<task>/README.md` until their implementation and
acceptance are complete. A relevant active plan is a scoped delta over the
current design; source declarations and generated native stubs remain
authoritative for exact interfaces. The root [`CONTEXT.md`](CONTEXT.md) owns
domain terminology.

Use the repo-local `write-plan` skill when a non-trivial change needs a tracked,
decision-complete target design. Use the repo-local `write-design` skill after
implementation and acceptance to update current design, fold durable task
decisions into their owning documents, and remove the completed task directory.
Read only the current design and active plan documents relevant to the task.

## Configuration

xpool runtime configuration must flow through `xpool.config`. Entry points
install one process-global config with `init_global_config()`, and business
logic reads it with `get_global_config()`. Do not copy or cache config values in
module globals, registries, plugins, or adapters.

Config sources resolve in this order: CLI arguments, allowlisted environment
variables, TOML config, then registry defaults. Required settings without a
default must fail fast. Most xpool settings are config-file only; `.env` is
primarily for SGLang/bootstrap settings such as `XPOOL_CONFIG` and
`SGLANG_PLUGINS`.

Model paths must be resolved with `XpoolConfig.model_path_of(model_id)`.
`ModelConfig.path` is the TOML schema field for explicit overrides, not a
runtime access pattern. Local development paths belong in ignored
`*.local.toml` files.

Every accepted `XPOOL_*` variable must be declared in the config registry. The
config layer should warn on unknown `XPOOL_*` variables instead of silently
turning them into policy. Debug settings use nested names such as
`debug.graph_observer.enable` and `debug.graph_observer.outdir`.

## Testing

Tests should prove behavior visible at public boundaries. Do not write tests
that mirror registry internals, config table structure, source strings, or
implementation text unless the test is an explicit quality gate for that text.
If a test would still pass when the user-visible behavior is broken, replace it
with a behavior test.

Repository tests may inspect source text only for explicit quality gates such as
public documentation coverage or allowlisted environment-variable references.

Put reusable test harnesses and process-management tools under
`tests/harness/`. Activate reusable fixtures explicitly in the test modules
that need them; do not use directory-level `conftest.py` imports to create
implicit cross-module fixture dependencies. Keep common and subsystem-specific
fixture setup separate so each fixture owns one coherent reset boundary. Keep
Python tests in three explicit layers: `tests/suites/unit/` mirrors xpool
modules and must not execute native behavior, initialize CUDA, launch
subprocesses, or load model weights; `tests/suites/integration/` covers
cross-module, pinned SGLang, Python/native, and component-scoped CUDA
subprocess contracts; `tests/suites/e2e/` owns full installed SGLang,
multi-process service, and model-weight workflows. Native C++/CUDA tests live
under `tests/suites/cext/`.
Unit tests may construct immutable values exposed by `xpool.native` when every
native operation is mocked and the behavior under test is a pure Python
projection; importing a bound value type alone does not move such a test into
Integration.
SGLang-facing tests should use SGLang's concrete
types such as `ServerArgs` rather than handwritten protocol or mock replacements
when those concrete types are available.

Every pytest session, including a Unit-only session, must preflight
`xpool.native` and the sole
`xpool.ops.ffn_shim` dispatcher registration; a missing or ABI-incompatible
native extension is a suite failure, never a skip. E2E files use
`test_e2e_*.py` names. Resource requirements use
`requires_cuda`, `requires_config`, `requires_mps`, and
`requires_model_weights`; unavailable resources skip by default and fail under
`--strict-requirements`, while invalid
explicit configuration always fails. E2E configuration comes only from the
`XPOOL_CONFIG` environment variable. `xtest run` is the canonical
complete-suite composition root and runs CTest, Unit, Integration, and E2E in
their accepted order. Direct pytest and CTest commands are focused debugging
interfaces. E2E tests run without strict mode whenever every declared and
derived requirement is available. Before canonical test commands,
conditionally run `if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi` so
uv supplies optional local test configuration without making pytest parse
dotenv files.

The SGLang E2E manifest owns test model IDs, placement, and graph modes.
`XPOOL_CONFIG` supplies the external model root and system settings, while each
test task materializes a private config containing only its selected manifest
models. Do not make an E2E case wait for unrelated models listed in the user's
runtime config.

Shim graph-mode coverage must distinguish Eager execution, Decode Full CUDA
Graph replay, and Prefill Breakable CUDA Graph replay. SGLang integration
evidence compares token IDs between Eager and Decode Full execution. Prefill
evidence compares Eager and Prefill Breakable first-prefill logits with forward
KL and uses Devkit Graph Observer evidence to prove Breakable execution
occurred; Prefill token IDs are diagnostic only.

## Build Style

Use `uv` for Python project management. `uv` owns the Python interpreter: keep
`.python-version` pinned to Python 3.12, keep `[tool.uv].python-preference =
"only-managed"`, and allow uv-managed Python downloads. Do not bind the project
to a system Python or hand-maintained virtual environment.

SGLang is the sole supported serving-engine dependency. Do not add vLLM
integration or dependencies unless an accepted repository design explicitly
changes that boundary.

CUDA Toolkit 13.2 and CCCL 3.2 are the native build baseline, not an
optional-dependency dimension. Keep SGLang, Torch, CUDA bindings, NVSHMEM
runtime libraries, and control-plane libraries in the main project dependencies
when xpool imports or relies on them directly. The NVSHMEM transport is implemented in C++/CUDA and
uses the NVIDIA NVSHMEM runtime package; do not add `nvshmem4py-cu13` unless a
new accepted design requires Python NVSHMEM bindings. Sync the full developer
environment with:

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
```

CUDA MPS is required for daemon readiness and transport execution. Configure
host-unique `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` values in
the ignored `.env`, create those directories, and start the controller before
xpool. The controller must see every GPU that PyTorch clients enumerate; use
GPU UUIDs to avoid ordinal remapping:

```bash
mkdir -p /tmp/xpool-mps-38373/{pipe,log}
export UV_ENV_FILE="$PWD/.env"
CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)" \
  uv run nvidia-cuda-mps-control -d
printf 'get_default_active_thread_percentage\n' | uv run nvidia-cuda-mps-control
```

MPS starts its server lazily when the first CUDA client connects. Stop SGLang
and agents cleanly before stopping the controller with
`printf 'quit\n' | uv run nvidia-cuda-mps-control`. The daemon observes MPS
readiness but does not own the controller lifecycle or change GPU compute mode.

- CMake enables ccache by default for C, C++, and CUDA when `ccache` is found
  and the corresponding compiler launcher is not already configured. Disable
  it explicitly with
  `--config-settings-package xpool:cmake.define.XPOOL_ENABLE_CCACHE=OFF`.
## Pre-Commit

Keep pre-commit hooks active and installed. Routine commits should use normal
`git commit`; the installed hooks run automatically during commit. Do not run
`uv run pre-commit run --all-files` before every commit unless the user
explicitly asks, the hook configuration changed, or a hook failure needs
troubleshooting.

Hooks are defined directly in `.pre-commit-config.yaml`; do not add a separate
pre-commit wrapper script. File-scoped hooks must operate on files passed by
pre-commit. Whole-project hooks such as type checks or the canonical test suite
may use `pass_filenames: false`, but must not recursively scan ignored
directories such as `.venv/`. The test-suite hook conditionally sets
`UV_ENV_FILE` to the ignored repository-root `.env` when that file exists;
shell-exported variables retain precedence over values loaded by uv.

## Workflow

- Keep changes small and scoped to the accepted design.
- Use `.codex/skills/git-commit/SKILL.md` for commit preparation.
- Use `.codex/skills/deep-review/SKILL.md` for the standard six-axis staged
  review only when the user explicitly requests delegated, deep, or pre-commit
  review. The git-commit skill does not invoke deep-review automatically.
- Use repo-local reviewer agents only when the user explicitly asks for
  delegated review or when an invoked skill requires them.
- Do not apply or pop a stash unless explicitly requested.
- Do not delete ignored or machine-local files such as `.venv/`, `.vscode/`,
  `.ruff_cache/`, or `.codex/self-evolve-last.txt` unless the user explicitly
  asks. These files may contain useful local state even when they are not
  committed.
