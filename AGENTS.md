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

For implementation work, update or create the canonical design document before
editing source code when the design is missing, stale, or materially changed.
Revise canonical sections in place as decisions change; do not keep competing
old and new designs. The canonical xpool design document is `PLAN.md`; do not
refer to removed paths such as `docs/plan.md` as current design truth.

Implementation plans must be decision-complete. For every interface change,
list the exact old and new symbols, signatures, ownership and lifecycle
semantics, failure behavior, and compatibility policy. For every data-structure
change, list added, removed, renamed, nested, or reordered fields together with
their types and wire order. Do not use abstract promises such as "align APIs",
"audit naming", or "update tests" without identifying the affected boundaries
and expected behavior.

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
`debug.loopback.enable`, `debug.loopback.site`, and
`debug.graph_observer.outdir`.

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
`XPOOL_CONFIG` environment variable. `python -m tests` is the canonical
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

Shim graph-mode coverage must distinguish eager execution, decode full CUDA
graph replay, and prefill piecewise CUDA graph replay. SGLang integration
evidence should compare token ids across the relevant modes and use devkit
graph-observer evidence when it needs to prove SGLang entered graph paths.

## Native Extension And Shim

The native extension is the Linux build-time, SOABI-tagged `xpool.native`
module. `xpool.cext` owns one locked ABI preflight against
`xpool.native.abi_version()`. Native control and resource lifecycle callsites
use the typed `xpool.native.fabric` and `xpool.native.transport` bindings
directly. Only the compile-visible Tensor data path is a Torch dispatcher op;
runtime shim code calls `xpool.ops.ffn_shim`, which dispatches to
`torch.ops.xpool.ffn_shim` and registers its fake implementation directly.
Do not model control-plane lifecycle as Torch operators.

Bind native metadata and trace value types at the `xpool.native` root and keep
Fabric and Transport lifecycle functions in their corresponding native
submodules. Generate PEP 561 stubs from the built extension with the pinned
`pybind11-stubgen`; do not hand-maintain a duplicate `.pyi` API.

The daemon loads and initializes the native extension with
`RuntimeRole.DAEMON`, but it does not initialize CUDA or join an NVSHMEM PE.
Its only business-level native operation is daemon-only
`xpool.native.fabric.create_uid`;
transport, participant Fabric lifecycle, coordinator, and Instance operators
must reject the daemon role before touching device resources.

Build native code through uv/scikit-build with `CMAKE_BUILD_PARALLEL_LEVEL`;
do not make direct CMake or `setup.py build_ext` commands the normal developer
path. The NVSHMEM transport is C++/CUDA-owned and must not introduce Python
NVSHMEM bindings without an accepted design change.

The production `ffn_shim` path must support eager execution, decode full CUDA
graph replay, and prefill piecewise CUDA graph replay before serving readiness
is claimed. There is no separate loopback operator: `xpool.ops.ffn_shim` is the
sole Tensor dispatcher API, and `debug.loopback.enable` plus
`debug.loopback.site` select an internal Instance, AtnAgent, or FfnAgent debug
execution path behind it. Those sites exercise progressively more of the data
plane, but none is real FFN execution evidence.

## Build Style

Use `uv` for Python project management. `uv` owns the Python interpreter: keep
`.python-version` pinned to Python 3.12, keep `[tool.uv].python-preference =
"only-managed"`, and allow uv-managed Python downloads. Do not bind the project
to a system Python or hand-maintained virtual environment.

SGLang is the sole supported serving-engine dependency. Do not add vLLM
integration or dependencies unless an accepted repository design explicitly
changes that boundary.

CUDA Toolkit 13.2 and CCCL 3.2 are the native build baseline, not an
optional-dependency dimension. Keep
SGLang, Torch, FlashInfer, CUDA Python, NVSHMEM runtime libraries, and
control-plane libraries in the main project dependencies when xpool imports or
relies on them directly. The NVSHMEM transport is implemented in C++/CUDA and
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

- Manage native builds with `CMakeLists.txt`; do not reintroduce `setup.py` as
  the primary native build system. Build and reinstall the extension through
  the canonical uv/scikit-build command above instead of invoking CMake
  directly. Prefix it with `CMAKE_BUILD_PARALLEL_LEVEL=<jobs>` when explicit
  native build concurrency is useful.
- CMake enables ccache by default for C, C++, and CUDA when `ccache` is found
  and the corresponding compiler launcher is not already configured. Disable
  it explicitly with
  `--config-settings-package xpool:cmake.define.XPOOL_ENABLE_CCACHE=OFF`.
- Use `CMAKE_BUILD_PARALLEL_LEVEL=<jobs>` for native build concurrency; do not
  hard-code a repository-wide job count.
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
- After every successful commit, push the current branch according to the
  repository push policy. If the branch has no upstream or a normal push is
  rejected, stop and ask before publishing a new upstream or using
  `--force-with-lease`.
- Do not use `git commit --no-verify` unless explicitly requested.
- Do not change global or local git `user.name` or `user.email`.
- Do not apply or pop a stash unless explicitly requested.
- Do not delete ignored or machine-local files such as `.venv/`, `.vscode/`,
  `.ruff_cache/`, or `.codex/self-evolve-last.txt` unless the user explicitly
  asks. These files may contain useful local state even when they are not
  committed.

## Commit Messages

Use concise English commit messages and include:

```text
Co-authored-by: Codex <codex@openai.com>
```
