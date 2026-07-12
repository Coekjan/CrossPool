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

## Design And Documentation

Keep design documents self-contained. A new engineer should be able to implement
from the repository without relying on prior chat context, a separate worktree,
or host-specific paths.

For implementation work, update or create the canonical design document before
editing source code when the design is missing, stale, or materially changed.
Revise canonical sections in place as decisions change; do not keep competing
old and new designs. The canonical xpool design document is `PLAN.md`; do not
refer to removed paths such as `docs/plan.md` as current design truth.

Documentation, comments, identifiers, tests, and commit messages must use
English. Public APIs must be well documented at the declaration site. For
Python, public modules, classes, functions, methods, dataclass fields, Pydantic
fields, and enum members need useful docstrings or field descriptions. Use
Google-style docstrings with `Args`, `Returns`, `Raises`, preconditions,
postconditions, and side effects where those sections apply. For C++ and CUDA,
public namespaces, classes, structs, enum values, constants, functions, and
fields need Doxygen-style `///` or `/** ... */` documentation. Public field
documentation must explain meaning, units, ownership, lifecycle, and why the
field exists.

Use inline comments only where they clarify ownership, device placement,
synchronization, CUDA graph capture, NVSHMEM ordering, IPC, or failure behavior
that is not obvious from the code.

## Engineering Quality

Keep implementation structure deliberate. Model-specific code belongs under the
model adapter that owns it; global integration layers should expose only generic
registration, discovery, binding, and shim contracts.

Use repository terminology consistently. `AtnAgent` is the current xpool
runtime entity; `FfnAgent` and lowercase `ffnagent` identify the reserved future
FFN runtime role and loopback site. Use the generic `Agent` term only for shared
lifecycle concepts. Use `PE` only when directly describing NVSHMEM APIs or
behavior.

Prefer behavior tests over source-string or implementation-text assertions.
Repository tests may inspect source text only for explicit quality gates such as
public documentation coverage or allowlisted environment-variable references.

Avoid boilerplate helper functions, especially private helpers used only once or
twice, when inlining keeps the calling code clearer. Add an abstraction only
when it carries a real ownership boundary, repeated behavior, or a typed
contract that improves local reasoning.

Do not add one-line wrappers, renamed constants, or pass-through functions that
only obscure the real API. Examples include `call_native_*` wrappers,
single-use `*_to_outdir` helpers, or constants that merely rename one local
literal. Inline the call unless the wrapper owns validation, synchronization,
lifetime, or a stable typed boundary.

Prefer explicit concrete types. Avoid broad `Any` or `object` unless they are
required for a dynamic third-party surface and the reason is documented close to
the annotation. In SGLang integration code, import pinned SGLang concrete types
directly instead of inventing local `*Like` protocols. Do not use
`TYPE_CHECKING` blocks or local imports to hide ordinary dependency cycles;
fix the ownership boundary instead.

Do not define xpool-owned Python variables, parameters, constants, functions,
methods, attributes, classes, or type aliases with a single leading underscore.
Use descriptive names, remove bindings that are not needed, and keep
framework-mandated unused parameters under their protocol names instead of
prefixing or deleting them. Python double-underscore protocols and unavoidable
private names owned by standard-library or third-party APIs are exempt. Do not
add compatibility aliases for renamed private implementation details.

For PEP 695 generic functions, prefer short local type parameter names such as
`R` and `W` when the scope is obvious. Avoid legacy-style verbose names such as
`ReturnT` or `WeightT` for local generic function parameters.

Prefer `match` statements when dispatching over a closed set of enum-like
states; avoid long `if`/`elif` ladders when a closed dispatch table or `match`
would make exhaustiveness clearer.

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

Put reusable test harnesses and process-management tools under
`tests/harness/`. Activate reusable fixtures explicitly in the test modules
that need them; do not use directory-level `conftest.py` imports to create
implicit cross-module fixture dependencies. Keep common and subsystem-specific
fixture setup separate so each fixture owns one coherent reset boundary. Keep
Python tests in three explicit layers: `tests/unit/`
mirrors xpool modules and must not exercise native behavior, launch subprocesses,
or load model weights; `tests/integration/` covers cross-module, pinned SGLang,
Python/native, and component-scoped CUDA subprocess contracts; `tests/e2e/`
owns full SGLang Engine, multi-process service, and model-weight workflows.
SGLang-facing tests should use SGLang's concrete
types such as `ServerArgs` rather than handwritten protocol or mock replacements
when those concrete types are available.

Every pytest session must preflight `xpool.ops`; a missing or ABI-incompatible
native extension is a suite failure, never a skip. E2E files use
`test_e2e_*.py` names. Resource requirements use
`requires_cuda`, `requires_config`, and `requires_model_weights`; unavailable
resources skip by default and fail under `--strict-requirements`, while invalid
explicit configuration always fails. E2E configuration comes only from the
`XPOOL_CONFIG` environment variable. Default collection includes unit,
integration, and E2E tests. E2E tests run without strict mode whenever every
declared and derived requirement is available. Before canonical pytest commands,
conditionally run `if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi` so
uv supplies optional local test configuration without making pytest parse
dotenv files.

Shim graph-mode coverage must distinguish eager execution, decode full CUDA
graph replay, and prefill piecewise CUDA graph replay. SGLang integration
evidence should compare token ids across the relevant modes and use devkit
graph-observer evidence when it needs to prove SGLang entered graph paths.

## Native Extension And Shim

The native extension is the Linux build-time `libxpool_cext.so` loaded through
`xpool.cext`. Native loading should use a single locked `NativeLibrary` path,
check only `torch.ops.xpool.abi_version()` against the Python ABI version, and
prewarm `xpool.ops` so Python custom-op wrappers are registered early. Runtime
code should call `xpool.ops.*` wrappers directly.

Build native code through uv/scikit-build with `CMAKE_BUILD_PARALLEL_LEVEL`;
do not make direct CMake or `setup.py build_ext` commands the normal developer
path. The NVSHMEM transport is C++/CUDA-owned and must not introduce Python
NVSHMEM bindings without an accepted design change.

The production `ffn_shim` path must support eager execution, decode full CUDA
graph replay, and prefill piecewise CUDA graph replay before serving readiness
is claimed. `ffn_shim_loopback` is only a development/debug substitute selected
by `debug.loopback.enable` with `debug.loopback.site=instance`; loopback
evidence is not real FFN or serving evidence.

## Build Style

Use `uv` for Python project management. `uv` owns the Python interpreter: keep
`.python-version` pinned to Python 3.12, keep `[tool.uv].python-preference =
"only-managed"`, and allow uv-managed Python downloads. Do not bind the project
to a system Python or hand-maintained virtual environment.

SGLang is the sole supported serving-engine dependency. Do not add vLLM
integration or dependencies unless an accepted repository design explicitly
changes that boundary.

CUDA 13 is the project baseline, not an optional-dependency dimension. Keep
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

Python code style:

- Run Python project tools through `uv run`; do not invoke `.venv/bin/...`
  commands directly.
- Format with `uv run ruff format`.
- Lint with `uv run ruff check`.
- Type-check with Astral ty using `uv run ty check`.
- Prefer typed, explicit APIs and avoid broad `Any` or unexplained ignores.
- Keep public Python documentation passing Ruff's public docstring checks.

C++ and CUDA style:

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
- Format C++, CUDA, and headers with `clang-format`.
- Keep public C++/CUDA documentation passing `doxygen Doxyfile`; Doxygen is a
  repository development prerequisite.
- Keep CUDA/NVSHMEM synchronization assumptions close to the code that depends
  on them.

## Pre-Commit

Keep pre-commit hooks active and installed. Routine commits should use normal
`git commit`; the installed hooks run automatically during commit. Do not run
`uv run pre-commit run --all-files` before every commit unless the user
explicitly asks, the hook configuration changed, or a hook failure needs
troubleshooting.

Hooks are defined directly in `.pre-commit-config.yaml`; do not add a separate
pre-commit wrapper script. File-scoped hooks must operate on files passed by
pre-commit. Whole-project hooks such as type checks or pytest may use
`pass_filenames: false`, but must not recursively scan ignored directories such
as `.venv/`. The pytest hook conditionally sets `UV_ENV_FILE` to the ignored
repository-root `.env` when that file exists; shell-exported variables retain
precedence over values loaded by uv.

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
