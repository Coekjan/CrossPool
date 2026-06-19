# Codex Instructions

## Scope

This file defines repository-level development conventions only. Do not use it
to preserve transient branch status, current task boundaries, or the detailed
procedure of any repo-local skill.

Repo-local skills live under `.codex/skills/<name>/SKILL.md` and own reusable
task workflows. `AGENTS.md` may require that a skill be used, but must not
duplicate that skill's internal procedure. Keep repo-local skills self-contained:
use skill-relative paths for bundled resources, and put reusable scripts under
the relevant skill's `scripts/` directory.

## Design And Documentation

Keep design documents self-contained. A new engineer should be able to implement
from the repository without relying on prior chat context, a separate worktree,
or host-specific paths.

For implementation work, update or create the canonical design document before
editing source code when the design is missing, stale, or materially changed.
Revise canonical sections in place as decisions change; do not keep competing
old and new designs.

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

Prefer behavior tests over source-string or implementation-text assertions.
Repository tests may inspect source text only for explicit quality gates such as
public documentation coverage or allowlisted environment-variable references.

Avoid boilerplate helper functions, especially private helpers used only once or
twice, when inlining keeps the calling code clearer. Add an abstraction only
when it carries a real ownership boundary, repeated behavior, or a typed
contract that improves local reasoning.

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
uv sync --group dev
```

Python code style:

- Format with `uv run ruff format`.
- Lint with `uv run ruff check`.
- Type-check with Astral ty using `uv run ty check`.
- Prefer typed, explicit APIs and avoid broad `Any` or unexplained ignores.
- Keep public Python documentation passing Ruff's public docstring checks.

C++ and CUDA style:

- Manage native builds with `CMakeLists.txt`; do not reintroduce `setup.py` as
  the primary native build system. Developers should build the extension through
  uv/scikit-build, for example
  `CMAKE_BUILD_PARALLEL_LEVEL=<jobs> uv sync --group dev --reinstall-package xpool`,
  instead of invoking CMake directly as the normal install path.
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
as `.venv/`.

## Workflow

- Keep changes small and scoped to the accepted design.
- Use `.codex/skills/git-commit/SKILL.md` for commit preparation.
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
