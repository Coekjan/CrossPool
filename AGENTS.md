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
English. Use docstrings and inline comments only where they clarify ownership,
device placement, synchronization, CUDA graph capture, NVSHMEM ordering, IPC,
or failure behavior that is not obvious from the code.

## Build Style

Use `uv` for Python project management. `uv` owns the Python interpreter: keep
`.python-version` pinned to Python 3.12, keep `[tool.uv].python-preference =
"only-managed"`, and allow uv-managed Python downloads. Do not bind the project
to a system Python or hand-maintained virtual environment.

The default environment is intentionally light enough for repository checks.
CUDA runtime dependencies such as SGLang, Torch, FlashInfer, and NVSHMEM belong
under the `runtime-cu13` optional extra and should be installed explicitly only
for runtime/device work:

```bash
uv sync --extra runtime-cu13
```

Python code style:

- Format with `uv run ruff format`.
- Lint with `uv run ruff check`.
- Type-check with Astral ty using `uv run ty check`.
- Prefer typed, explicit APIs and avoid broad `Any` or unexplained ignores.

C++ and CUDA style:

- Manage native builds with `CMakeLists.txt`; do not reintroduce `setup.py` as
  the primary native build system.
- Format C++, CUDA, and headers with `clang-format`.
- Keep CUDA/NVSHMEM synchronization assumptions close to the code that depends
  on them.

## Pre-Commit

Keep pre-commit hooks active. Run checks through pre-commit directly:

```bash
uv run pre-commit run --all-files
```

Hooks are defined directly in `.pre-commit-config.yaml`; do not add a separate
pre-commit wrapper script. Hooks must operate on files passed by pre-commit and
must not recursively scan ignored directories such as `.venv/`.

## Workflow

- Keep changes small and scoped to the accepted design.
- Use `.codex/skills/git-commit/SKILL.md` for commit preparation.
- Use repo-local reviewer agents only when the user explicitly asks for
  delegated review or when an invoked skill requires them.
- Do not push branches unless the user explicitly asks.
- Do not use `git commit --no-verify` unless explicitly requested.
- Do not change global or local git `user.name` or `user.email`.
- Do not apply or pop a stash unless explicitly requested.

## Commit Messages

Use concise English commit messages and include:

```text
Co-authored-by: Codex <codex@openai.com>
```
