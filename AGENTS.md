# Codex Instructions

## Scope

This file owns repository development constraints and document routing.
Reusable workflows belong in `.codex/skills/<name>/SKILL.md`; transient task
status does not belong here. Keep skills self-contained, with skill-relative
resource paths and the structure defined by `skill-creator`.

## Code Style

Follow [docs/code-style.md](docs/code-style.md) for Python, C++, CUDA,
bindings, tests, scripts, and build definitions. It owns code-level conventions;
reference it rather than copying its rules into skills or reviewer prompts.

## Design And Documentation

Keep design documents self-contained so an engineer can work from the repository
without prior chat context or host-specific files.

- [docs/designs/README.md](docs/designs/README.md) maps implemented and accepted
  architecture.
- [docs/plans/README.md](docs/plans/README.md) maps candidate workstreams, not
  owners, schedules, progress, or accepted target designs.
- `docs/plans/<task>/README.md` owns an active task's scoped target delta.
- [CONTEXT.md](CONTEXT.md) owns domain terminology.

Source declarations and generated native stubs remain authoritative for exact
implemented interfaces. Read the current design and active plan relevant to the
task. Use `write-plan` for non-trivial changes needing a tracked target design;
use `write-design` after implementation and acceptance to fold durable decisions
into their owners and handle user-confirmed plan cleanup.

## Configuration

Resolve runtime configuration through `xpool.config`. Entry points install the
process-global config with `init_global_config()`; business logic reads it with
`get_global_config()` instead of caching policy in modules or adapters. Follow
[Control Plane](docs/designs/control-plane.md#configuration-and-integration)
for precedence, environment registration, and model-path ownership.

## Testing

Follow [tests/README.md](tests/README.md) for test placement, execution,
resource requirements, and harness conventions. Follow
[Qualification](docs/designs/qualification.md) for acceptance evidence and
invalidation. Use focused checks during iteration and complete the acceptance
required by the task.

## Build And Environment

Use `uv` with its managed interpreter and the repository's pinned toolchain.
Direct runtime dependencies belong in the main project dependencies. The
[README](README.md#quick-start) owns installation and MPS commands; project
configuration owns exact dependency versions.

The [supported deployment boundary](docs/designs/overview.md#supported-boundary)
requires an accepted design change before adding another serving engine or
platform. CUDA MPS is externally managed: the daemon observes readiness but does
not start the controller or change GPU compute mode. Stop CUDA clients before
stopping their MPS controller.

## Pre-Commit

Keep installed hooks active and use normal `git commit`. Let hooks run during
commit; reserve manual `uv run pre-commit run --all-files` for an explicit
request, hook changes, or hook-failure diagnosis. The hook definitions live
directly in [.pre-commit-config.yaml](.pre-commit-config.yaml).

## Workflow

- Keep changes scoped to the accepted design.
- Use `.codex/skills/git-commit/SKILL.md` for commit preparation.
- Use `.codex/skills/deep-review/SKILL.md` when the user requests delegated,
  deep, or pre-commit review. Commit preparation does not invoke it automatically.
- Use reviewer agents only when the user asks for delegation or an invoked
  skill requires it.
- Do not apply or pop a stash unless explicitly requested.
- Preserve unrelated work and ignored or machine-local files, including
  `.venv/`, `.vscode/`, caches, and `.codex/self-evolve-last.txt`. Delete them
  only when explicitly requested.
