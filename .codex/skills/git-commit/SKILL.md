---
name: git-commit
description: >
  Use as the final workflow step when preparing a commit and push. Stages
  intentionally, invokes the repo-local self-evolve skill, persists accepted
  lessons, commits with Codex trailers, and applies the repository push policy.
---

# Git Commit

Use this skill only as the final workflow step, after documentation,
implementation, tests, and review-ready changes for the task are complete.

## Workflow

1. Inspect the working tree with `git status --short`.
2. Stage only the intended files.
3. Verify staging with `git status --short` and `git diff --cached --stat`.
4. Identify the single accepted repository design section, bootstrap/reset
   policy, or workflow item covered by staged changes. The initial reset may be
   scoped to repository cleanup and workflow metadata because no active design
   document exists yet. After `PLAN.md` exists, code/test commits must map to
   one current design section. Pure mechanical formatting commits may skip
   design-section mapping only when they are standalone, contain no intended
   behavior change, and the commit message/final report label them format-only.
5. For non-format-only work after `PLAN.md` exists, confirm the relevant section
   reflects the staged completion; update and restage it before review if it is
   stale. For bootstrap/reset, docs-only, or pure mechanical formatting commits,
   verify that no design-status update is needed.
6. Invoke the repo-local `self-evolve` skill and spawn its reviewer. Keep the
   session-excerpt and durable-lesson procedure in that skill.
7. Synthesize the self-evolve report in the main session.
8. Fix blocking findings, update staging, and rerun affected checks.
9. Persist accepted self-evolve lessons to repo instructions and, when
    permitted, Codex memory according to the active memory policy.
10. Write a clear English commit message with required trailers and run
    `git commit -F <message-file>`.
11. After the commit succeeds, apply the push policy and push the current branch.

If the user explicitly disables self-evolve review for one
profile/debug/reference branch commit, treat that as a one-off exception. Skip
the self-evolve reviewer only for that requested commit, still run staging
checks and required trailers, and explicitly restore the normal workflow for
the next commit. Do not infer that self-evolve should be skipped for later
commits or for mainline.

If reviewer subagents cannot be spawned because of quota, tool unavailability,
or agent infrastructure failure, follow the fallback policy in `self-evolve`.
Record the fallback in the final report and keep the same staging checks,
accepted-lesson persistence rules, commit trailers, and push policy.

While `git commit -F <message-file>` is running hooks, use the wait time only for
non-mutating exploration of the staged diff and future work. Inspect staged
changes, the accepted repository design document when present, adjacent risks, stale
follow-up notes, and likely next commit scope, then keep concise next-work notes
for use after the commit/push workflow finishes. Do not edit files, restage,
amend, change git config, run formatters, run tests, run benchmarks, run
profilers, or start GPU work while hooks are running. If a hook fails, revise or
discard those notes as needed and address the hook failure first.

Do not use `git commit --no-verify` unless the user explicitly requests it.
Do not change global or local git `user.name` / `user.email`. Do not repair
author attribution by changing git config.
Do not delete ignored or machine-local files as commit cleanup unless the user
explicitly asks; leave `.venv/`, `.vscode/`, `.ruff_cache/`, and similar local
state alone.

## Self-Evolve Subagent

The self-evolve subagent also uses `reviewer`. It reviews the staged diff, task
history, repo instructions, and Codex session excerpts for durable lessons and
instruction conflicts. Before spawning it, the main session should invoke the
repo-local `self-evolve` skill and supply or cite its session excerpts. Keep the
self-evolve procedure in that skill rather than duplicating it here.

The self-evolve subagent does not write memory or modify files on its own. It
should consume excerpts supplied by the main session; if it needs to run the
bundled self-evolve scanner directly, it must use the scanner's non-mutating
mode. The main Codex session decides which lessons qualify. For each accepted
lesson:

- If the lesson shows that user intent conflicts with `AGENTS.md`, the accepted
  repository design document, `.codex/skills/`, or `.codex/agents/`, update the
  conflicting repo instruction in the same self-evolve phase.
- If code-quality, configuration, or testing corrections recur at least twice in
  the same conversation or across recent excerpts, explicitly evaluate them as
  durable lessons instead of treating them as one-off patch feedback.
- Persist memory only according to the active Codex memory policy. Do not edit
  memory files directly when the active environment requires an ad-hoc memory
  note or explicit user permission.
- Add one `Self-Evolved: <lesson>` line to the commit message.

If no lesson is accepted and persisted, omit `Self-Evolved:` entirely.

## Commit Message

Use an English commit message:

```text
type(scope): imperative subject

Explain what changed and why. Include quantitative data for benchmark results.

Self-Evolved: <persisted lesson>

Co-authored-by: Codex <codex@openai.com>
```

Include one `Self-Evolved:` line per persisted lesson. Omit the line entirely
when no lesson is persisted. Always include the Codex co-author trailer.

Allowed types: `feat`, `fix`, `refactor`, `perf`, `test`, `chore`, `docs`.

## Push Policy

Every successful commit should be followed by a push of the current branch.
Inspect the author with `git show -s --format='%an <%ae>' HEAD`, then run
`git push` to the current upstream when a normal fast-forward push is possible.

If the branch has no upstream, stop and ask before publishing or setting one.
If the commit was amended/rebased or a normal push is rejected as
non-fast-forward, stop and ask before running `git push --force-with-lease`.

Never use bare `--force`. Never change git config to make the author canonical.

## Checks

Run the non-hook checks that match the staged diff's blast radius. Do not run
`uv run pre-commit run --all-files` as a routine pre-commit step; the installed
pre-commit hooks run automatically during `git commit -F <message-file>`.
Manual pre-commit runs are for explicit user requests, hook configuration
changes, or hook-failure troubleshooting.

Hooks are defined directly in `.pre-commit-config.yaml`; do not add a separate
pre-commit wrapper script. The installed pre-commit hooks run:

- `git diff --cached --check`
- `uv run ruff format --check ...`
- `uv run ruff check ...`
- `uv run ty check`
- `uv run pytest`
- `clang-format --dry-run --Werror ...` for C++/CUDA files
- `doxygen Doxyfile` for native documentation coverage

The default uv environment includes the SGLang-based runtime dependencies.
CUDA 13 is the project baseline, not an optional-extra dimension. The NVSHMEM
transport is implemented in C++/CUDA and should use the NVIDIA NVSHMEM runtime
package directly; do not add Python NVSHMEM bindings unless an accepted design
requires them. Sync the full developer environment with:

- `uv sync --group dev`

Native install/build verification belongs to the accepted design's explicit
check commands, such as the uv/scikit-build reinstall flow, not an ad hoc direct
CMake path. File-scoped hooks must use pre-commit's file list instead of
recursively scanning the working tree. Whole-project hooks such as type checks
or pytest may use `pass_filenames: false`. For docs-only or config-only
changes, at minimum run `git diff --cached --check` and targeted search checks
for stale policy references.
