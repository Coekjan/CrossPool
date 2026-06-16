---
name: git-commit
description: >
  Use as the final workflow step when preparing a commit and push. Stages
  intentionally, normally spawns six review subagents plus one self-evolve
  subagent, persists accepted lessons, commits with Codex trailers, and applies
  the repository push policy.
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
   document exists yet. After a canonical design document exists, code/test commits
   must map to one current design section. Pure mechanical formatting commits
   may skip design-section mapping only when they are standalone, contain no
   intended behavior change, and the commit message/final report label them
   format-only.
5. For non-format-only work after a canonical design document exists, confirm the
   relevant section reflects the staged completion; update and restage it before
   review if it is stale. For bootstrap/reset, docs-only, or pure mechanical
   formatting commits, verify that no design-status update is needed.
6. Spawn six independent `reviewer` subagents, one per axis below.
7. Spawn one `reviewer` self-evolve subagent in parallel with the review subagents.
8. Wait for all seven subagents, then synthesize their reports in the main session.
9. Close completed subagents after their reports are consumed unless an immediate
   same-task follow-up needs their existing context.
10. Fix blocking findings, update staging, and rerun affected checks.
11. Persist accepted self-evolve lessons to repo instructions and, when
    permitted, Codex memory according to the active memory policy.
12. Write a clear English commit message with required trailers and run
    `git commit -F <message-file>`.
13. After the commit succeeds, apply the push policy and push the current branch.

If the user explicitly disables review for one profile/debug/reference branch
commit, treat that as a one-off exception. Skip both the six-axis review and
self-evolve subagent fan-out only for that requested commit, still run staging
checks and required trailers, and explicitly restore the normal review workflow
for the next commit. Do not infer that review should be skipped for later commits
or for mainline.

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

## Six Review Subagents

Each review subagent uses the `reviewer` custom agent defined in
`.codex/agents/reviewer.toml`. It runs `gpt-5.5` with high reasoning and
read-only sandboxing. Each review subagent gets the staged diff and exactly one
review axis:

| Axis | Subagent focus |
|------|----------------|
| A - Code to Design | Code matches the accepted repository design API, scope, ownership boundaries, and one-step commit scope. For reset/bootstrap commits, verify the staged repository shape matches AGENTS.md. |
| B - Code to Docstrings | Docstrings match signatures, types, tensor shapes, returns, raises, preconditions, and postconditions. |
| C - Code to Comments | Inline comments still describe real concurrency, ordering, shape, and hardware behavior. |
| D - Stale References | Docs, tests, benchmarks, and readmes do not reference removed or renamed APIs, paths, commands, or phases. |
| E - Environment Hardcoding | No hardcoded `/home/`, `/data/`, hostnames, ports, model paths, or cluster assumptions bypass config. Local reference paths may appear only as clearly labeled non-runtime evidence. |
| F - Engineering Quality | Code follows KISS, DRY, cohesive ownership, appropriate OOP boundaries, clear naming, local style, and testable structure without over-abstraction. |

Subagent reports are candidate evidence. The main Codex session must verify concrete
findings before changing code or declaring the commit ready.

Every review subagent prompt must be bounded and read-only. It must forbid file
edits, staging, commits, pushes, amends, git config changes, memory writes,
network/browser/image tools, and spawning other agents.

For pure formatting commits, keep the same six-agent fan-out unless the one-off
profile/debug/reference exception above or the quota/unavailability policy
applies. In prompts, tell reviewers the commit is intended to be mechanical and
ask them to focus on format-only purity, generated or unrelated changes,
build/lint risk, and stale workflow references rather than feature semantics.

## Self-Evolve Subagent

The self-evolve subagent also uses `reviewer`. It reviews the staged diff, task
history, repo instructions, and Codex session excerpts for durable lessons and
instruction conflicts. Before spawning it, the main session should invoke the
repo-local `self-evolve` skill and supply or cite its session excerpts. Keep the
self-evolve procedure in that skill rather than duplicating it here.

The self-evolve subagent does not write memory or modify files on its own. The
main Codex session decides which lessons qualify. For each accepted lesson:

- If the lesson shows that user intent conflicts with `AGENTS.md`, the accepted
  repository design document, `.codex/skills/`, or `.codex/agents/`, update the
  conflicting repo instruction in the same self-evolve phase.
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

- `uv run ruff format --check ...`
- `uv run ruff check ...`
- `uv run ty check`

The default uv environment includes the SGLang-based runtime dependencies.
CUDA 13 is the project baseline, not an optional-extra dimension. The NVSHMEM
transport is implemented in C++/CUDA and should use the NVIDIA NVSHMEM runtime
package directly; do not add Python NVSHMEM bindings unless an accepted design
requires them. Sync the full developer environment with:

- `uv sync --group dev`

After the native scaffold exists, pre-commit should run `clang-format` checks
for C++/CUDA files and the CMake build/test commands defined by the accepted repository
design. Hooks must use pre-commit's file list instead of recursively scanning
the working tree. For docs-only or config-only changes, at minimum run
`git diff --cached --check` and targeted search checks for stale policy
references.
