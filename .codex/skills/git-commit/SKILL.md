---
name: git-commit
description: Use as the final workflow step when preparing a commit and push. Stages intentionally, invokes the repo-local self-evolve skill, commits with Codex trailers, and applies the repository push policy.
---

# Git Commit

Use this skill only after documentation, implementation, validation, and
review-ready changes for the task are complete.

## Workflow

1. Inspect the working tree with `git status --short`.
2. Identify the relevant current documents under `docs/designs/` and any
   applicable task plan under `docs/plans/`. Confirm that the intended change
   matches the current baseline plus that plan's scoped target delta.
3. For an intentionally incomplete phase, keep the active plan and do not
   publish unimplemented target behavior as current design. For a completed
   implementation, confirm `write-design` has updated current design, folded
   durable task decisions into their owners, and resolved plan cleanup under
   its completion rule. A commit request is the usual checkpoint to ask whether
   to remove a completed plan, recommending removal if the user has not already
   decided; it is not implicit deletion permission. A local change that does
   not alter accepted design needs no design edit.
4. Stage only the intended files after the cleanup decision, then verify them
   with `git status --short` and `git diff --cached --stat`.
5. Use `self-evolve` to check the current task for new durable lessons. The main
   agent performs this check; delegation and history scanning are not required.
   Persist only accepted lessons through their owning document or an explicitly
   authorized memory mechanism.
6. Fix blocking findings, restage, and rerun checks affected by those fixes.
7. Write a concise English commit message with the required trailers and run
   `git commit -F <message-file>`.
8. After the commit succeeds, apply the push policy below.

Pure mechanical formatting commits may omit design mapping when they are
standalone, contain no intended behavior change, and are labeled format-only.
Bootstrap or docs-only changes must still leave instruction and document
routing internally consistent.

While commit hooks run, use the wait only for non-mutating inspection. Do not
edit files, restage, amend, change git configuration, run checks, or start GPU
work until the command completes. Address a failed hook before continuing.

## Commit Message

Use an English message:

```text
type(scope): imperative subject

Explain what changed and why. Include quantitative data for benchmark results.

Self-Evolved: <persisted lesson>

Co-authored-by: Codex <codex@openai.com>
```

Allowed types are `feat`, `fix`, `refactor`, `perf`, `test`, `chore`,
and `docs`. Include one `Self-Evolved:` line per lesson actually persisted;
omit it when no lesson was accepted. Always include the Codex co-author trailer.

## Push Policy

Push the current branch to its configured upstream after a successful commit.
If the branch has no upstream, stop and ask before publishing one. If a normal
push is rejected or rewritten history requires force, stop and ask before using
`--force-with-lease`. Never use bare `--force`, bypass hooks without explicit
authorization, or change git identity configuration.

## Checks

Run non-hook checks proportional to the staged diff. Let installed pre-commit
hooks run normally during `git commit`; a manual all-files run is reserved for
an explicit request, hook configuration changes, or hook-failure diagnosis.
Use the canonical commands and environment policy in `tests/README.md` and
inspect `.pre-commit-config.yaml` when exact hook composition matters.
