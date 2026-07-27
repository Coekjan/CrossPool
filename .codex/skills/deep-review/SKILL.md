---
name: deep-review
description: Run xpool's six-axis staged-diff review with read-only reviewer subagents. Use when preparing a commit, when the user asks for delegated/deep/pre-commit review, or when another repo-local skill needs the standard xpool review fan-out.
---

# Deep Review

Use this skill to review staged xpool changes before commit. It owns the review
fan-out only. It does not stage files, fix findings, run self-evolve, persist
memory, commit, or push.

## Workflow

1. Inspect the review target with `git status --short`,
   `git diff --cached --stat`, and the intended staged diff. Review the staged
   diff only unless the user explicitly asks for staged, unstaged, and
   untracked files.
2. Identify the intended change, whether it is format-only, and the accepted
   `PLAN.md` section or workflow rule the staged diff claims to satisfy.
3. Spawn six independent `reviewer` subagents, one per axis below. Each prompt
   must include the staged diff scope, the intended change, the assigned axis,
   and the hard read-only constraints.
4. Wait for all six reports. Close completed subagents after their reports are
   consumed unless an immediate same-task follow-up needs their existing
   context.
5. Verify concrete findings in the main session before changing code or
   declaring the review clean. Subagent reports are candidate evidence, not
   final truth.
6. If a reviewer subagent cannot be spawned because of quota, tool
   unavailability, or agent infrastructure failure, do not silently skip the
   review. Continue in the main session by reviewing the staged diff against
   all six axes and record the fallback in the final report.

## Review Axes

| Axis | Focus |
|------|-------|
| A - Code to Design | Code matches the accepted `PLAN.md` API, scope, ownership boundaries, and one-step commit scope. For reset/bootstrap commits, verify the staged repository shape matches `AGENTS.md`. |
| B - Code to Docstrings | Docstrings match signatures, types, tensor shapes, returns, raises, preconditions, postconditions, and side effects. |
| C - Code to Comments | Inline comments still describe real concurrency, ordering, shape, hardware behavior, CUDA graph capture, ABI, plugin, native loader, and failure behavior. |
| D - Stale References | Docs, tests, benchmarks, and readmes do not reference removed or renamed APIs, paths, commands, or phases. |
| E - Environment Hardcoding | No hardcoded `/home/`, `/data/`, hostnames, ports, model paths, CUDA paths, local build directories, or cluster assumptions bypass config. Local reference paths may appear only as clearly labeled non-runtime evidence. |
| F - Engineering Quality | Code follows `docs/code-style.md`; repository architecture, configuration, testing, and workflow remain consistent with `AGENTS.md` and the accepted `PLAN.md`. |

Axis F reviewers must read `docs/code-style.md` and apply its complete current
rules. Reviewer prompts must reference that file rather than embedding a copied
checklist. Treat documented blocking requirements as blocking unless the
accepted design records a scoped exception.

## Subagent Constraints

Every reviewer prompt must be bounded and read-only. It must forbid file edits,
formatters, staging, commits, pushes, amends, git config changes, stash
mutation, memory writes, network/browser/image tools, destructive commands, and
spawning other agents.

For pure formatting commits, keep the same six-agent fan-out unless the user
explicitly disables review for that one commit or the fallback policy applies.
Tell reviewers the commit is intended to be mechanical and ask them to focus on
format-only purity, generated or unrelated changes, build/lint risk, and stale
workflow references rather than feature semantics.

## Output

Report blocking findings first, ordered by severity, with file and line
references where possible. Then report non-blocking risks and any axes that were
clean. If no issue is found, say that clearly and mention the review scope.
