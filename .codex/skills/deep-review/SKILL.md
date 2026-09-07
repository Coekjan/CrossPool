---
name: deep-review
description: Review xpool changes across six dimensions with read-only reviewers grouped by scope when delegated, deep, or pre-commit review is requested.
---

# Deep Review

Use this skill to review xpool changes, defaulting to the staged diff. It owns
the review fan-out only. It does not stage files, fix findings, run self-evolve,
persist memory, commit, or push.

## Workflow

1. Inspect `git status --short` and honor the user's review scope. Default to
   the staged diff when no scope is specified. A request to review the current
   working tree includes relevant staged, unstaged, and untracked changes.
   State the selected scope and inspect its diff and any new files.
2. Identify the intended change, whether it is format-only, the relevant
   current documents under `docs/designs/`, and any applicable target delta
   under `docs/plans/`.
3. Group applicable axes by the evidence they inspect. Combine overlapping
   concerns and separate independent, substantial concerns; honor explicit
   requests for six independent reviewers. Spawn `reviewer` subagents for those
   assignments. Each prompt includes the scope, intended change, assigned axes,
   and read-only constraints. State which axes each assignment covers.
4. Wait for the assigned reports and consume their findings.
5. Verify concrete findings in the main session before changing code or
   declaring the review clean. Subagent reports are candidate evidence, not
   final truth.
6. If a reviewer subagent cannot be spawned because of quota, tool
   unavailability, or agent infrastructure failure, do not silently skip the
   review. Continue in the main session against all six axes for the selected
   scope and record the fallback in the final report.

## Review Axes

| Axis | Focus |
|------|-------|
| A - Code to Design | Code matches the relevant current design plus any applicable active plan's scoped target delta. Unrelated design boundaries remain unchanged. For reset/bootstrap changes, verify the reviewed repository shape matches `AGENTS.md`. |
| B - Code to Docstrings | Docstrings match signatures, types, tensor shapes, returns, raises, preconditions, postconditions, and side effects. |
| C - Code to Comments | Inline comments still describe real concurrency, ordering, shape, hardware behavior, CUDA graph capture, ABI, plugin, native loader, and failure behavior. |
| D - Stale References | Docs, tests, benchmarks, and readmes do not reference removed or renamed APIs, paths, commands, or phases. |
| E - Environment Hardcoding | No hardcoded `/home/`, `/data/`, hostnames, ports, model paths, CUDA paths, local build directories, or cluster assumptions bypass config. Local reference paths may appear only as clearly labeled non-runtime evidence. |
| F - Engineering Quality | Code follows `docs/code-style.md`; terminology, current architecture, active target changes, configuration, testing, and workflow remain consistent with their owning repository documents. |

Axis F reviewers must read `docs/code-style.md` and apply its complete current
rules. Read `CONTEXT.md`, current design documents, and active plans only when
they are relevant to the selected scope. Reviewer prompts must reference the
owning files rather than embedding copied checklists. Treat documented blocking
requirements as blocking unless an applicable plan records a scoped exception.

## Subagent Constraints

Every reviewer prompt must be bounded and read-only. It must forbid file edits,
formatters, staging, commits, pushes, amends, git config changes, stash
mutation, memory writes, network/browser/image tools, destructive commands, and
spawning other agents.

For formatting-only changes, review semantic preservation, unintended scope,
and build/formatting risk together. Six separate reviewers are not required
unless explicitly requested. Tell reviewers the change is intended to be
mechanical.

## Output

Report blocking findings first, ordered by severity, with file and line
references where possible. Then report non-blocking risks and any axes that were
clean. If no issue is found, say that clearly and mention the review scope.
