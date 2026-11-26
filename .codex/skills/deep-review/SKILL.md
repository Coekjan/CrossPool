---
name: deep-review
description: Review xpool changes across six dimensions with read-only reviewers grouped by scope when delegated, deep, or pre-commit review is requested.
---

# Deep Review

Use this skill to review xpool changes, defaulting to the staged diff. It owns
the review fan-out only. It does not stage files, fix findings, run self-evolve,
persist memory, commit, or push.

## Workflow

1. Inspect the [Git state](#git-inspection) and honor the user's review scope.
   Default to the staged diff when no scope is specified. Working-tree review
   includes relevant staged, unstaged, and untracked changes.
   State the scope and read the matching source. For staged review, use
   `git diff --cached` and `git show :<path>` for file contents. For a commit
   comparison, inspect the selected revisions. For working-tree review, read
   the on-disk files and inspect the in-scope untracked files.
2. Identify the intended change, whether it is format-only, the relevant
   current documents under `docs/designs/`, and any applicable target delta
   under `docs/plans/`. Follow changed interfaces through their actual producers
   and consumers, including dynamic discovery and generated interfaces where
   relevant. Tests and declarations can show a contract without proving a
   shipped producer or consumer.
3. Group applicable axes by the evidence they inspect. Combine overlapping
   concerns and separate independent, substantial concerns; honor explicit
   requests for six independent reviewers. Assign each group as a read-only
   task to a native subagent when the session can wait for and collect its report.
   Prefer a configured reviewer profile when available. Include the scope,
   intended change, assigned axes, and read-only constraints in each assignment.
   State which axes each assignment covers.
4. Wait for the assigned reports and consume their findings.
5. Verify concrete findings in the main session before changing code or
   declaring the review clean. Subagent reports are candidate evidence, not
   final truth.
6. If a reviewer fails to start or the session cannot wait for native reviewer
   reports, apply all six axes in the main session and record the fallback.
   Return the completed review with the evidence for the selected scope.

## Git inspection

Place pathspecs after `--` in `status` and `diff`. For a commit comparison, use
the selected base and target revisions in place of `HEAD`. For `show`, pass the
full file selector before `--`. Use `:path` for the index or `<revision>:path`
for a committed tree.

```bash
git --no-optional-locks --no-pager status --short --
git --no-optional-locks --no-pager diff --no-ext-diff --no-textconv --cached --
git --no-optional-locks --no-pager diff --no-ext-diff --no-textconv HEAD --
git --no-optional-locks --no-pager show --no-ext-diff --no-textconv :path --
```

## Review Axes

| Axis | Focus |
|------|-------|
| A - Code to Design | Code matches the relevant current design plus any applicable active plan's scoped target delta. Unrelated design boundaries remain unchanged. For reset/bootstrap changes, verify the reviewed repository shape matches `AGENTS.md`. |
| B - Code to Docstrings | Docstrings match signatures, types, tensor shapes, returns, raises, preconditions, postconditions, and side effects. |
| C - Code to Comments | Inline comments still describe real concurrency, ordering, shape, hardware behavior, CUDA graph capture, ABI, plugin, native loader, and failure behavior. |
| D - Stale References | Docs, tests, benchmarks, and readmes do not reference removed or renamed APIs, paths, commands, or phases. |
| E - Environment Hardcoding | No hardcoded `/home/`, `/data/`, hostnames, ports, model paths, CUDA paths, local build directories, or cluster assumptions bypass config. Local reference paths may appear only as clearly labeled non-runtime evidence. |
| F - Engineering Quality | Code follows `docs/code-style.md`; terminology, current architecture, active target changes, configuration, testing, and workflow remain consistent with their owning repository documents. |

Axis F reviewers must read [code style](../../../docs/code-style.md) and apply
its complete current rules. Read [CONTEXT.md](../../../CONTEXT.md),
[current designs](../../../docs/designs/README.md), and active plans only when
they are relevant to the selected scope. Reviewer prompts must reference the
owning files rather than embedding copied checklists. Treat documented blocking
requirements as blocking unless an applicable plan records a scoped exception.

For test and fixture changes, use [Reliability And Isolation](../../../tests/README.md#reliability-and-isolation)
to assess the concurrency and cleanup actually exercised. For failure and
lifecycle changes, apply the failure-ownership rules in
[code style](../../../docs/code-style.md#ownership-and-abstraction) to the affected
state and its consumers. Substantiate a finding with a reachable execution path
and consequence.

## Subagent Constraints

Keep reviewer assignments bounded and read-only. Reviewers preserve files,
the Git index and history, configuration, stashes, and agent memory. The main
session assigns reviewers and verifies their findings.

For formatting-only changes, review semantic preservation, unintended scope,
and build/formatting risk together. Six separate reviewers are not required
unless explicitly requested. Tell reviewers the change is intended to be
mechanical.

## Output

Report blocking findings first, ordered by severity, with file and line
references where possible. Then report non-blocking risks and any axes that were
clean. If no issue is found, say that clearly and mention the review scope.
