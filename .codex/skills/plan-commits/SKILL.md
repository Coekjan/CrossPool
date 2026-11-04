---
name: plan-commits
description: Plan small, independently reviewable commits when a working tree or existing branch mixes changes. Use for commit breakdowns and branch splits before staging.
---

# Plan Commits

Leave the working tree, index and local branches as found. Deliver the series
in the conversation; use an ignored review directory for patches or a file map
when they make a large split easier to inspect.

1. **Establish the baseline.** Identify the target branch and record its
   commit. Fetch if the user wants the latest remote base. Inspect staged and
   unstaged changes and read relevant untracked files. When a PR already has
   commits, compare its merge base to its tip as well as its tip to the
   working tree. A file added on the PR and since deleted may need no commit.
2. **Assign changes by purpose.** Follow each behavior through callers, tests
   and owning docs. For multiple branches, assign shared prerequisites to
   the base and feature-specific changes to the dependent branch, following
   the user's routing. Mark shared files by symbol or hunk. Account for every
   in-scope change, including deletions, and identify what stays out.
3. **Order by dependency.** Give each commit one review question and leave
   it buildable on its parent. Separate mechanical moves from behavior when
   that makes the latter easier to read. Keep a changed contract with its
   required consumers and a focused check. If splitting an ABI, lifetime or
   accounting change would break an intermediate revision or require a
   throwaway adapter, keep it together and explain why.
4. **Verify a tricky split.** In a disposable copy at the recorded base,
   replay the proposed patches in dependency order. For a stacked PR, inspect
   the base result on its own and the PR-only diff above it. Compare the
   reconstructed result with the intended files, contents, modes and deletions.
   Report what was reproduced. Matching the final tree does not prove that
   each intermediate commit builds or passes its tests.

## Required output

Present the proposal in this order:

1. **Files and changes.** Name the comparison base. Show which files changed,
   what behavior changed and why. For a large diff, group by branch and
   subsystem, give file counts and added/deleted lines, and link a complete
   file-to-commit map. Identify shared files by hunk or symbol.
2. **Commit series.** In dependency order, give each commit an ID, branch,
   subject, the change a reviewer should look for, and a focused check.
   Explain why a large commit needs to stay together. Quote per-commit sizes
   only when measured from that commit's patch.
3. **Patch previews.** Use the same IDs. For each commit, link its unified
   diff against its intended parent or show it inline if short. If a mixed
   hunk has not yet been separated, show a real excerpt, label it partial,
   and say what remains. For a branch split, also link each branch's aggregate
   patch. State which patches were replayed and which revisions were tested.

Keep unresolved review findings separate from changes already in the series.
Stop at the proposal when the user asks to confirm it. For authorized
execution, hand off to [git-commit](../git-commit/SKILL.md) with the agreed
branch order, authorship and any limits on rebasing or pushing.
