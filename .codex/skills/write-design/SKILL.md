---
name: write-design
description: Create or update xpool current-state architecture documents under docs/designs after an accepted implementation changes implemented behavior, invariants, ownership, lifecycle, or readiness contracts.
---

# Write Design

Keep `docs/designs/` aligned with implemented and accepted repository truth. A
target that is still being designed or implemented remains under `docs/plans/`.

## Workflow

1. Read `CONTEXT.md`, the relevant current design documents, the complete task
   directory when one exists, final source declarations, and acceptance tests
   or evidence.
2. Confirm the implementation and its required validation are complete. When
   they are not, revise the active plan instead of publishing target behavior
   as current design.
3. Update the document that owns each changed fact. Define a fact once; other
   design documents summarize the cross-module flow and link to its owner.
4. Preserve non-inferable invariants, ownership, lifecycle, failure behavior,
   and rationale. Leave exact signatures and field layouts to source
   declarations and generated stubs.
5. Update `CONTEXT.md` when the accepted implementation changes the domain
   language. Keep it a glossary without implementation details.
6. Update the design index and instruction pointers when document routing
   changes. Update the repository README when supported capabilities,
   configuration, or user-facing commands change.
7. Fold durable facts from the completed task, including any task-local `adr/`
   decisions, into their owning design, glossary, or repository guidance. For
   instruction-only tasks, update the owning instructions without inventing a
   runtime architecture change. Retain the task directory pending the user's
   cleanup decision, following the completion rule below.

## Completion And Cleanup

Implementation and validation completion do not authorize plan deletion. Fold
durable decisions into their current owners, report completion, and retain
`docs/plans/<task>/` until the user confirms removal. Recommend removal once
the task is complete. Honor an explicit removal or retention decision without
asking again; retaining a completed plan does not prevent a requested commit.
After removal is confirmed, remove only the completed task directory. Existing
Git history remains available; no separate archive is required.

Current design documents contain no task status, changelog, superseded design,
implementation diary, test run log, or speculative future architecture. Keep
unsupported boundaries only when they constrain the current system. Git and
the task review retain completed-plan history.

The design update is complete when the documents match live source and accepted
evidence, terminology has one owner, and cross-references resolve. A completed
plan awaiting cleanup confirmation is not a second current-design authority.
