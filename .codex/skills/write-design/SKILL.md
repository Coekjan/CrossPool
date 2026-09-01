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
6. Update `docs/designs/README.md`, the repository README, or instruction
   pointers only when the document set or its routing changes.
7. Fold every durable fact from a completed task directory into its owning
   design or glossary, then remove the complete `docs/plans/<task>/` directory.

Current design documents contain no task status, changelog, superseded design,
implementation diary, test run log, or speculative future architecture. Keep
unsupported boundaries only when they constrain the current system. Git and
the task review retain completed-plan history.

The design update is complete when the documents match live source and accepted
evidence, terminology has one owner, cross-references resolve, and no completed
task document remains as a second source of truth.
