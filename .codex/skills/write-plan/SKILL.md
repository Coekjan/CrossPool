---
name: write-plan
description: Create or revise a decision-complete xpool change plan under docs/plans before a non-trivial implementation changes accepted architecture, interfaces, data structures, ownership, lifecycle, or validation contracts.
---

# Write Plan

Write one self-contained target delta for one active task. A simple local change
that neither needs design decisions nor coordinates multiple implementation
steps does not need a tracked plan.

## Workflow

1. Read `CONTEXT.md`, the relevant current documents under `docs/designs/`,
   adjacent source and tests, and an existing task plan when one exists.
2. Establish the task's baseline and every material decision. If the current
   request also invokes `grill-with-docs`, consume its accepted decisions. In
   other cases, build a dependency tree and ask one round containing every
   currently unblocked question with a recommended answer. Continue until the
   decision frontier is empty.
3. Respect the requested review boundary. Present the proposed plan in the
   conversation until the user authorizes repository writes.
4. Create or revise `docs/plans/<task>/README.md`. Use a descriptive kebab-case
   task name; add directly named supporting documents in that directory only
   when the main plan cannot carry an independently useful decision, research
   result, or prototype result clearly.
5. Record resolved domain terms in the root `CONTEXT.md`. Keep implementation
   details and decisions in the task directory.
6. Stop before source changes while any material question remains open.

## Plan Contract

Use these sections when they apply:

- Goal
- Baseline
- Accepted Changes
- Interface Changes
- Data-Structure Changes
- Implementation
- Validation
- Out of Scope
- Open Questions

Name exact old and new symbols, signatures, ownership and lifecycle semantics,
failure behavior, compatibility policy, and field types or wire order for every
affected interface or data structure. Write `None` when an explicitly reviewed
boundary is unchanged. Keep implementation phases large enough to build and
validate coherent dependency layers.

The plan describes the target delta, not chat history or project management.
Exclude owners, status enums, progress logs, test run counts, host paths, and
completed-work narration. Do not create fixed `adr`, `research`, `prototype`,
or archive subdirectories. A rare task-local decision document is justified
only by a hard-to-reverse, surprising trade-off.

The plan is complete when a new engineer can implement it from the repository,
all material decisions are settled, and the Open Questions frontier is empty.
