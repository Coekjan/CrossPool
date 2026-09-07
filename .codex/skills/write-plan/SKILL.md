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
2. Resolve repository facts and previously accepted decisions before asking
   questions. Ask about unresolved choices that materially affect behavior,
   interfaces, ownership, resource use, or acceptance. Do not reopen settled
   decisions without conflicting evidence.
3. Follow the user's requested discussion granularity and review boundary.
   Distinguish proposals from accepted decisions and wait for confirmation when
   the user requests review before implementation. Existing authorization does
   not require repeated approval; permission to write a plan alone does not
   authorize its implementation.
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

Describe changed interfaces and data structures precisely enough to implement
them. Include old and new symbols, signatures, field types, ordering, ownership,
lifecycle, failure behavior, and compatibility policy where their exact form
matters. Omit unchanged inventories and ceremonial `None` sections. Keep
implementation phases large enough to build and validate coherent dependency
layers; leave implemented interface details to authoritative source declarations.

The plan describes the target delta, not chat history or project management.
Exclude owners, status enums, progress logs, test run counts, host paths, and
completed-work narration. Keep ordinary decisions in the task README and create
supporting documents only when they are independently useful.

## Grilling And ADRs

The workflow above works without personal skills. When `grill-with-docs` is
available and invoked, incorporate its accepted decisions. If it recommends an
ADR for the current discussion, put the ADR under `docs/plans/<task>/adr/`.
Create that directory only when needed, not as empty scaffolding. Repository
document ownership takes precedence over a generic suggestion to use a root
`docs/adr/` directory. Keep resolved domain terms in `CONTEXT.md`.

At completion, `write-design` folds durable decisions, including ADRs, into their
owning current documents before removing the completed task directory. Git
retains the history; no separate archive is required.

The plan is complete when a new engineer can implement it from the repository,
all material decisions are settled, and no unresolved question blocks execution.
