---
name: explain-plan
description: Explain an xpool task plan against current code, covering directory structure, system architecture, data structures, APIs and core code changes.
---

# Explain Plan

Give the user a concrete, evidence-backed view of the proposed change. The
default workflow is a read-only conversation; explanation does not authorize
implementation or document edits.

## Workflow

1. Resolve the requested task plan and read it completely, including relevant
   supporting documents. Clarify an ambiguous target. Read [AGENTS.md](../../../AGENTS.md),
   [CONTEXT.md](../../../CONTEXT.md), related [current designs](../../../docs/designs/README.md),
   source declarations, consumers and tests to establish the implemented baseline.
2. Explain the problem and the accepted target through the five views below.
   Keep the presentation within the user's requested scope and level of detail.
3. Distinguish implemented behavior, accepted target changes, assumptions and
   unresolved choices. Label proposed code and diagrams as illustrative, and
   reference the actual baseline declarations.
4. Connect each material change to the problem it solves. Explain rationale,
   trade-offs and relevant simplification opportunities using repository evidence
   or authoritative external sources. State uncertainty explicitly; a newly
   suggested alternative remains a proposal for the user to decide.

## Five Views

- Directory structure: show the relevant target tree and identify added, moved
  and removed files or modules.
- System architecture: explain component responsibilities, dependencies, data
  flow and relevant ownership or lifecycle changes.
- Data structures: compare changed types, fields, invariants and ownership.
- APIs: compare signatures, entry points, calling conventions and affected
  consumers.
- Core code: provide representative before-and-after examples or focused diffs
  showing how the central mechanism changes. Declarations alone do not explain
  the implementation.

Each view covers the affected area. Briefly identify an unchanged surface when
it is relevant instead of inventing a change. Use diagrams only where they
clarify relationships or timing; leave unrelated inventories out.

The explanation is complete when the user can connect the actual problem,
baseline, target architecture and core code changes, with material assumptions
and unresolved decisions visible. Repository files remain unchanged unless the
user separately requests an edit.
