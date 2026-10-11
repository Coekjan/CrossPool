---
name: implement-plan
description: Implement accepted xpool task plans when the user asks to execute them, including required validation and documentation alignment.
---

# Implement Plan

Complete the authorized target delta with reviewable changes and acceptance
evidence. Implementation authorization is separate from plan discussion.

## Workflow

1. Resolve the requested task plan or explicitly selected set of plans under
   `docs/plans/`. Clarify a missing or ambiguous target before source changes.
   The Roadmap identifies candidate workstreams, not executable specifications.
2. Read the complete task plan and its supporting documents, [AGENTS.md](../../../AGENTS.md),
   [CONTEXT.md](../../../CONTEXT.md), relevant [current designs](../../../docs/designs/README.md),
   and affected source, consumers and tests. Establish the current baseline and
   preserve unrelated work.
3. Implement the accepted change across its dependency boundaries. Make routine
   implementation choices within that scope directly. When evidence exposes a
   critical design defect or a material unresolved decision, pause affected
   changes and explain the conflict, its consequences and concrete alternatives.
   Use [write-plan](../write-plan/SKILL.md) to record the user's resolution;
   implementation continues under the resulting authorization.
4. Complete the plan's authorized validation under [Test Architecture](../../../tests/README.md)
   and [Qualification](../../../docs/designs/qualification.md). Fix failures
   within the accepted target and reuse valid evidence for the final version.
   Explain a blocker before changing scope or acceptance criteria. Review final
   code and documentation; delegated reviews follow the user's request and
   applicable workflow rather than being automatic.
5. After implementation and required validation are complete, use
   [write-design](../write-design/SKILL.md) to synchronize the owning current
   documents and apply its user-confirmed plan-cleanup rule. Instruction-only
   changes update their owning instructions.
6. Report the implemented changes, validation evidence and any remaining limits
   or blockers. Keep completed work distinct from unfinished or unverified work.

Finish with changes available for review. This skill never invokes the commit
workflow, commits or pushes.

Completion requires the accepted delta, affected consumers, required validation
and documentation alignment to be complete. A partial implementation or an
unproven acceptance result remains unfinished.
