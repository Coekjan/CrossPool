---
name: self-evolve
description: Check an xpool task for durable workflow lessons before committing, consulting session history when needed or explicitly requested.
---

# Self-Evolve

The main agent checks the current task for useful durable lessons. No reviewer
subagent or history scan is required for an ordinary commit. A finding is a
candidate rule change, not permission to persist it.

## Workflow

1. Review the current task's user corrections and execution problems.
2. Check the owning code-style, workflow, design, glossary, or skill document.
   If it already covers the lesson, add no duplicate rule. Determine whether the
   issue was noncompliance or unclear wording.
3. Propose only lessons useful beyond this patch. Repeated corrections are
   evidence to inspect, not an automatic trigger for another prohibition.
4. Record accepted changes in their owning documents within the user's
   authorization. Use the current task context when sufficient; consult
   relevant history only to resolve a missing fact or when historical review
   is requested.

## Historical Lookup

Paths below are relative to this skill directory. To inspect a specific range:

```bash
uv run python scripts/summarize_sessions.py --since 2026-06-01 --no-update-last
```

Without `--since`, the scanner uses `.codex/self-evolve-last.txt`. If neither
an explicit range nor a nonempty marker is available, it exits nonzero and asks
for `--since`; it does not default to all history. The range filters record
timestamps, not session-directory dates, so resumed sessions remain eligible.
`--mode preferences` limits excerpts to user corrections. The scanner is a
reading aid, not proof that all relevant lessons have been found.

A completed scan updates the marker to its start time unless
`--no-update-last` is supplied. Reaching the excerpt limit exits without
updating it. Read-only reviewers, when explicitly assigned, must use
`--no-update-last` and report findings without modifying files.

## Persistence

Use the existing owner: code conventions in `docs/code-style.md`, repository
routing in `AGENTS.md`, vocabulary in `CONTEXT.md`, current architecture in
`docs/designs/`, target changes in `docs/plans/<task>/`, and reusable
procedures in their skill. Improve an existing rule before adding a new one.

Memory writes require explicit user authorization and must use the active
memory mechanism. Neither this skill nor commit preparation grants it.
The scanner never writes memory.

Include a `Self-Evolved:` trailer only for a lesson actually accepted and
persisted. Otherwise omit the trailer; no extra report or document is required.
