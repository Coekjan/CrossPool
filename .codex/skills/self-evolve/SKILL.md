---
name: self-evolve
description: Use when Codex needs to review xpool Codex session history for durable workflow lessons, user preferences, repository instruction conflicts, new tools, validation expectations, or recurring failure modes; normally invoked by the git-commit skill before committing.
---

# Self-Evolve

Use this skill to turn recent Codex session history into candidate durable
lessons. The output is advisory: the main session decides what to persist and
where. Bundled-resource paths in this skill are relative to this skill
directory.

## Workflow

1. Inspect the staged diff and current repo instructions that may be affected:
   `AGENTS.md`, `.codex/skills/`, and `.codex/agents/`.
2. Read recent Codex session excerpts. The main session may run the bundled
   script because it owns repo-local state updates. A read-only delegated
   reviewer should consume excerpts supplied by the main session, or run the
   script only with `--no-update-last`.

   To scan from an explicit point in time:

   ```bash
   python3 scripts/summarize_sessions.py --since 2026-06-01
   ```

   When `--since` is omitted, the script reads the repo-level
   `.codex/self-evolve-last.txt`. If that file is absent or empty, it scans all
   available sessions and reports that fallback on stderr. After a successful
   full scan, the script records the scan start timestamp. Later runs should
   normally omit `--since`:

   ```bash
   python3 scripts/summarize_sessions.py
   ```

   The script first indexes session files from the `sessions/YYYY/MM/DD`
   directory layout, then reads matching JSONL files and filters individual
   records by their JSON `timestamp`. If the output limit is reached, the script
   exits without updating the last timestamp so unreviewed excerpts are not
   skipped.

   Read-only review mode:

   ```bash
   python3 scripts/summarize_sessions.py --no-update-last
   ```

   Preference-focused review mode:

   ```bash
   python3 scripts/summarize_sessions.py --mode preferences --no-update-last
   ```

3. Treat the script as an index, not as the self-evolve decision. For long or
   emotionally corrective sessions, also review the current conversation and run
   a targeted read-only search over recent session files for repeated user
   corrections, preferences, and workflow complaints. Do not conclude "no
   lesson" merely because the marker-filtered script returned sparse excerpts or
   because a delegated reviewer did not see a direct instruction conflict.
4. Run an explicit preference extraction pass for user corrections when recent
   history includes code-quality, configuration, testing, review, or workflow
   complaints. Cluster repeated corrections by theme and compare them against
   `AGENTS.md`, `PLAN.md`, `.codex/skills/`, `.codex/agents/`, and existing
   memory before deciding whether they are new, covered, or conflicting.
5. Extract only lessons that are durable beyond the current patch:
   user preferences, workflow rules, tool availability, validation standards,
   recurring failure modes, and instruction conflicts.
6. Drop lessons already covered by `AGENTS.md`, `PLAN.md`, `.codex/skills/`,
   `.codex/agents/`, or existing memory.
7. Report candidate lessons and exact instruction conflicts. Do not write memory
   or edit files from a delegated reviewer/self-evolve subagent.

If the self-evolve reviewer subagent cannot be spawned because of quota, tool
unavailability, or agent infrastructure failure, do not skip the self-evolve
decision. The main session should run the scanner in non-mutating mode, inspect
the staged diff and recent task context itself, record the fallback in the final
report, and apply the same persistence rules below.

## Persistence Rules

The main session owns persistence. For each accepted lesson:

- Update the relevant repo instruction when user intent conflicts with the
  repository's current workflow.
- If a code-quality, configuration, or testing correction appears at least twice
  in the same conversation or across recent excerpts, explicitly decide whether
  it belongs in `AGENTS.md`, a repo-local skill, reviewer instructions, or an
  ad-hoc memory note. Do not leave repeated corrections as only transient chat
  context.
- Persist memory according to the active git-commit workflow.
- Add `Self-Evolved: <lesson>` to the commit message only when the lesson was
  actually persisted.

If no lesson is accepted and persisted, omit `Self-Evolved:` entirely.

The bundled script is a reading aid only. It must not become an automatic memory
writer.
