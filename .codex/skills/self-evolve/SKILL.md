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
2. Read recent Codex session excerpts. The first run needs an explicit cutoff:

   ```bash
   python3 scripts/summarize_sessions.py --since 2026-06-01
   ```

   After a successful full scan, the script records the scan start timestamp in
   the repo-level `.codex/self-evolve-last.txt`. Later runs should omit
   `--since`:

   ```bash
   python3 scripts/summarize_sessions.py
   ```

   The script first indexes session files from the `sessions/YYYY/MM/DD`
   directory layout, then reads matching JSONL files and filters individual
   records by their JSON `timestamp`. If the output limit is reached, the script
   exits without updating the last timestamp so unreviewed excerpts are not
   skipped.

3. Extract only lessons that are durable beyond the current patch:
   user preferences, workflow rules, tool availability, validation standards,
   recurring failure modes, and instruction conflicts.
4. Drop lessons already covered by `AGENTS.md`, the accepted repository design
   document, `.codex/skills/`, `.codex/agents/`, or existing memory.
5. Report candidate lessons and exact instruction conflicts. Do not write memory
   or edit files from a delegated reviewer/self-evolve subagent.

## Persistence Rules

The main session owns persistence. For each accepted lesson:

- Update the relevant repo instruction when user intent conflicts with the
  repository's current workflow.
- Persist memory according to the active git-commit workflow.
- Add `Self-Evolved: <lesson>` to the commit message only when the lesson was
  actually persisted.

If no lesson is accepted and persisted, omit `Self-Evolved:` entirely.

The bundled script is a reading aid only. It must not become an automatic memory
writer.
