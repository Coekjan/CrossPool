---
name: agy-delegate
description: Delegate an explicit request to Antigravity after confirming its prompt and model.
---

# Delegate to Antigravity

Use the installed `agy` CLI when the user explicitly invokes `$agy-delegate`.

1. Prepare the outgoing prompt from the rest of the current user request.
   Translate explicit `$skill-name` invocations to `/skill-name` in the same
   order. After any skill invocations, include the
   [one-turn instructions](#one-turn-instructions). Preserve the task text,
   literals, and requested scope. Check that requested skills resolve in
   `agy -p '/skills' --output-format json`.
2. Run `agy models` for available model names and IDs. Prefer the selection
   shown by `agy -p '/model' --output-format json` when it is available.
3. Show the checkout, exact outgoing prompt, suggested model, and available
   model IDs in a regular chat message. Use a short confirmation
   question with choices to run with the suggested model or cancel. Entering a
   listed ID as free text approves the displayed prompt with that model. Wait
   for approval. End the request on cancellation.
4. Run one `agy` process from the confirmed checkout with the approved prompt
   and model ID. Keep the prompt in one argument. Antigravity handles tool
   approvals.

   ```bash
   agy --model "$agy_model_id" --mode auto --output-format json -p "$agy_prompt"
   ```

5. Check the exit code, JSON `status`, `response`, any `denied_actions`, and
   stderr. An empty response, pending reviewer reports, or missing required
   evidence makes the result incomplete even when the status is `SUCCESS`.
   Report the result, inspection limits, and any mode diagnostics.
   If the selected model becomes unavailable, report the error and ask the
   user to select another model for the same prompt.

## One-turn instructions

```text
Complete this task in one headless turn. When using native subagents, start
the independent assignments before waiting. Ask each subagent to send its
completed report to this parent with send_message before ending its turn.
Use manage_subagents with the list action to check outstanding assignments.
While reports are pending, stay in this turn and use schedule for a one-shot
check. Receive and verify the reports before returning one combined answer
for the requested scope. Identify any missing reports and mark the result
incomplete if a required report cannot be collected.
```
