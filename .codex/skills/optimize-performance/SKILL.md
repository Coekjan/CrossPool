---
name: optimize-performance
description: 'Use when investigating or optimizing CrossPool performance, designing realistic benchmarks, profiling runtime or tooling paths, or turning performance evidence into measured behavior-preserving fixes.'
---

# Speed Up CrossPool

Turn a broad “make it faster” request into reproducible user-path measurements and small, evidence-backed fixes. This is guidance, not a quota or a script: survey broadly, follow measured cost, and reject attractive changes that do not improve the workload users actually run.

## Establish scope and current authority

Read [AGENTS.md](../../../AGENTS.md), the relevant [current designs](../../../docs/designs/README.md), [testing policy](../../../tests/README.md), [code style](../../../docs/code-style.md), and the applicable active plan.

Agree on the user-visible endpoint, workload range, resource constraints, acceptable minor behavior differences, and stopping rule. Keep component and serving end-to-end measurements separate: a fast Router or allocator does not prove fast request completion. State excluded costs rather than calling a partial measurement complete product latency. A diagnosis-only request remains read-only; an optimization request authorizes changes within its stated scope. Material design decisions follow [write-plan](../write-plan/SKILL.md).

Inspect the exact current base, not just the running checkout. Study final merged diffs, owning source, tests, and resolved review threads; a PR body can describe an abandoned implementation. Separate merged, closed-unmerged, superseded, estimated, and newly measured evidence.

## Survey user paths, then rank candidates

Delegate independent domains only when authorized and useful; require measurements and production call sites, not guesses. Useful domains include:

- Cold startup, model preparation, Graph capture, and first request.
- Prefill and Decode, routing and Expert execution, Graph replay, and Transport/Fabric handoff.
- Multi-model queueing, offered load, request lengths, and TP/DP and Executor Lane geometry.
- Elastic Context State allocation, reclamation, cancellation, and teardown.

Vary independent cost drivers: bytes, tokens, requests, models, ranks, and retained context state are different quantities. Include typical and tail workloads, but avoid a combinatorial matrix with no decision value.

Rank candidates by observed user latency, CPU/allocations, retained memory, occurrence, and confidence. For each, name the production consumer, the repeated work, the expected complexity, the smallest falsifiable intervention, and the behavior that must remain stable. A suspicious loop, unused cache, or large file alone is not evidence of a bottleneck.

## Build realistic synthetic benchmarks first

Reuse [xbench measurements](../../../docs/designs/benchmark.md) and the existing focused tooling. Catalogue and deployment authoring belong to [add-benchmark-case](../add-benchmark-case/SKILL.md). Resource admission, cleanup, and evidence invalidation remain under [shared tooling](../../../docs/designs/tooling.md) and [Qualification](../../../docs/designs/qualification.md). Run only experiments authorized for the task.

If the user authorizes local corpus inspection, extract only aggregate workload characteristics. Never copy prompts, outputs, paths, identities, IDs, credentials, recordings, or recognizable snippets into fixtures, logs, screenshots, PRs, or artifacts. Generate fixed inputs from reviewed constants; resolve machine resources through the repository's configuration rather than embedding private data or host paths.

Before implementation, record a measurement card:

| Field | Required decision |
|---|---|
| User operation | Exact action and externally observable completion condition |
| Workload | Fixed dimensions, distributions, construction seed/constants, and why they exercise ordinary and tail use |
| Entry path | Production calls and installed artifacts; explicitly identified substituted boundaries |
| Clock | Included setup, cold/warm state, timing start/end, and excluded costs |
| Memory | Reachable endpoint objects, baseline, allocator policy, retained versus transient usage |
| Verdict | Raw samples, chosen aggregate, comparison conditions, and uncertainty |
| Behavior | Owning functional tests/snapshots and permitted minor differences |

Measure the installed production path with the repository's pinned toolchain through `uv run`. Do not add a production export solely for measurement or copy the algorithm into a “benchmark implementation.”

Use fresh children and private temporary roots for cold/process-memory samples. Warm samples explicitly retain the intended cache; never let fixture setup secretly warm a cold scenario. Keep the same input, validations, completion condition, and reachable output on both sides. A parse-and-discard baseline is not comparable with validated retained output.

Report all samples and the aggregate that decides the result. Serialize measured work against other owned CPU- or device-heavy jobs; measure reference and candidate under comparable conditions. Do not select a lucky run to hide a regression. Performance measurements remain report-only under Qualification; this skill adds no CI timing threshold.

Measure end-to-end latency independently from component phases. Track retained memory with intended objects still reachable, and transient pressure separately through an appropriate peak measurement. Faster execution with unbounded retention is not an automatic win.

## Prove the regression, then remove work

Run the unoptimized workload before changing production code. Save the command, revision, runtime/platform, fixture dimensions, raw measurements, and verdict. Reduce a failing scenario until it still exercises the real bottleneck, then rank falsifiable hypotheses before patching. Use profiles, allocation samples, work counts, or phase timings to distinguish them.

Common patterns worth testing, not automatic prescriptions:

- Keep compact representations compact through downstream readers; avoid intermediate objects when the consumer needs one aggregate.
- Remove duplicate parsing, copying, freezing, and validation only after identifying the actual ownership and trust transition. Typed same-process borrowing is not permission to weaken durable or wire parsing.
- Stream artifact transformations and bound intermediate state. Include publication and verification obligations where the user operation requires them.
- Separate read-only preparation from write/publication work without moving awaited work past a correctness-required endpoint.
- Prefer a suitable data structure to repeated shifting, scanning, or rebuilding. Measure the whole consumer path, not just the isolated container operation.
- Use revision-keyed reuse or singleflight only with explicit invalidation, bounded retention, independent waiter cancellation, and disposal ownership. Avoid caching expanded representations merely to make repeated benchmarks look fast.

Change one causal factor at a time. Re-run both the focused scenario and its end-to-end parent. When claiming a regression fix, use a negative control where practical: the original implementation or a controlled reintroduction of the targeted cost should reproduce the regression. Account for the measurement noise before claiming an improvement.

## Preserve behavior and resource ownership

Performance measurements complement functional evidence; they do not replace it. Run or add the narrow owning tests for output, ordering, errors, cancellation, concurrency, and disposal as applicable. Preserve accepted numerical references and tolerances, atomic publication, required validation, resource ownership, and readiness. Do not silently change lifecycle semantics to reach a number.

Keep functional expectations independent of benchmark internals; benchmark assertions need enough evidence to reach the real endpoint, not a second semantic test suite.

Reject an optimization when gains disappear end-to-end, a typical workload regresses materially, complexity outweighs a small gain, or cancellation/retention/durability cannot be explained and tested. Record the rejected hypothesis briefly instead of expanding scope to justify it.

## Deliver a bounded, reviewable result

Keep the reusable procedure here and scenario-specific truth with its benchmark or owning module. Use [write-plan](../write-plan/SKILL.md) for accepted changes to design and [write-design](../write-design/SKILL.md) for current documentation after implementation and validation. Keep the user's scope rather than turning performance work into a corpus-wide cleanup.

Apply [Test Architecture](../../../tests/README.md) and [Qualification](../../../docs/designs/qualification.md), report only executed evidence, and distinguish local observations from CI evidence. Performance measurements do not replace required functional or resource acceptance.

Summarize each result as: workload → before/after absolute values and ratio → endpoint and memory semantics → behavior evidence → negative control when available → exact checks → exclusions. Separate author-reported historical numbers, fresh local measurements, and CI evidence. Stop at the agreed scenario/fix scope; retain a short ranked follow-up list instead of chasing unrelated opportunities.

Adapted from DeepSeek Harness `dsh-speed-up-perf`; see [Third-Party Notices](../../../THIRD_PARTY_NOTICES.md#deepseek-harness).
