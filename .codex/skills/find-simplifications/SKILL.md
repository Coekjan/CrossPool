---
name: find-simplifications
description: Find evidence-backed simplifications in CrossPool code, APIs, configuration, tests, and prose; write or consolidate proposals, identify small inline cleanups, or assess simplifications from another branch. Use for removing dead, duplicated, speculative, or unnecessarily maintained behavior and infrastructure.
---

# Finding CrossPool Simplifications

Find changes that remove maintained obligations: APIs, representations, lifecycle states, configuration paths, dependencies, tests, or documentation. Prefer a few well-supported candidates over a count of deletions. This is guidance, not a checklist; keep the user's scope and distinguish a survey from permission to implement its proposals.

## Establish scope and constraints

Read [AGENTS.md](../../../AGENTS.md), the relevant [current designs](../../../docs/designs/README.md), [code style](../../../docs/code-style.md), and [testing policy](../../../tests/README.md). Consult the applicable active plan under `docs/plans/` for intentional decisions. Requirements and plans introduced by the code under review are design choices to assess, not independent evidence that its distinctions are necessary.

For broad requests, inspect substantial production machinery as well as obvious unused symbols; do not stop after the first promising deletion. Delegation follows [AGENTS.md](../../../AGENTS.md); when authorized, give each agent a bounded domain and require consumer evidence and rejected alternatives.

## Search for removable obligations

Use these questions to guide discovery.

- **Does a declared feature have a complete effect path?** Trace producer, transformations, provider or endpoint support, and observable result. A field copied everywhere may still have no setter, no reader, or only implementations that reject it. For CLI and configuration pass-throughs, follow authoritative reads and precedence to find forwarded values that are shadowed before use. Search constructors and discriminant emitters as well as callers.
- **Which distinctions change a consumer's action?** First ask whether the distinction, field, or adapter is needed at all. Trace existing source fields and discriminants through their emitters and authorization or capability checks before adding a transport-specific marker; a broad envelope category need not have the source field's meaning. Consider deleting a distinction separately from relocating its declaration, and state which inputs or capabilities each alternative admits. Several internal states may require one model-facing status or one usable identifier. Preserve required authorization, residency, durability, and ownership behavior; remove public distinctions that only invite an unusable choice.
- **Could a smaller explicit behavior remove the subsystem?** Fixed intervals, caller-supplied time zones, or reported activation failure can eliminate generalized calendars, persisted defaults, or automatic rollback. Name the capability given up and the machinery deleted. A production caller makes this a behavior decision, not an automatic rejection; implementation must remain within the user's authorization.
- **Can a consumer read the authoritative value when needed?** Look for copied histories, promotion ledgers, invalidation events followed by reads, and feature caches beside a shared projection stream. Derivation or reuse may delete subscriptions, retained data, stale-response handling, and recovery paths together. Establish the required time of observation first.
- **Is composition being mistaken for policy?** Optional service presence, a provider method, or a sibling tool can express availability without deciding deployment behavior. Explicit configuration or a caller-intent operation may remove registries and dependency checks. Preserve distinct operations when their ownership or completion timing differs.
- **What owns the complete maintenance cost?** Trace duplicate application trees, demo launchers used by products, resolver-only workspaces, and installation helpers with their own upgrade lifecycle. For required behavior, compare existing producer/consumer extension points, with changes to distant core or shared type declarations under [the ownership rule](../../../docs/code-style.md#ownership-and-abstraction). Compare the entire removed system with the replacement, including residual glue; moving an unnecessary concept closer to its owner or adding a gate to keep duplicate definitions equal does not remove its maintenance cost.

## Prove reachability and the trade-off

Start with `rg`, then read the matches. Search exact symbols, property reads and writes, discriminants, event and wire strings, config keys, package names, and both `.method(` and `method(` forms. Tests and declarations can show a contract without proving a shipped producer or consumer.

Include Python sources, native declarations and bindings, generated stubs, CMake, package manifests and exports, CLI and configuration consumers, plugin and model discovery, harnesses, and supported deployments. Classify examples and fixtures by their actual entry point rather than directory name. Public third-party extension paths can be intentional despite zero repository call sites. A static unused-code checker must account for dynamic discovery and generated outputs.

For each candidate, record the current owner, effective producer/consumer path, what disappears, what remains, capability and compatibility effects, and the strongest reason to retain it. Distinguish:

- removal of unreachable or unread behavior;
- a narrower public or product behavior with an explicit loss;
- a protected obligation or insufficient evidence.

Reject a candidate when it breaks a retained obligation, merely relocates the same complexity, or has no meaningful reduction. Read rejection reasons at the level of the proposed change: a stale inventory or a mixed proposal's rejected behavior does not settle every independent item. Refresh the evidence before re-proposing one.

## Preserve ownership and failure semantics

For copies, freezes, validators, and callback captures, name the value's origin, next owner, and trust boundary. Typed same-process calls ordinarily borrow readonly values; parsers, config loaders, queues, model/tool JSON, durable files, workers, processes, and wire decoders own or validate their inputs. Hostile-getter or callback-replacement tests do not establish a requirement by themselves. A frozen root does not prove its descendants immutable, and equal ids do not prove object identity.

For asynchronous machinery, map each future, flag, cancellation path, disposer, and reservation to its owner and transition. Collapse mechanisms only when they express the same fact. Preserve synchronous publication and rollback, callback containment, first-terminal-outcome arbitration, worker/process ownership, and dispose-to-quiescence where required. Durable start/end markers can record real facts even when result summaries ignore them; dropping interrupted work is a capability loss, not redundant-log cleanup.

## Replace infrastructure only for a net reduction

Search existing dependencies and the standard library before adding a package. Follow the toolchain and dependency ownership in [AGENTS.md](../../../AGENTS.md). Identify the exact implementation and dedicated tests the replacement deletes, its maintenance and transitive cost, and every residual behavior the glue must preserve.

Exercise the mismatches most likely to defeat the replacement: protocol truncation, cancellation and teardown, configured bounds, deterministic test clocks, platform packaging, and source versus built entry points. A library that covers framing but requires rebuilding the same framing to enforce a size limit is not a simplification. An accepted edge-behavior change belongs in the proposal and relevant tests. Do not re-prove a dependency's generic specification while omitting tests for the harness-specific shim.

## Record, consolidate, and validate

Follow [code style](../../../docs/code-style.md) when prose is in scope. Remove code-restating or remotely owned explanations while preserving local behavior, timing, ownership, and failure obligations. Update affected READMEs, docstrings, configuration, consumers, and generated files with their owning change.

A survey reports proposals without editing. For each supported candidate, include concrete consumer evidence, the removed maintenance cost, the capability given up, and observable acceptance conditions. Distinguish supported candidates from meaningful rejections and insufficient evidence; do not describe an unverified search as exhaustive.

A substantial accepted change uses [write-plan](../write-plan/SKILL.md). Update an existing owner when the decision is the same; do not create duplicate plans to preserve candidate counts. Implementation follows the user's authorization and [implement-plan](../implement-plan/SKILL.md). Current-design updates and plan cleanup follow [write-design](../write-design/SKILL.md).

When folding another branch, compare its independent diff against the target base, port only supported non-overlapping proposals, and consolidate overlapping rationale. Closing another PR requires authorization or clear ownership of that housekeeping.

Select checks through [Test Architecture](../../../tests/README.md) and [Qualification](../../../docs/designs/qualification.md). Report the surveyed areas, supported candidates, meaningful rejections or deferrals, and commands actually run. A survey alone does not authorize fixes, commits, or publication.

Adapted from DeepSeek Harness `dsh-find-simplifications`; see [Third-Party Notices](../../../THIRD_PARTY_NOTICES.md#deepseek-harness).
