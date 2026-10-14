# Reference Fixtures v1

These immutable inputs and expected outputs implement the
[shared contract](../shared-contract.md) and [calibration algorithm](../calibration.md).
They are synthetic contract evidence, not captured serving or clock measurements.
Fixture readers use exact integers and reduced rational values. Each scenario
starts from a fresh Run state; identical IDs across scenarios never join them.

## Files

| File | Contract surface |
| --- | --- |
| [layout.json](layout.json) | Complete 256-byte field table, presence groups, event IDs, and FFN enums |
| [metadata.json](metadata.json) | Immutable metadata revision used by valid raw encoding; SHA-256 is over its exact bytes |
| [encoding.json](encoding.json) | Full one-event chunk hex and malformed variants with exact structured errors |
| [causality.json](causality.json) | Complete/partial events, explicit associations, Replay/Eager, identity isolation, Context, tokens, quota, and tail cases |
| [snapshots.json](snapshots.json) | Complete Transport/Fabric JSON projections, exact phase pairs, and retention/missing-fact cases |
| [quality.json](quality.json) | Full quality checkpoint and conservation, export failure, unknown tail, and counter saturation cases |
| [routing.json](routing.json) | Publication/attachment outcomes and independent event/tensor availability |
| [routing.safetensors](routing.safetensors) | Literal published attachment, live_rows=2, effective_topk=2 |
| [calibration.json](calibration.json) | Exact blocks, feasible vertices, mappings, held-out failures, drift, gaps, Epochs, and durations |
| [checksums.json](checksums.json) | Byte lengths and SHA-256 for the data files; excludes this README and itself |

No generator or runtime parser is installed by C0. Downstream implementations
consume these files directly; they must not regenerate their own expected
outputs using the implementation under test. An intentional contract change
updates version, bytes, semantic expectations, and checksums together.

## Case Interpretation

JSON top-level contract_version is 1. Each `cases` entry has a unique `id`,
`input`, and `expected`. Expected dictionaries are assertions on the named
properties, not a license to discard other quality evidence. The complete
`normal_dense` case and snapshot projections specify full ordinary behavior.
Small counter/Context/quota cases deliberately describe that owning interface
rather than manufacturing a complete protocol trace for it.

Derived causal inputs use base_case=normal_dense. Copy that input, remove events
by `[producer,event_sequence]`, replace listed events by the same identity, then
append the listed events in order. Replace other explicitly supplied input
properties; unspecified properties are inherited. Remove/replace must match
exactly one base event. Appends may intentionally duplicate an identity to test
conflicts. This fixed expansion convention avoids repeating the complete trace
in every case and is separate from the production wire format.

For event cases, `events` entries contain Producer plus a full canonical wire
record: every absent field is explicitly zero. Use layout presence bits to
distinguish known zero from absent. Producer/Run come from the surrounding
context; `metadata` references a fixture-relative file. `metadata_overrides`
defines a fresh scenario's metadata revision: endpoints_add appends Endpoint
definitions, and model_topologies replaces the Generation's topology. IDs in
these lists are composite keys, not tensor-slot indices or array positions.
producers_add and clock_domains_add append registered definitions, and
generations_replace replaces the matching Generation by ID. Scenario capability
overrides replace the named capabilities for that scenario.
Independent scenario IDs are not serialized into the production format.

`normal_dense` retains one Instance operation, its AtnAgent observation,
Fabric entry, three Fabric role observations, one SGLang Replay, and one native
FFN activity. The Host return at 350 is distinct from the FFN compute completion
at raw Device 240; clocks in this example have no implicit cross-domain mapping.
Same-domain intervals in snapshots remain calculable without calibration.

Snapshot input_case references that case. Fabric has two local operations in
Producer 2 (AtnAgent local trace 1 and Coordinator local trace 2), and one in
Producer 5. These local trace IDs are not the common Invocation sequence.
Transport snapshots classify completed using the legacy site rule. Missing
events/facts are asserted in separate cases; do not apply the complete-vector
len(records) equation to partial evidence. The golden phase pairs match the
existing writers, including AtnAgent total beginning at SubmissionPrepared.

For Routing, capabilities.routing=enabled changes the base Dense fixture to a
synthetic MoE publication scenario. Published Routing data is:

```text
topk_ids:     [[0, 1], [1, 0]]       int32
topk_weights: [[0.75, 0.25], [0.5, 0.5]]  float32
```

Its final attachment index uses the fixture-relative routing.safetensors path.
The event's artifact_id is zero because runtime file publication occurs later;
the exact attachment identity resolves the final index. Not-collected, pool
capacity, payload oversize, I/O failure, missing file, and unknown crashed tail
are different expected outcomes. Safetensors metadata records preserves its
baseline Invocation fields and adds the association fields defined by v1.

## Numerical Oracles

Calibration input declares its attempted range, provenance, raw Epoch, load
phase, samples, and query Device times. Positions 4 and 9 are held out. Each
valid expected model includes reduced rational vertices (a,c), a and b,
support, fitting/validation sequences, and exact/rounded mapping results.
Unavailable model entries assert their reason; their public numerical values
are null. Sample failure retains null G/H_after, never a fabricated Device read.

known_offset has exactly a=1, b=100. known_rate has a=2, b=50.
asymmetric_windows has a=1, b=105, with feasible slope range [8/9,10/9]; its
estimate is conditional, not independent truth. ties_to_even checks half-integer
estimates at G=3 and G=5. held_out_failure changes a validation sample only,
so it must not silently change the fitted mapping to pass. model_conflict
changes a fitting window and must not be repaired by deleting that sample.

Series cases define consecutive blocks or distinct Epochs. The drift case has
rates 1 and 101/100, separated support, and resolved slope difference 10000 ppm.
between_blocks remains unavailable. Duration fixtures explicitly preserve
negative bounds and protocol edges when temporal ordering is indeterminate.
unresolved_drift keeps a rate difference interval containing zero. Duplicate,
conflicting, and invalid-bracket samples test the input boundary before fitting.
wire_sample_cases connect the fixed runtime Event payload to the public Sample
shape, including timeout with unknown Device time and retained resource ownership.

## Required Downstream Checks

- Encode every field of the valid vector and compare all 384 bytes. Decode it
  independently and compare the complete expected record, Producer, and CRCs.
  Validate metadata hashing before payload semantic decoding.
- Reject every malformed vector at the specified error stage. Recomputed CRCs
  on unknown kinds/flags/presence ensure semantic rejection is actually tested.
- Reconstruct both Native bounded views and offline legacy projections from the
  same events; compare exact shapes, timestamps, nullability, phase durations,
  counts, and source identity. Assert no guessed edge in missing_association.
- Cover out-of-order commits, completed-copy-only slot reuse, stale generation,
  Context exhaustion and retirement, operation prefix retention, and identity
  exhaustion without modifying production outcomes.
- Read the literal Safetensors attachment; verify checksum, metadata identity,
  dtype, shape, values, and tensor keys. Event loss does not erase a tensor;
  a tensor alone never creates a missing event.
- Recompute feasible regions using exact rational arithmetic; compare sorted
  vertices, representative mapping, validation, support, bounds, rounding, and
  duration classifications. Test duplicate/conflicting sample inputs and
  invalid bracketing windows as structured input errors.

CPU fixture agreement enables parallel implementation. It proves neither
concurrent CUDA correctness nor actual Host/Device accuracy. Those acceptance
boundaries remain in the [task plan](../README.md).
