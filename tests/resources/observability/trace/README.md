# Offline trace reference

`bundle/` contains synthetic contract-v7 Perfetto traces and reports. It is an
offline query fixture, not evidence of runtime collection or hardware timing.
`identities.sql` inspects retained entity identities; `durations.sql` exposes
boundary status, uncertainty and calibration coverage alongside display time.

The input is the shared `normal_dense` causality case plus ten successful paired
samples for Host Producer 1 and Device clock domain 2, Epoch 1. Attempt `i`
(1 through 10) has Device time `100*i` and equal Host-before/after time
`100*i + 100` ns. Sample Event sequences start at 1001. The sampler declares
attempts `[1, 10]` and observation interval `[0, 1200]` ns. Source paths are
stable synthetic references `synthetic/producer-<id>.xpool`; offsets retain
the test fixture expansion positions. Export uses `reconstruct()`,
`build_calibration()` and `export_traces()` with no input digests.

Regenerate all bundle members together and update `bundle.json` digests. Keep
the shared contract inputs unchanged; independent SQL assertions own acceptance.
