# Deterministic Calibration v2

The [shared contract](shared-contract.md) preserves the original clock domains.
Contract version 2 retains calibration algorithm_version=1 and its existing
numerical expectations; the contract revision changes no calibration mathematics.
This document fixes the numerical clock calibration algorithm, not merely a set of acceptable
outputs. Implementations must reproduce [the calibration fixtures](fixtures/calibration.json).

## Input and Sampling

One sampler is `(Run, Host Producer, Device clock_domain_id)`. The Device domain
identifies physical Device UUID, process creation, and raw-clock Epoch; two
processes on the same physical Device have distinct domains until independently
validated mappings establish their relationship to the Host reference.

The sampler's Device-side storage (the raw-`G` readout buffer written by the
sampling Kernel) is a per-process allocation owned by that process's Device
Producer, independent of any Transport `observation_region`; see
[the shared contract's calibration sampler storage](shared-contract.md#calibration-sampler-storage-per-process-device-producer-not-inside-observation_region)
for its default layout and budget accounting. This placement does not change
the sampling or fitting algorithm described below.

Sample attempt sequences are positive and contiguous over a sampler Epoch.
Use one independent nonblocking CUDA Stream and at most one in-flight sample.
Read CLOCK_MONOTONIC immediately before submission (`H_before`), sample raw
Device `%globaltimer` (`G`), and read CLOCK_MONOTONIC after asynchronous
completion confirmation (`H_after`). A successful sample must prove the Device
read occurred within this window. Native and Python Host clocks must use the
same CLOCK_MONOTONIC reference; record implementation provenance and verify the
relationship during integration. Merely similarly named clocks are insufficient.

An unfinished attempt does not generate another in-flight task. Periodic
scheduling opportunities skipped while it is pending increment a separate
sampler `skipped_busy` diagnostic; they are not fabricated sample attempts.
Initialization, periodic collection, and normal finalization use the same
sequence. Timeouts publish failure evidence but retain in-flight resources.
A later completion records diagnostic recovery of that attempt, not a second
successful fit sample replacing the already failed outcome. A new attempt
starts only after resource completion permits it.

Input validation uses integer nanoseconds, requires H_before <= H_after,
positive IDs/sequences, and matching provenance. A duplicate byte-identical
sample is idempotent; contradictory samples are `sample_conflict`. Missing
attempts and explicit failed attempts make their block unavailable. Different
Epochs never share a block. Epoch change is explicit metadata, not a guessed
time-offset repair. A decreasing raw G is `clock_jump`; it invalidates that
block and requires a fresh Epoch before further fitting.

## Blocks and Feasible Region

Block `k` contains attempt sequences `10*k+1` through `10*k+10`, for k >= 0.
Positions 4 and 9 are validation; positions 1,2,3,5,6,7,8,10 fit the model.
Blocks are fixed by attempt number, not by arrival order or only successful
samples. A partial final block is `incomplete_block`. No sliding windows,
outlier deletion, automatic refitting, or fallback midpoint regression exists
in v2. The default interval is 100 ms; actual sample spacing remains evidence.

Use local coordinates x = G - G0, with G0 the first fitting sample's G, and
h = H_before(first fitting sample). Parameterize:

```text
H(G) = a * (G - G0) + c + h
H_before_i - h <= a * x_i + c <= H_after_i - h
a >= 0
```

The closed region uses a >= 0 to compute conservative extrema; admissible
models require a > 0. Require at least two distinct fitting G values, then
enumerate intersections of all pairs of boundary lines. Retain intersections
that satisfy every half-plane exactly, deduplicate exact rational vertices,
and sort lexicographically by (a,c). This defines P, including line/point
degeneracies. With finite windows and distinct fitting G values, a nonempty P
is bounded. An empty P or maximum feasible a <= 0 is `model_conflict`.
Distinct fitting G values must span the validation G values; otherwise mark
`validation_outside_support` and provide no model.

Arithmetic is exact rational arithmetic on original integer observations.
No epsilon, floating-point least squares, or magic slope threshold is used.
For rational p/q, q is positive, gcd(abs(p),q)=1, and zero is 0/1. JSON stores
numerator and denominator as decimal strings so rational parameters have no
fixed integer-width truncation. Public shape is
`{"numerator":"...","denominator":"..."}`.

Let V be the distinct vertices. Select a_star = mean(a over V) and
c_star = mean(c over V). A point is its own mean; a line uses its endpoints.
The mean has positive slope whenever P admits a positive slope. Report the
equivalent intercept b_star = h + c_star - a_star*G0. Keep the full vertices,
origins, source fitting sequences, and algorithm_version=1.

## Held-Out Validation and Validity

For each held-out sample, test exactly:

```text
H_before <= a_star * (G - G0) + c_star + h <= H_after
```

Both must pass; otherwise the whole block is `validation_failed`. Validation
does not tighten P, choose a different vertex, or split/refit the failed block.
The model remains conditional on an affine clock relationship over its segment;
this check is model consistency, not independent hardware timing truth.

The inclusive validity interval is [min(G_fit), max(G_fit)]. Raw events outside
it are `outside_support`, even if close. Preserve the uncovered interval
between adjacent blocks. For overlapping valid intervals, model ID is
`(sampler, block_index)`; require exactly one applicable block or return
`ambiguous_model` rather than silently selecting a model. No averaging or
interpolation across models/Epochs is allowed.

Unavailable-block precedence is: invalid input/sample conflict (error),
incomplete_block, sampling_gap, sample_failed, clock_jump, insufficient_span,
model_conflict, validation_outside_support, validation_failed. A complete block
with a missing numbered attempt is sampling_gap, even when its records arrive
unordered. Capture all contributing reasons as diagnostics; precedence selects
the single public status. Incomplete means the declared attempt range ends
before its tenth position; it is distinct from missing records in a closed full
attempt range.

## Mapping and Duration Outputs

For a supported event G:

```text
estimate_exact = a_star*(G-G0) + c_star + h
lower_exact = min(a*(G-G0) + c + h for (a,c) in V)
upper_exact = max(a*(G-G0) + c + h for (a,c) in V)
estimate_ns = round_to_nearest_integer_ties_to_even(estimate_exact)
lower_ns = floor(lower_exact)
upper_ns = ceil(upper_exact)
```

Return status, model ID, Epoch, original G/domain, exact estimate/lower/upper,
integer estimate/lower/upper, and `assumption=segment_affine_bracketing`.
Do not serialize non-finite numbers. Missing calibration returns null for all
mapped values and a structured reason. Host timestamps have a degenerate
identity mapping to the common Host reference; Host readout overhead is already
part of sampling windows, not a hidden correction.

For end minus start, propagate without assuming statistical independence:

```text
estimate_exact = estimate_end - estimate_start
lower_exact = lower_end - upper_start
upper_exact = upper_end - lower_start
```

Apply outward integer rounding after the subtraction. This conservative rule
also applies to two endpoints of the same model; it intentionally does not
exploit parameter covariance. If either mapping is unavailable, duration is
unavailable. Ordering is `after` only when lower_exact > 0, `before` only when
upper_exact < 0, `simultaneous` when both are zero, and `indeterminate` otherwise.
Never clamp a negative estimate or bound to zero. A protocol causal edge is
retained independently of this timing result.

Rate reporting uses `(a_star - 1) * 10^6` ppm with corresponding min/max feasible
slopes. Differences between block slopes or Device mappings use interval
subtraction; if zero lies in the difference interval, drift is unresolved.
Sample window widths, validation results, coverage intervals/gaps, attempted
and failed samples, skipped scheduling opportunities, source load phase, and
sampling overhead enter reports. Residuals are not true error; midpoint window
half-width is not an all-event error guarantee. No fixed accuracy or performance
threshold is introduced.

## Public Offline Interface

Proposed Python values are frozen typed models `Sample`, `SampleBlock`,
`CalibrationModel`, `MappedTime`, and `DurationEstimate`, serialized with the
field shapes in the calibration fixtures. Samples carry attempt_sequence,
H_before, G, H_after, outcome, and provenance. A model stores model_id,
algorithm_version, status, G0, h, vertices, a, b, fit_sequences,
validation_sequences, and valid_G. Unavailable models use null for numerical
fields. Errors carry code and source sample references.

```python
def calibrate(block: SampleBlock) -> CalibrationModel: ...
def map_time(model: CalibrationModel, raw_time_ns: int) -> MappedTime: ...
def estimate_duration(start: MappedTime, end: MappedTime) -> DurationEstimate: ...
```

`SampleBlock` includes declared attempt range, Run, Host Producer, Device clock
domain/Epoch, and load phase. Fit/validation membership is derived from attempt
position and exposed in outputs. Samples and models connect through these
public values; readers do not access collector slot layout. Raw artifacts stay
immutable. Perfetto conversion may choose a display origin, but retains the raw
integer clock, model ID, bounds, and exact mapping provenance in queryable data.
