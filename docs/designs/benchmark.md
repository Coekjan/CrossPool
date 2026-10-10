# Serving Benchmarks

`xbench` measures multi-model serving through native SGLang streaming, either
against external endpoints or by starting a declared CrossPool deployment.
This document owns workload, measurement, continuation and report contracts.
Shared configuration, catalogue authoring, scheduling and process ownership
belong to [Test and Benchmark Tooling](tooling.md); the
[tooling tutorial](../tutorials/tooling.md) owns command usage. Performance values
are report-only under [Qualification](qualification.md). The
[glossary](../../CONTEXT.md) distinguishes cases, repetitions, attempts,
Measurements and Reports.

## Benchmark cases and workload execution

[`case.py`](../../src/xpool-dev/xbench/harness/serving/case.py) owns strict, immutable catalogue
declarations. Owned cases reference a portable deployment plus Instance/graph
launch settings and an optional explicit runtime base. Client cases name
externally owned endpoints and take no serving, device or MPS ownership.
Their workload and metric definitions are the same. Exact
models, rates, layouts and length policies belong to those fixed declarations.
Listing validates declarations without establishing deployment feasibility or
performance.

Prompt and arrival inputs are independent. Prompt JSONL supplies target-scoped
sample IDs with exactly one of text or token IDs; trace JSONL supplies unique
request IDs, targets, planned arrivals, prompt references and output caps.
Random prompts and per-target Poisson arrivals also work without external
datasets, in all generated/file-backed combinations. New formats normalize to
the existing `PreparedWorkload` rather than changing the executor.

Preparation is offline and precedes timing. Every file prompt is checked against
available local model metadata before selection; newly generated prompts are
validated at creation. Random token IDs come from matching local tokenizer/model
metadata, excluding special and out-of-vocabulary IDs.
Random prompt length belongs to the target's prompt declaration. Poisson output
caps belong to the target's `output_tokens`; arrival declarations own only
duration and per-target rates. Trace rows supply their own authoritative
`max_new_tokens`. Lengths and generated output caps use fixed values, inclusive
uniform ranges or truncated log-normal distributions. The serving integration
resolves model context and request reserves; generic workload code owns seeded
sampling within the legal inclusive interval `[L, U]`. Both log-normal policies
locate their untruncated median at `L + (U - L) * median_fraction`; truncation can
shift the sampled median. The catalogue owns each policy's fraction and log
standard deviation.

Joint conditioning samples input first, reserving the minimum legal output,
then samples output within the remaining request budget. An optional positive
`max_output_input_ratio` belongs only to output log-normal policies. For input
length `I` and ratio `R`, its ceiling is
`min(floor(R * I), limits.output_budget(I))`; the input interval must also admit
the output minimum under that ratio.
Empty intervals fail preparation. Fixed counts and file-backed values remain
authoritative rather than being clipped or resampled. Measured and warmup requests
use the same bounds. Retained workloads record resolved model context and actual
prepared prompt lengths, which the startup check reuses. File prompts establish
their input length before any generated output budget. Model/purpose-separated
random streams preserve the remaining target's replay in isolated comparisons.
Poisson arrivals use exponential first/interarrival gaps within `[0, T)` and a
stable merge; traces preserve source-row order for ties. Resolved prompt content,
schedule and horizon, rather than seeds alone, are the replay authority.

Owned execution checks prepared requests against the actual static service
limits after startup and before warmup. The integration interprets `/server_info`;
Elastic KV's immutable Capacity Group ceiling is distinct from its currently
mapped prefix. A mismatch retains the offending request and fails the case,
without resizing or resampling. Client endpoints acquire no arbitrary metadata
query requirement. Model context, request acceptance and immediate physical
backing remain separate facts.

The workload is prepared once and reused across repetitions. Execution settings
own a positive repetition count, default one, rather than the catalogue. Every
target completes separately prepared warmup before the common measurement origin.
Owned attempts start fresh systems; client attempts leave external state
under external control. Caches are not flushed implicitly. An empty schedule
retains a no-data result and skips allocation, serving, warmup and timing.

The client dispatches absolute planned arrivals through FIFO admission and a
global active-request limit, default 128. Capacity queues requests rather than
rejecting or dropping them. Actual enqueue, HTTP start and termination remain
separate observations. Warmup and measured requests wait for protocol completion
by default. An explicit positive `request_timeout_seconds` sets an absolute
deadline from HTTP start, including engine queueing, prefill and decode while
excluding client queue wait. Stream progress does not renew this deadline.
Complete owned startup defaults to 1,800 seconds. Normal drain has no total
deadline. Individual request failures are not retried and do not stop later
arrivals; owned-process or recording failures
interrupt execution. Cancellation accounts for every planned request and leaves
external servers running.

Warmup and measured requests flush their terminal records before closing the
HTTP response. A response-close failure is an infrastructure error; a completed
successful request retains its sample and latency while the attempt fails.
Recording failures remain infrastructure failures.

Each target selects `sglang` or `openai` through
[`ApiAdapter`](../../src/xpool-dev/xbench/harness/serving/api.py). The factory creates fresh
request-local state for every warmup and measured request. `SglangAdapter`
implements native `/generate` construction and response validation;
`OpenaiAdapter` raises `NotImplementedError` during protocol preflight, before
run allocation or HTTP execution. The client owns transport, timestamps and
event history. A malformed frame raises `ResponseProtocolError`, retaining its
rejected observation when available.
The client counts the current SSE wire frame incrementally, including field
prefixes and line delimiters, and bounds unfinished lines. Blank-line frame
boundaries reset the budget. Empty `data:` lines consume it; a transport batch
containing several individually valid frames does not combine their budgets.

## Measurement definitions

Native `/generate` streaming uses one client monotonic origin per attempt.
Raw times are seconds; wall time identifies the execution, not latency. Events
are timestamped before decoding/serialization, and cumulative generated text is
not repeated in every record. HTTP start is a client observation, not a claimed
server-receipt timestamp. Startup, warmup and shutdown are outside the window.

- HTTP TTFT is first positive-token observation minus HTTP start.
- Arrival TTFT is first positive-token observation minus actual enqueue.
- Queue wait is HTTP start minus enqueue; arrival lateness is enqueue minus
  planned arrival.
- Completion is valid `[DONE]` receipt, or establishment of failure/cancellation;
  subsequent transport close does not extend latency.

Token-count progress, including empty-text chunks, establishes first-token
timing. HTTP 200 alone is insufficient: success requires monotonic valid counts,
normal final usage/finish reason and `[DONE]`. Failed requests retain their valid
prefix and offending observations separately. No-token requests have unavailable
TTFT/ITL, not zero-valued samples.

A single-token increment after first progress supplies an observed ITL; a
multi-token increment of `k` supplies a token-estimated gap divided by `k`, with
interval weight `k`. Duplicate counts preserve the previous positive-progress
anchor; regressions fail protocol validation. For final count `N > 1` and first
observed count `C_first`, coverage is `(N - C_first) / (N - 1)`. First-chunk
intervals are unavailable. `stream_interval=1` does not establish independently
observed per-token clocks.

Successful requests alone enter main latency and ITL populations. Observed,
estimated and combined ITL distributions retain their distinct labels and
token-interval weights. Request TPOT is `(completion - first-token time) /
(N - 1)` and includes terminal tail; it is not the ITL distribution. Statistics
use population standard deviation and linear percentile rank
`(sample_count - 1) * p / 100`; CDFs are empirical steps. Empty populations retain
sample count zero and nullable statistics.

Logical input throughput includes successful requests' prompt tokens, including
cache hits, attributed at completion. It does not estimate device Prefill work.
Output throughput attributes positive count increments at their observed times,
with failed/cancelled partial output separate. The retained window classification
distinguishes three lifecycle facts:

- `complete`: normal arrival and queue drain ended. The window ends at the later
  of the declared horizon and final request termination, including normally
  drained request failures.
- `interrupted`: the measurement owner observed cancellation or a failure that
  aborted measurement and retained its actual stop before HTTP and serving
  teardown. Cancellation terminal observations fit within this end; neither
  teardown nor the unelapsed planned horizon extends it.
- `observed_prefix`: owner loss left no retained stop. Parent recovery uses only
  the last timestamp supported by validated retained observations. The actual
  stop is unknown; reports label prefix-only rates and time-series explicitly.

The measurement owner persists normal completion or observable interruption in
the attempt checkpoint before phase teardown. Later cleanup failure fails the
attempt without changing that timing fact. Reporting failure preserves
measurement facts and the original verdict. Terminal records for every request
do not prove that the planned horizon elapsed: an
interruption during its idle remainder stays incomplete. Summary execution
completeness uses the retained lifecycle fact separately from raw-evidence and
cleanup completeness.

Without usable timed evidence, the window end and classification are both
unavailable, even with a retained origin and a positive planned horizon. Startup
and warmup failures likewise have no measured window. Reports preserve valid
latency/ITL samples and failure/cleanup facts. Buckets use actual window width,
including a shortened final bucket and observations exactly at the end;
coalesced tokens receive no invented intra-chunk timestamps.

[`measure.py`](../../src/xpool-dev/xbench/harness/serving/measure.py) owns validation and metric
math; [`client.py`](../../src/xpool-dev/xbench/harness/serving/client.py) owns transport/timestamps,
and [`api.py`](../../src/xpool-dev/xbench/harness/serving/api.py) owns protocol construction and
validation. Live token progress is validated by the adapter and timestamped by
the client's monotonic clock. Offline measurement validation checks retained
chronology, token progress and terminal claims before statistical calculation.
Private interval, distribution and attribution helpers rely on those established
invariants.
`RequestState.terminal` calculates request metrics during execution. The recorder
persists them in `requests.jsonl`. Reporting consumes these saved values for
request-level distributions. Individual ITL samples and time-resolved throughput
use retained events, whose token increments and observation times cannot be
represented by request-level means and totals alone.

## Measurement evidence and reports

One locked benchmark invocation owns its measurements and derived reports:

```text
run.json
cases/<case-id>/
  case.json, workload.json
  prompts.jsonl, trace.jsonl, warmup.jsonl
  repetition-0001 -> repetition-0001.attempt-0001
  repetition-0001.attempt-0001/
    repetition.json, requests.jsonl, events.jsonl
    measurement.json          Common origin, only after timing starts
    environment.json, warmup.json, launch/, logs/
    report/                   Derived by offline reporting
      summary.json, report.md, cdf.csv, throughput.csv, render.json
      ttft-cdf.*, itl-cdf.*, throughput.*
```

Tool-owned result records follow their current declarations and contain no
schema, format, metric-revision or rendering-version tags. Catalogues are
field-driven declarations without a schema version. External serving metadata
retains its own input schema field. Actual software/build versions describe the
environment.
`run.json` retains required `tool_config`, selected case IDs, case references and
the current invocation outcome separately from report generation. Retained
execution settings supply the expected repetition count; current configuration
does not reinterpret a previous run.
`case.json` owns the case declaration, deployment provenance and repetition
references selecting one physical attempt per logical repetition. Attempts and
runtime/report projections obtain their case label from that parent declaration.
`workload.json` retains lightweight timing, seed
and digest metadata plus references to the normalized prompt, trace and warmup
JSONL files. Local model metadata is preparation input. Lightweight repetition
checkpoints retain timing, execution/error facts, core digests and nullable
cleanup/evidence flags. The worker cannot seal its own cleanup proof. `run`
retains JSONL and checkpoints; it neither persists an aggregate summary nor
invokes CSV export or Matplotlib.

`xbench run --all --continue DIR` resumes the original selection, repetition
count, effective deployment and prepared replay in the same run directory.
It installs the saved development configuration before ordinary bootstrap and
rejects explicit case, config, catalogue, repetition-count and cache overrides.
Current catalogue declarations locate executable modules and must retain the
original experiment conditions; description, provenance and authentication
secrets do not redefine an experiment. Device leases are newly acquired from
the actual execution environment. Continuation validates all selected original
prepared inputs before reopening; missing declarations, deployment snapshots or
replay fail without changing the artifact. Default continuation executes only
logical repetitions with no prior attempt; a startup failure remains attempted
even when measurement never began. Failed, interrupted or unverifiable effective
attempts remain unchanged unless `--rerun-failure` selects them for one complete
new attempt each. Missing measurement output with intact prepared replay follows
the same selection rule, not request-offset recovery or resampling. Both
`--rerun-failure` and `--fast-fail` are invocation actions rather than retained
configuration settings; the latter is also available on ordinary run.

A successful sealed attempt is reusable only after raw evidence validation and
its own verified cleanup, independently of the interrupted parent's seal.
Selected unsuccessful repetitions execute the complete saved workload at fresh
`repetition-NNNN.attempt-NNNN` paths. The runner preserves prior run manifests,
attempts and reports, and replaces only effective references and current run
state. The run verdict aggregates effective attempts rather than historical
failures. Skipped unsuccessful effective attempts keep the current run nonzero;
failures superseded by a new attempt contribute only to history. A fully
verified successful sealed run is a read-only no-op.
Continuation does not reclaim an unresolved runtime domain or establish that
the cause of an earlier failure has been fixed.

Logical repetition links point to the latest started attempt and update under
the existing exclusive run lock. Workers write physical paths. A link agrees
with its case's effective reference and targets the same logical repetition
within that case; it does not fall back to an older successful attempt.
Existing real repetition directories remain in place. Reopening withdraws the
parent completion marker; safe finalization restores it. Run prints the offline
report command after continuation but neither deletes nor regenerates reports.

Replay digests identify normalized prompt, trace and warmup content. Repetition
digests cover requests/events and the origin whenever it is retained,
independently of window availability. A usable timed window requires an origin;
reporting validates each retained origin's schema and digest. Recording loss
preserves original bytes/valid prefix and represents unsupported outcomes as
`evidence_missing`, with unknown timing rather than fabricated dispatch or engine
claims. Valid partial samples remain reportable with incomplete labels.

`environment.json` labels its bounded environment whitelist with
`environment_source`. Owned execution records `effective_serving_launch` from
the actual `system.launch.environment`; before startup establishes that launch,
the source is `unknown` with an empty mapping. Client execution records
`local_client`, describing load-generator inputs rather than external serving
conditions. Owned hardware observations capture
the allocated devices' UUID/name, memory bytes, PCI identity, links, CPU/NUMA affinity
and target/role placement once before timing, using bounded read-only queries.
Client `serving_metadata_path` optionally supplies declared external hardware and
package/build versions; absent values remain unknown. The tool does not substitute
load-generator hardware or query external metadata endpoints. Software values
identify their source; unavailable CUDA build information stays unknown rather
than triggering library or installation audits. Metadata capture failures are
diagnostics, not metric failures.

Prompt content is intentional replay data, explicitly retained by `run`.
Client `case.json` retains the complete endpoint URL, including plaintext
username and password, for reproducible execution and continuation. Sharing
raw artifacts shares those credentials. Report projections do not directly
export endpoint URLs. Diagnostic environment capture remains bounded rather
than dumping the complete environment. Logs, warmup diagnostics, metadata and
generated reports are outside mandatory metric digests and do not veto valid offline
aggregation.

`report` validates retained replay, request accounting, chronology, core digests
and final checkpoints, then aggregates saved request metrics and event samples.
It reads no previous aggregate summary and preserves measurement files and
original execution verdicts. A contradictory zero-result checkpoint is rejected,
and incomplete recording cannot become a successful measurement through reporting.

Each physical attempt's fixed `report/` directory owns `summary.json`, `report.md`,
`cdf.csv`, `throughput.csv`, `render.json` and the selected figure formats. Its
summary contains that attempt's metrics, replay identities, environment,
deployment and original verdict, plus the available invocation manifest.
Matplotlib uses a local DejaVu Serif nine-point paper style, embedded/path fonts
and colors plus line styles for grayscale differentiation. Width presets are
3.3 inches (`single`), 6.8 inches (`double`) and 1.65 inches (`half`); explicit
per-figure dimensions override presets. Subplot columns determine rows, with
2.4 inches per row when height is unset. Legends fit inside the selected canvas
and use retained Model IDs plus the aggregate series, without user label maps.
Metric titles and horizontal axis labels wrap to each subplot's width, keeping
their wording and units inside the canvas and clear of scientific-notation offsets.
Captions appear in Markdown; the optional title is the report heading. Formats
default to PDF, SVG and PNG, with 300 PPI for raster output. Effective rendering
settings and dimensions accompany the report. Latency axes use milliseconds;
throughput uses tokens per second. HTTP/arrival TTFT, observed/estimated/combined
ITL and aligned input/output throughput remain explicit. Unavailable data is annotated; defaults
do not smooth or clip tails. Rendering-library versions belong to report output.

`xbench report` resolves exact artifact IDs below its configured run root:
`RUN_ID` or
`RUN_ID/cases/FULL_CASE_UUID/repetition-NNNN` or the corresponding physical
`repetition-NNNN.attempt-NNNN` address. Benchmark run inputs expand to their
effective attempts; physical addresses can select historical sealed attempts.
Logical links select only the latest attempt, while existing real repetition
directories retain their physical meaning. Multiple inputs select a deduplicated
batch of independent
reports. Each benchmark report shows the models and aggregate for exactly one
attempt, and CLI stdout lists the generated directories. Exclusive run-store protection covers
loading through publication, coordinating report writers, readers and cleanup.

`report --list` discovers inactive sealed metadata without parsing all samples
or recomputing digests. Required execution settings establish eligibility;
unsupported or unsealed records are not reported. Failed or normally interrupted
runs retain their original outcome. A partial benchmark invocation lists eligible
repetition addresses rather than advertising the entire run. Generation protects
the run and performs full evidence validation; discovery does not recover or seal it.
Eligible historical attempts are also listed by physical address, with no
duplicate logical-link entry. Both the parent and historical attempt must be
sealed; a later parent completion does not seal old unfinished attempts.

Reports default to their owning physical benchmark attempt. Optional
`--output DIR` exports below
`DIR/xbench/RUN_ID/cases/CASE_UUID/repetition-NNNN.attempt-NNNN/report/`, preserving attribution
without relocating raw evidence. Current catalogue and deployment files are not
report inputs.

Repeated reporting overwrites tool-generated files in the existing directory
while preserving unrelated files. Rendering failure returns an error and leaves
measurement bytes and saved verdicts unchanged. Publication may partially
update a report; another invocation regenerates it without manual deletion.

Benchmark run code zero requires complete valid measurement, at least one
successful sample per repetition and safe cleanup. Drained request failures or
no-data results use one; configuration, infrastructure or recording failures use
two. Only a zero worker exit permits the normal measurement-result branch;
abnormal worker exits, including one, are infrastructure failures even with
successful retained requests. Signals retain codes 130 and 143 after cleanup.
Valid raw samples survive worker failure. The report command
returns zero for successful reporting even when source execution failed;
input/schema/output failures return two.
