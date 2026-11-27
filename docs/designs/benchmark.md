# Serving Benchmarks

`xbench` measures multi-model serving through streaming OpenAI Completions.
Run a two-model example in all three modes with the
[xbench tutorial](../tutorials/xbench.md). See
[Test and Benchmark Tooling](tooling.md) for configuration, catalogue
authoring and scheduling. The [qualification rules](qualification.md)
treat benchmark performance as report-only.

## Benchmark cases and workload execution

Enter serving experiments under `serving_cases`. The three modes share the
same workload preparation, measurement and reporting code. Their case schemas
are in [`case.py`](../../src/xpool-dev/xbench/harness/serving/case.py).

| Mode | How servers start | Case inputs |
| --- | --- | --- |
| `client` | The caller runs the servers. | An unauthenticated HTTP(S) root per target, with `served_model_name` if the endpoint uses an alias. |
| `hosted` | xbench assigns devices and ports, then runs a script supplied by the caller. | The executable script and optional script-defined `launch_settings`. |
| `owned` | xbench starts CrossPool through the SGLang integration. | The portable deployment, per-target Graph settings and optional runtime configuration. |

Client URLs accept an HTTP(S) scheme and host, with an optional port and
trailing slash. Reports identify the target by its Model ID even when a
client endpoint uses an alias. Hosted and owned servers use the Model ID
as the API model name. The [checked-in catalogue](../../benches/benches.toml)
compares mixed deployments with matched isolated references, then varies
request rates, token lengths, traffic balance, Executor Lanes or TP geometry.

### Preparing a workload

A case can generate random prompts or read them from JSONL. Each file prompt
has an ID unique within its target and supplies either text or token IDs.
Choose arrivals separately. Generate a Poisson sequence for each target or
read a JSONL trace with unique request IDs, targets, planned arrivals,
prompt IDs and output caps. xbench converts both kinds of input to
`PreparedWorkload` before starting measurement.

Preparation runs offline. Owned cases load the local model configuration and
tokenizer. Client and hosted cases also load them for random prompts or
log-normal output lengths, using paths from the
[runtime configuration](../configuration.md#development-tools-and-shared-cache).
Random prompts sample from the tokenizer vocabulary and skip special tokens.
Preparation validates file prompts when it has local model metadata. With
file prompts and fixed or explicitly ranged output caps, client and hosted
need only the workload files. Choose those caps to fit the server.

For generated prompts, set the input length in the prompt declaration. With
Poisson arrivals, set each target's rate in `arrivals` and its output length
policy in `output_tokens`. File trace rows set their own `max_new_tokens`.
Generated lengths can be fixed, uniform over an inclusive range, or truncated
log-normal. In that distribution, `median_fraction` places the untruncated
median at `L + (U - L) * median_fraction` for the legal interval `[L, U]`,
and `sigma` sets the standard deviation in log space. Truncation can shift the
median.

To sample a request, xbench reserves the minimum output length before drawing
the input length, then draws the output length from the remaining budget.
Owned cases account for SGLang's request reserves. External cases account for
the remaining context when they know both the context and input length.
An optional positive `max_output_input_ratio` applies only to log-normal
output lengths. With input length `I` and ratio `R`, the output cap is
`min(floor(R * I), limits.output_budget(I))`, and the input range must admit
the minimum output under that ratio. An empty range or an out-of-bounds fixed
value fails preparation. Warmup uses the same limits.

xbench draws exponential gaps for Poisson arrivals within `[0, T)` and merges
them stably. Traces preserve file order for simultaneous arrivals. Separate
random streams per model and purpose let a model keep its traffic when a peer
is removed. Token lengths also depend on the available request bounds, which
can vary across modes with the same seed. For a controlled comparison, keep
the prompt and trace JSONL fixed. Set `sampling.ignore_eos` explicitly when
exporting generated traffic for replay because it defaults to true for
random prompts and false for file prompts.

### Running requests

After startup, hosted cases check known prompt lengths and output caps
against any `max_model_len` the servers report. Owned cases read
`/server_info` and check the static Capacity Group ceiling, which can exceed
the currently mapped capacity. The workload stores the model context when
available. Token IDs give an exact input length, while text prompts have a
known length when preparation loads the tokenizer. If a check fails, xbench
saves the startup details and identifies the request before warmup. Client
mode sends requests directly to the configured endpoint.

xbench prepares each workload once for all repetitions. The repetition count
comes from the execution settings and defaults to one. Each managed attempt
starts fresh servers, while the client-mode caller controls the initial
cache state. Warmup ends before the common measurement clock starts, and
server caches carry over into measurement. With an empty arrival schedule,
xbench records a no-data result and skips device allocation and serving.

The client uses the arrival schedule to feed a FIFO queue and starts up to
128 HTTP requests at once by default. Requests wait for protocol completion
unless a positive `request_timeout_seconds` sets a fixed limit from HTTP
start. That limit covers the engine queue, prefill and decode. Client
queueing happens before this clock starts. Managed startup has a default
limit of 1,800 seconds, and normal queue drain can run until all requests end.
xbench makes at most one HTTP attempt per request and continues with later
arrivals after an individual request fails. A managed process or recording
failure interrupts the attempt. On cancellation, xbench accounts for every
planned request and leaves client-mode servers running.

The client writes each request's result before closing its HTTP response.
If closing fails, the attempt reports an infrastructure error and preserves
any completed latency sample.

[`OpenaiAdapter`](../../src/xpool-dev/xbench/harness/serving/api.py) prepares
`POST /v1/completions` with text or token IDs and an explicit `max_tokens`.
Every warmup and measured request sends `stream = true` and requests both
`include_usage` and `continuous_usage_stats` in `stream_options`. Requests
advertise `Accept-Encoding: identity` so the SSE reader can consume the response
bytes directly. External endpoints must return cumulative output counts on
generation frames and a final usage frame. The client timestamps and records
the events. `StreamEvent` validates the extracted fields. If a malformed
frame yields a rejected event, the client records it with the protocol error.
The client counts the current SSE wire frame incrementally, including field
prefixes and line delimiters, and bounds unfinished lines. Blank-line frame
boundaries reset the budget. Empty `data:` lines consume it; a transport batch
containing several individually valid frames does not combine their budgets.

### Hosted launch protocol

Set `script` relative to the catalogue. xbench runs it directly, and the
script selects the Python environment, engine options and checkpoints.

| Command | Contract |
| --- | --- |
| `SCRIPT requirements` | From the script directory, emit one JSON object to stdout with a nonnegative integer `device_count` and nonempty `listeners`. Write diagnostics to stderr. xbench validates the declaration and retires the probe's process group before leasing devices. |
| `SCRIPT start --model MODEL_ID@URL [--model MODEL_ID@URL ...]` | In the attempt directory, start services in the foreground inside the supervised process group. Use the assigned URLs and ordered `CUDA_VISIBLE_DEVICES`. Exit with failure if any required service exits unexpectedly. |

`listeners` groups the Model IDs by shared listening address. For example,
`[["team/first", "team/second"], ["team/third"]]` requests two addresses.
The groups must cover the case's targets exactly once. xbench supplies an
absolute `--settings PATH` JSON file to both commands when the case declares
`launch_settings`. Zero devices declares a CPU-only launcher.

xbench uses the [shared TCP reservation](tooling.md#devices-and-endpoints) for
each loopback address. It checks for cancellation and releases the TCP listeners
just before starting the script. It holds the port locks until cleanup succeeds.
To confirm readiness, xbench checks that the launcher's process group owns each
listener and that `GET /v1/models` advertises each Model ID at its assigned
address only. One startup deadline covers all listeners and model-list requests.
Probes and benchmark requests use direct local connections. The task supervisor
returns the devices once the processes have stopped.

To include startup details in the report, write `launch/commands.json` before
starting the servers. For each command, record its arguments and selected
environment variables, with optional Python and package details. When the
file is missing or invalid, xbench records a diagnostic and continues.
Generate the JSON Schemas with `model_json_schema()` on
[`HostedRequirements`](../../src/xpool-dev/xbench/harness/serving/case.py) and
[`HostedLaunchRecord`](../../src/xpool-dev/xbench/harness/serving/environment.py).

## Measurement definitions

The client measures time in seconds on a monotonic clock and records the wall
time at the start of measurement. It timestamps events before decoding and
records when it starts each HTTP request. Startup, warmup and shutdown
fall outside the measurement window.

- HTTP TTFT is first positive-token observation minus HTTP start.
- Arrival TTFT is first positive-token observation minus actual enqueue.
- Queue wait is HTTP start minus enqueue. Arrival lateness is enqueue minus
  planned arrival.
- Completion is the receipt of `[DONE]` or the time when the client records a
  failure or cancellation. The later HTTP close happens outside that latency.

Token-count progress, including empty-text chunks, establishes first-token
timing. A successful stream provides nondecreasing cumulative output counts on
generation frames, a normal finish reason, a final `choices = []` usage frame,
and `[DONE]` in that order. Final usage must match the last observed output
count. It supplies totals and any reported cache counts, with zero timing or
throughput samples. Failed requests retain their valid prefix and offending
observations. TTFT and ITL are unavailable when the stream has no token progress.

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
- `interrupted` records the measurement owner's stop once. After supervised
  cleanup the runner extends that stop to the latest retained measurement
  observation if needed. Serving shutdown and the unelapsed planned horizon
  are outside the measurement window.
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
`requests.jsonl` retains terminal timestamps, token counts, and outcomes.
`events.jsonl` retains timestamped stream observations. Live recording and
offline validation use the same stream progress rules. Reporting derives
request metrics, ITL distributions and time-resolved throughput from the
validated records. Each request's token intervals supply both its ITL metrics
and the pooled distribution.

## Measurement evidence and reports

### Saved files

Each run saves the prepared workload and the results of every attempt.

```text
run.json
cases/<case-id>/
  case.json, workload.json
  prompts.jsonl, trace.jsonl, warmup.jsonl
  repetition-0001 -> repetition-0001.attempt-0001
  repetition-0001.attempt-0001/
    repetition.json, requests.jsonl, events.jsonl
    measurement.json          Common origin, only after timing starts
    environment.json, launch/, logs/
    warmup/                    Created when warmup starts
      requests.jsonl, events.jsonl
    report/                   Derived by offline reporting
      summary.json, report.md, cdf.csv, throughput.csv, render.json
      ttft-cdf.*, itl-cdf.*, throughput.*
```

`run.json` saves `tool_config`, the selected case IDs, their paths and the
current run result. Each `case.json` describes the case and its effective
deployment or hosted launch declaration. It stores the prepared prompt, trace
and warmup digests and points to the effective attempts. `workload.json` names
the replay files and records their timing settings and known model contexts.
`repetition.json` records the attempt's timing, outcome, digests
and whether measurements and cleanup were verified. The supervisor verifies
cleanup before xbench can mark the attempt complete. Reports and continuation
read the current schemas for these files.

An attempt worker validates the prepared workload against its case.
Continuation and report generation also verify the normalized replay records
against the case's digests. Attempt digests cover request and event records and
any saved timing origin. A timed report requires a valid origin. When recording
stops unexpectedly, xbench preserves the recorded prefix and marks unknown outcomes
as `evidence_missing`. It leaves their timing unknown. Reports label the
remaining samples as incomplete. Startup details, logs, warmup diagnostics and
reports are kept outside the required measurement digests.

`environment.json` records the load generator's software and selected
environment variables. Once managed servers are ready, xbench adds their
endpoints, assigned devices and available launch details under `serving`.
Device details include UUID, name, memory, PCI identity, topology and CPU/NUMA
affinity when available. For owned cases it also records attention and FFN
device assignments and the effective runtime configuration path. Client
callers record their servers' configuration separately. If an environment
query fails, xbench keeps what it collected and reports the error in
`capture_errors`.

### Continuing a run

`xbench run --all --continue DIR` restores the saved configuration, case
selection, repetition count, deployments and prepared workload from the run
directory. The original catalogue, deployment TOML and datasets are inputs to
fresh runs. A resumed run leases devices from the current environment, and
xbench rejects overrides to its saved case or tool settings.

Before changing the run directory, xbench validates the saved declarations,
deployments and workload. For each hosted case that needs a new attempt,
it runs the saved script's `requirements` command with the saved settings.
The script digest and reported requirements must match, and xbench checks
the digest again just before launch.

By default, continuation runs repetitions that have never started. Startup
failure counts as an attempt. If owned weight preflight fails or startup
fails with verified cleanup, xbench prints the case ID, repetition number, exit
code and error to stderr and saves that error in the attempt. Use
`--rerun-failure` to execute the complete saved workload once for each failed,
interrupted or unverifiable effective attempt. `--fast-fail` also works on
fresh runs and responds to new execution failures. Both flags apply to the
current invocation.

xbench reuses a successful sealed attempt if its measurements and cleanup
validate, even when the parent run was interrupted. If a sealed run already
has a reusable attempt for every required repetition, continuation returns
the saved result. A new attempt gets its own directory and becomes the
effective attempt. Earlier attempts and reports remain available.
Failures that still belong to the effective attempts keep the run's exit
code nonzero. Resolve incomplete cleanup before attempting to continue.

Under the run lock, xbench points each `repetition-NNNN` symlink at its latest
started attempt, named `repetition-NNNN.attempt-NNNN`. The target must be a
sealed attempt for that repetition in the same case to use the symlink for
reporting. Reopening clears the run's completion marker. Finalization
restores it when all cleanup checks pass.

### Reports

`xbench report` validates the saved workload, request and event records and
attempt checkpoints before computing statistics. Continuation uses the same
checks to decide which attempts to reuse. Reports are built from the raw records
and preserve the saved measurements and execution result. If the checkpoint
claims success despite missing or contradictory measurements, reporting fails.

Each attempt's `report/` directory contains `summary.json`, `report.md`,
`cdf.csv`, `throughput.csv`, `render.json` and the selected figures. The summary
includes metrics, workload digests, environment, deployment and original
result, plus the run manifest when available.

Figures use nine-point DejaVu Serif with line styles that distinguish series
in grayscale. The default widths are 3.3 inches (`single`), 6.8 inches
(`double`) and 1.65 inches (`half`). Set explicit dimensions to override
the presets. The number of columns determines the number of rows, which
are 2.4 inches high by default. Legends use the Model IDs and aggregate
series. Titles and horizontal axis labels wrap within each subplot.
Captions appear in Markdown, and an optional title becomes the report
heading. Reports default to PDF, SVG and PNG with 300 PPI for raster output.
The report records the rendering settings, dimensions and library versions.

Latency axes use milliseconds and throughput uses tokens per second. Figures
label HTTP and arrival TTFT, distinguish observed, estimated and combined ITL,
and show input and output throughput on the same clock. The defaults plot
the full distributions without smoothing and mark unavailable data.

`xbench report` accepts a run ID or an address such as
`RUN_ID/cases/FULL_CASE_UUID/repetition-NNNN`. A run ID selects the effective
attempts, and a `.attempt-NNNN` suffix selects a particular attempt. The
command generates each selected report once, with separate model and aggregate
results, and prints the report directories. It holds the run lock through
validation and writing so report generation and cleanup can coordinate.

`report --list` reads metadata from inactive runs. It lists only sealed
results with supported settings, including failed and interrupted runs with
their saved outcomes. For a partial run, it lists the eligible attempts by
address. A historical attempt needs its own seal as well as the run's seal.
Listing leaves validation of the measurement files to report generation.

Reports default to the attempt directory. Use `--output DIR` to write them
under `DIR/xbench/RUN_ID/cases/CASE_UUID/repetition-NNNN.attempt-NNNN/report/`.
Report generation reads the saved run files, so the original catalogue and
deployment files can be moved. A rendering error can leave a partial report.
Rerun the command to regenerate its files and keep unrelated files in place.

Benchmark run code zero requires complete valid measurement, at least one
successful sample per repetition and safe cleanup. Drained request failures or
no-data results use one; configuration, infrastructure or recording failures use
two. Only a zero worker exit permits the normal measurement-result branch;
abnormal worker exits, including one, are infrastructure failures even with
successful retained requests. Signals retain codes 130 and 143 after cleanup.
Valid raw samples survive worker failure. The report command
returns zero for successful reporting even when source execution failed;
input/schema/output failures return two.
