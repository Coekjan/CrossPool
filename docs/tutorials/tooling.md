# Test and Benchmark Workflow

Complete [installation](quick-start.md#install), create the two ignored local
configuration files and load `.env` in the invocation shell:

```bash
cp configs/xpool.example.toml configs/xpool.local.toml
cp configs/xkit.example.toml configs/xkit.local.toml
cp .env.example .env
export UV_ENV_FILE="$PWD/.env"
```

Edit existing files instead of overwriting local setup. Runtime devices, model
paths and optional calibration belong to `xpool.local.toml`; catalogue,
retention, test defaults and report presentation belong to `xkit.local.toml`.
The [configuration reference](../configuration.md#development-tools-and-shared-cache)
owns sources and relative-path rules.

## Discover and select

```bash
uv run xtest list --suite e2e
uv run xtest list --suite unit -k config
uv run xbench list
```

Test inventory shows concrete pytest node IDs and resources. Known typed cases
also show description, Model IDs, portable geometry and row-specific graph
mode. Ordinary tests need no catalogue identity. Benchmark inventory explains
each fixed experiment under its full UUID.

```bash
uv run xtest run
uv run xtest run --suite cext --suite integration
uv run xtest run --suite unit -k config
uv run xtest run tests/suites/unit/config/test_schema.py
uv run xtest run --suite Qwen/Qwen3-0.6B --strict-requirements

uv run xbench run --case 58fcaacc
uv run xbench run --case CASE_PREFIX_A --case CASE_PREFIX_B
uv run xbench run --all
```

Tests default to configured suites. Explicit suites constrain Python paths and
filters; Python paths without suites define their own scope, while option-only
filters narrow configured Python suites. Neither implicitly includes CTest.
Graph qualification requires complete comparison groups.

Benchmark execution requires `--all` or repeated `--case`. A nonempty prefix
must identify exactly one catalogue UUID; ambiguous prefixes report candidates.
The complete catalogue is an expensive experiment, not a discovery command.

Cases run concurrently when their complete deployments fit disjoint device
leases; client cases use no local lease. Repetitions of one case are ordered and
reuse its prepared replay. The default is one; `run --repetitions N` overrides
`xbench.repetitions` for the selected cases. Parallel cases share host resources.
Select one case per invocation when the comparison requires isolated execution.

A safely cleaned-up case failure is recorded while other cases and later
repetitions continue. Use `run --fast-fail` to stop new admissions after a new
failure; active experiments still finish and clean up. Input preparation errors
and unconfirmed cleanup retain their safety stopping boundaries. Ctrl+C or
SIGTERM requests cancellation of active experiments.

Continue an interrupted or failed prepared experiment with:

```bash
# Run only repetitions that have never started.
uv run xbench run --all --continue BENCH_RUN_DIR

# Also give unsuccessful attempted repetitions one new attempt each.
uv run xbench run --all --continue BENCH_RUN_DIR --rerun-failure
```

`--all` selects the artifact's original cases, not all current catalogue rows.
Saved settings, deployment and replay remain authoritative; explicit `--case`,
`--config`, `--catalog`, `--repetitions` and `--cache-root` cannot accompany
continuation. Resolve any retained runtime domain before continuing. Complete
successful attempts with validated evidence and verified cleanup are reused.
Failed or interrupted attempts require `--rerun-failure`; they execute their
entire saved workload in a new attempt directory. Without that flag their
failures still keep the run unsuccessful. `--fast-fail` can accompany either
continuation mode and responds only to new execution failures, not retained
ones. Neither option is saved in configuration. Preparation-stage interruptions
with missing original inputs require
a new run. Continuation preserves previous attempts and reports; generate the
latest reports explicitly when execution finishes.

The serving catalogue uses independent per-model Poisson rates and
bounded truncated log-normal lengths. Inspect each declaration with `list`:
rates are per model. Both length policies place their untruncated median within
the legal interval `[L, U]` as `L + (U - L) * median_fraction`. Input is sampled
first; output is then bounded by the remaining request budget and the optional
output log-normal `max_output_input_ratio`. The current cases use a ratio of
two. Truncation can shift the sampled median. Runtime request limits are checked
before owned warmup. A capacity mismatch
retains an infrastructure failure rather than changing the workload. The
arrival window ends before queue drain, so long outputs can extend execution
well beyond that window. Requests wait for completion by default; the case's
optional `request_timeout_seconds` limits HTTP duration when explicitly set.
See [request timing and failure semantics](../designs/benchmark.md#benchmark-cases-and-workload-execution).

## Add a fixed experiment

```bash
uv run xbench case-gen --type serving --from 58fcaacc
uv run xtest case-gen --type topology --from aa7a86cf
```

The tool appends a validated raw declaration with a new UUID and prints its
absolute file and inclusive line range. Edit the copied conditions and English
description before execution. Keep its full table key permanent. A different
fixed experiment receives a new UUID; description and presentation changes
retain it. Deployment references remain suffix-free basenames under the shared
Model ID directory convention. Inspect the edited declaration with `list`.

## Discover evidence and render

```bash
uv run xtest report --list
uv run xtest report RUN_ID
uv run xbench report --list
uv run xbench report RUN_ID
uv run xbench report RUN_ID/cases/FULL_CASE_UUID/repetition-0001
uv run xbench report RUN_ID/cases/FULL_CASE_UUID/repetition-0001.attempt-0001
uv run xbench report RUN_ID --layout half --format pdf --output exported-reports
```

Artifact addresses use exact retained identities, not catalogue prefixes.
Discovery reads inactive sealed metadata; failed or normally interrupted runs
retain their original outcome. Partial benchmark invocations expose eligible
repetition addresses. Generation validates the retained evidence and uses no
current catalogue or deployment files. Unsupported or unsealed evidence is not
reportable.

Whole-run reporting generates independent reports for the effective attempts,
not one combined report across cases. The logical `repetition-NNNN` symlink
selects the latest started attempt. Its physical `.attempt-NNNN` address names
one exact execution, including sealed historical failures listed by discovery.
An unsealed latest target does not fall back to an earlier attempt. Existing
real repetition directories keep their physical address meaning.

Default reports belong to a test run or physical benchmark attempt. `--output DIR`
exports to `DIR/xtest/RUN_ID/report/` or
`DIR/xbench/RUN_ID/cases/FULL_CASE_UUID/repetition-NNNN.attempt-NNNN/report/`. Repeated inputs
produce each concrete report once. Regeneration replaces owned derived files;
measurements, original verdicts and unrelated report files remain intact.
The command prints generated directories and returns zero when reporting
succeeds, independently of the benchmark execution's verdict. Client artifacts
retain passwords contained in endpoint URLs; do not share raw artifacts without
considering their credentials and prompt content.

Configure common figure presets with `--layout single|double|half`, selected
formats with repeated `--format`, and raster density with `--ppi`. `--width`
and `--height` set canvas inches; `--no-legend` hides legends. Per-figure subplot
columns, captions, dimensions and report title belong in `xbench.report` TOML.
Captions are Markdown, not image text.

## Inspect and retain

```bash
uv run xtest config dump
uv run xbench config dump
uv run xtest clean --dry-run
uv run xbench clean --keep 5
uv run xbench clean --all
```

Inspection shows effective settings, sources, selected files and the shared
cache location. Cleanup acts only on inactive runs; publishing a report does
not change execution age. `--all` and `--keep` are mutually exclusive.
