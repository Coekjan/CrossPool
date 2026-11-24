# Test and Benchmark Workflow

After [installation](quick-start.md#install), create any missing local
configuration files and choose the `.env` file that uv should load.

```bash
cp configs/xpool.example.toml configs/xpool.local.toml
cp configs/xkit.example.toml configs/xkit.local.toml
cp .env.example .env
export UV_ENV_FILE="$PWD/.env"
```

Keep your existing local files and edit their settings. Use `xpool.local.toml`
for devices, model paths and calibration, and `xkit.local.toml` for tool
settings such as catalogues, retention and report style. See the
[configuration reference](../configuration.md#development-tools-and-shared-cache)
for precedence and relative paths.

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

Use `--catalog` to select an independent baseline catalogue. From another
directory, point `uv` at the installed CrossPool development project.

```bash
uv run --project /work/CrossPool xbench list --catalog /work/baselines/benches.toml
uv run --project /work/CrossPool xbench run --catalog /work/baselines/benches.toml --case CASE_PREFIX
```

Declare experiments under `serving_cases`. Use `mode = "client"` for an
existing OpenAI Completions endpoint, `mode = "hosted"` for an executable local
launcher, and `mode = "owned"` for a portable CrossPool deployment. The
[benchmark design](../designs/benchmark.md#benchmark-cases-and-workload-execution)
defines the case fields, script contract and required streaming usage.

If a hosted launcher cannot supply its resource declaration, `list` shows
`devices=-` for that case and prints the cause on stderr. Running that case
requires a valid declaration.

Cases run concurrently when their complete deployments fit disjoint device
leases; client cases use no local lease. Repetitions of one case are ordered and
reuse its prepared replay. The default is one; `run --repetitions N` overrides
`xbench.repetitions` for the selected cases. Parallel cases share host resources.
Select one case per invocation when the comparison requires isolated execution.

xbench keeps running other cases after a failure if cleanup succeeds. Use
`--fast-fail` to stop scheduling after the first failure while current attempts
finish. Preparation errors or incomplete cleanup stop the run. Use Ctrl+C to
cancel a run that is still active.

Continue an interrupted or failed prepared experiment with:

```bash
# Run only repetitions that have never started.
uv run xbench run --all --continue BENCH_RUN_DIR

# Also give unsuccessful attempted repetitions one new attempt each.
uv run xbench run --all --continue BENCH_RUN_DIR --rerun-failure
```

With `--continue`, `--all` uses the original cases and their saved workload,
deployment and settings. The original catalogue and dataset files can change
once preparation is complete. If preparation was interrupted, start a fresh
run. If xbench reports incomplete cleanup, resolve it before resuming.

xbench reuses successful attempts after verifying their measurements and
cleanup. With `--rerun-failure`, each selected failed or interrupted repetition
starts again from the beginning in a new directory. Any failure left in place
still counts toward the run's final result. xbench checks a hosted launcher
again only if the case needs another attempt. `--fast-fail` responds to new
failures in the current invocation. Generate reports again after resuming to
see the new results.

Rates in the list are per model. Requests can finish after the arrival window
closes while the queue drains. See [workload and request timing](../designs/benchmark.md#benchmark-cases-and-workload-execution)
for sampling rules and timeout settings.

## Run the serving example

Follow the [xbench tutorial](xbench.md) to benchmark two models in client, hosted
and owned mode and render the reports.

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

Use complete run and case IDs when addressing reports. `report --list` shows
saved results from inactive runs, including failed and interrupted ones that
have finished writing their results. A partial benchmark run may list individual
attempts. Report generation checks the saved measurements and can run after
the original catalogue or deployment files have moved.

A benchmark run ID produces a separate report for each effective attempt. Use
`repetition-NNNN` for the latest attempt once it has finished writing its
results. Include the `.attempt-NNNN` suffix to select a particular attempt,
including a historical one shown by `report --list`.

Default reports belong to a test run or physical benchmark attempt. `--output DIR`
exports to `DIR/xtest/RUN_ID/report/` or
`DIR/xbench/RUN_ID/cases/FULL_CASE_UUID/repetition-NNNN.attempt-NNNN/report/`.
Regenerating a report replaces its generated files and preserves the measurements,
benchmark result and any unrelated files. The command prints the report
directories and returns zero if it renders them successfully.

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
