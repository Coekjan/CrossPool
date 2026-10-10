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
```

Test inventory shows concrete pytest node IDs and resources. Known typed cases
also show description, Model IDs, portable geometry and row-specific graph
mode. Ordinary tests need no catalogue identity.

```bash
uv run xtest run
uv run xtest run --suite cext --suite integration
uv run xtest run --suite unit -k config
uv run xtest run tests/suites/unit/config/test_schema.py
uv run xtest run --suite Qwen/Qwen3-0.6B --strict-requirements
```

Tests default to configured suites. Explicit suites constrain Python paths and
filters; Python paths without suites define their own scope, while option-only
filters narrow configured Python suites. Neither implicitly includes CTest.
Graph qualification requires complete comparison groups.

## Add a fixed experiment

```bash
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
uv run xtest report RUN_ID --output exported-reports
```

Reports preserve the original execution verdict and read retained evidence.
Default reports belong to `RUN_ID/report/`; exports belong to
`DIR/xtest/RUN_ID/report/`.

## Inspect and retain

```bash
uv run xtest config dump
uv run xtest clean --dry-run
```

Inspection shows effective settings, sources, selected files and the shared
cache location. Cleanup acts only on inactive runs; publishing a report does
not change execution age. `--all` and `--keep` are mutually exclusive.
