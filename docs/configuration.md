# Configuration

CrossPool configuration is shared by the daemon, AtnAgents, FfnAgents, and
SGLang Instances. Start from [`configs/xpool.example.toml`](../configs/xpool.example.toml)
for deployment settings and [`.env.example`](../.env.example) for process
environment settings. Keep machine-local paths in an ignored `*.local.toml`
file and load `.env` into uv commands with `UV_ENV_FILE`.

The [Control Plane design](designs/control-plane.md#configuration-and-integration)
defines configuration semantics and validation contracts. This page is the
user-facing reference for selecting and inspecting those values.

## Sources and precedence

Supported sources take precedence in this order:

1. CLI arguments
2. Allowlisted environment variables
3. The TOML file selected by `XPOOL_CONFIG`
4. Defaults

Each setting declares its accepted sources. Use the command's `--help` for
available CLI overrides and `.env.example` for common environment overrides.
Debug controls use their registered environment variables. For example, set
`XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE`; debug settings have no TOML section.

Bootstrap environment variables are separate from the TOML schema:

| Variable | Purpose |
| --- | --- |
| `XPOOL_CONFIG` | Selects the runtime TOML file. |
| `SGLANG_PLUGINS=xpool` | Loads the CrossPool SGLang plugin. |
| `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` | Selects the externally managed MPS controller's directories. |

All processes in one deployment must use the same resolved configuration and
CUDA device numbering.

## TOML overview

The following is a selected configuration overview, not a complete schema
reference:

| Setting | Purpose |
| --- | --- |
| `daemon.host` / `daemon.port` | Selects the local control-plane address; SGLang serving uses its own listener. |
| `vendor.model_base_uri` | Sets the absolute local model root. |
| `models[].id` / `models[].path` | Identifies a model and optionally overrides its absolute local weight path. |
| `models[].ffn_tp_size` | Fixes the model's FFN tensor-parallel width; omission uses the number of FfnAgents. |
| `scheduler.slo` / `models[].slo` | Sets default TTFT/TBT targets; models may override both for Elastic KV arbitration. |
| `atn.devices` / `ffn.devices` | Assigns attention-side and FFN-side CUDA devices. |
| `scheduler.ffn_concurrency` | Sets the Executor Lane count, not a row or token budget. |
| `scheduler.ffn_policy` | Selects `fifo` or `random` admission. |
| `atn.device_memory_utilization` | Sets the maximum attention-device memory fraction used to freeze the Elastic KV Capacity Pool. |
| `scheduler.atn_concurrency` | Reserved for future attention compute admission; currently has no runtime effect. |
| `logging.level` / `logging.color` | Sets runtime log level and enables terminal-aware color on stderr. |
| `ffn.loader.parallelism` | Sets the number of checkpoint readers per FfnAgent. |
| `ffn.placement.parallelism` / `ffn.placement.timeout_seconds` | Sets solver workers and the whole-solve timeout in seconds. |
| `ffn.device_memory_extra_margin_bytes` | Adds an explicit device-memory admission margin. |
| `ffn.device_memory_calibration` | Selects an optional environment-qualified memory calibration profile. |

Each device list must be nonempty, unique, and ascending, and the two roles
cannot share a device.

## Model paths

Each `[[models]]` entry selects local weights: an explicit `path` takes
precedence; otherwise the path is `vendor.model_base_uri / id`. For example,
`id = "Qwen/Qwen3-14B"` below `/srv/models` resolves to
`/srv/models/Qwen/Qwen3-14B`. These settings select existing local weights.
Pass the same resolved path to SGLang's `--model-path`.

## Inspecting resolved configuration

Run the configuration dump to inspect resolved values and their sources as JSON:

```bash
uv run xpool config dump
```

The [Quick Start](tutorials/quick-start.md) shows a complete two-GPU setup.

## Memory calibration

Memory calibration is optional: leave `ffn.device_memory_calibration` unset to
use analytic admission. To generate a profile, set it to an absolute output
path and run `uv run xpool memory-profile` with MPS running and the daemon and
serving processes stopped. The profiler uses a fixed model-independent corpus
and runs without loading the configured model weights.

At startup, an explicitly configured profile must exist and match the
deployment environment; startup reports missing, malformed, or incompatible
profiles.
