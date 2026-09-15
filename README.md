# CrossPool

CrossPool develops the control and data-plane infrastructure for
resource-disaggregated multi-model serving. It integrates SGLang with a
CrossPool-owned control plane, CUDA IPC transport, and an NVSHMEM Fabric for
coordinating specialized GPU roles.

## Introduction

Multi-model serving brings GPU memory objects with different lifetimes into the
same system. Model weights are stable and model-defined, while KV-cache is
transient and follows request demand. Treating both as one monolithic allocation
couples static model placement to the dynamic memory available for active
requests, reducing aggregate resource flexibility.

CrossPool's central design idea is to manage these resources through separate
GPU roles: keep attention and its KV-cache together, organize model-side
resources in a pooled execution tier, and exchange hidden states over a
low-latency interconnect. CrossPool provides the control and data-plane
infrastructure for this separation.

CrossPool uses SGLang as its serving engine. SGLang continues to own request
scheduling, attention, logical KV allocation and prefix-cache semantics, CUDA
graph selection, and output postprocessing. CrossPool supplies elastic physical
KV backing, capacity coordination, model adapters, process lifecycle,
rank-local transport, Fabric coordination, and the graph-compatible tensor
boundary used to connect the GPU roles.

## Highlights

- **SGLang integration:** a pinned SGLang plugin installs architecture-specific
  adapters without replacing SGLang's request or KV-cache runtime.
- **Typed control plane:** one daemon coordinates process identity, generation
  membership, readiness, resource leases, failure, and orderly shutdown.
- **GPU data plane:** CUDA IPC mailboxes connect each SGLang rank to an AtnAgent;
  an NVSHMEM Fabric coordinates distributed invocations and executor admission.
- **Graph-backed FFN execution:** FfnAgents retain only their planned
  tensor-parallel weight shards and execute model-defined FFN layers through
  independently replayable Executor Lane graphs.
- **Elastic KV backing:** stable attention-side virtual addresses allow
  physical KV capacity to move between co-located Instances while SGLang keeps
  logical allocation and prefix-cache ownership.
- **CUDA graph integration:** the adapter and shim test matrix exercises eager
  execution, Decode Full CUDA Graph replay, and Prefill Breakable CUDA Graph
  replay when attention data parallelism is one.
- **Observable validation:** native and Python observers record graph, transport,
  and Fabric evidence, while the repository test runner owns GPU leases and
  descendant process cleanup.

## System Architecture

CrossPool has four process roles:

1. **SGLang Instance** is one model-serving deployment containing one or more
   Instance Ranks. Each rank owns its attention-side runtime; the CrossPool plugin
   binds the model, derives workload geometry, and routes shim calls into that
   rank's Transport arena.
2. **CrossPool daemon** is the host-only control plane. It owns registration,
   generation planning, Transport leases, readiness, failure selection, and
   shutdown coordination. It does not own a CUDA device.
3. **AtnAgent** owns CUDA IPC Transport arenas for one configured GPU and bridges
   rank-local requests into the generation Fabric.
4. **FfnAgent** is an NVSHMEM participant for one configured GPU. FfnAgents own
   the resident Fabric path and its distributed Executor Lanes; the first
   FfnAgent participant also hosts the Coordinator.

A model-layer request moves through the system as follows:

1. Every SGLang rank registers its resolved model workload with the daemon and
   attaches to a geometry-matching CUDA IPC arena.
2. The SGLang Instance publishes a rank-local request through
   `xpool.ops.ffn_shim`.
3. The AtnAgent consumes that mailbox operation and publishes a matching Fabric
   Submission.
4. The Fabric Coordinator forms an Invocation after all configured AtnAgents
   agree on its identity and geometry, then admits it to an Executor Lane.
5. The selected FfnAgents bind the target layer's resident weight shards,
   replay the Lane Graph, and deliver the required complete or rank-local
   output form.
6. Completion and output state return through the Fabric and Transport
   protocols to the originating SGLang rank.

System Ready is reported only after the configured processes are live,
Transport arenas are eligible, the Fabric generation is executable, every
Instance has crossed its initialization barrier, and the external CUDA MPS
controller is responsive. The daemon then checks each Instance's HTTP `/health`
endpoint and reports Serving Healthy separately; this is a one-time startup
observation, not continuous availability monitoring.

## Requirements

- Linux on x86-64
- [uv](https://docs.astral.sh/uv/) 0.11.19 or newer
- An uv-managed Python 3.12 interpreter
- CUDA Toolkit 13.2 and CCCL 3.2
- NVIDIA GPUs able to execute the selected kernels and CUDA graphs, with CUDA
  IPC and NVSHMEM access required by the selected topology; the example uses
  one attention-side GPU and one FFN-side GPU
- An externally managed CUDA MPS controller for runtime and GPU validation
- Local model weights for serving and model-dependent validation

The native extension is built through uv and scikit-build-core. CUDA bindings,
Torch, SGLang, and the NVIDIA NVSHMEM runtime are direct project dependencies.
uv uses the interpreter pinned in `.python-version` with managed Python
downloads enabled. NVSHMEM runs through the native C++/CUDA implementation;
Python NVSHMEM bindings are not required.

## Quick Start

Clone the repository and create machine-local configuration files:

```bash
git clone ${REPO_URL}
cd xpool
cp .env.example .env
cp configs/xpool.example.toml configs/dev.local.toml
```

For this minimal two-GPU example, edit `configs/dev.local.toml` to retain only
the `Qwen/Qwen3-14B` model, set `atn.devices = [0]` and `ffn.devices = [1]`,
and set `vendor.model_base_uri` to the directory containing the `Qwen/`
subdirectory. Edit `.env` so `XPOOL_CONFIG` points to that file and configure
host-unique `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` paths under
`/tmp/xpool-mps-$(id -u)`. Expand the command in a shell before writing the
absolute paths into `.env`; dotenv does not execute shell substitutions. Keep
`SGLANG_PLUGINS=xpool`. Both files are ignored by Git.

Install the complete development environment and rebuild the native extension:

```bash
export UV_ENV_FILE="$PWD/.env"
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
uv run xpool config dump
```

Create the MPS directories configured in `.env`, then start the controller with
every GPU visible to PyTorch clients. Keep these paths identical to the values
in `.env`.

```bash
mkdir -p "/tmp/xpool-mps-$(id -u)"/{pipe,log}
CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)" \
  uv run nvidia-cuda-mps-control -d
printf 'get_default_active_thread_percentage\n' | uv run nvidia-cuda-mps-control
```

MPS starts its server lazily when the first CUDA client connects. The controller
must cover every GPU enumerated by those clients; UUIDs avoid ordinal remapping.

Start the CrossPool processes from separate terminals in the repository root. All
terminals must use the same configuration and GPU ordinal space; do not remap
`CUDA_VISIBLE_DEVICES` independently for each process.

```bash
# Terminal 1: control plane.
export UV_ENV_FILE="$PWD/.env"
uv run xpool daemon serve
```

```bash
# Terminal 2: attention-side transport participant.
export UV_ENV_FILE="$PWD/.env"
uv run xpool atnagent --cuda-device 0
```

```bash
# Terminal 3: FFN execution participant.
export UV_ENV_FILE="$PWD/.env"
uv run xpool ffnagent --cuda-device 1
```

After the Agents have registered, start the configured model through the pinned
SGLang CLI. `MODEL_PATH` must resolve to the same model selected by
`configs/dev.local.toml`.

```bash
# Terminal 4: SGLang Instance.
export UV_ENV_FILE="$PWD/.env"
MODEL_PATH=/absolute/path/to/models/Qwen/Qwen3-14B
uv run sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000
```

Wait for CrossPool's System Ready verdict and the SGLang HTTP endpoint, then send
one request through the real FFN path. `xpool daemon check` checks the former;
its success does not imply that the public HTTP endpoint is healthy.

```bash
export UV_ENV_FILE="$PWD/.env"
until uv run xpool daemon check; do sleep 1; done
until curl --fail --silent --show-error --max-time 5 http://127.0.0.1:30000/health; do sleep 1; done

curl -sS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain pooled GPU execution in one sentence.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
  }'
```

Stop SGLang first, then the AtnAgent and FfnAgent processes, and finally the
daemon. Stop the MPS controller only after every CUDA client has exited:

```bash
printf 'quit\n' | uv run nvidia-cuda-mps-control
```

## Configuration

Start from [`configs/xpool.example.toml`](configs/xpool.example.toml) for
editable deployment settings and [`.env.example`](.env.example) for process
environment settings. Keep machine-local paths in an ignored `*.local.toml`
file and load `.env` into uv commands with `UV_ENV_FILE`, as in Quick Start.

Bootstrap environment variables are separate from the TOML schema:

| Variable | Purpose |
| --- | --- |
| `XPOOL_CONFIG` | Selects the runtime TOML file. |
| `SGLANG_PLUGINS=xpool` | Loads the CrossPool SGLang plugin. |
| `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` | Selects the externally managed MPS controller's directories. |

For each setting, supported sources take precedence in this order:

1. CLI arguments
2. Allowlisted environment variables
3. The TOML file selected by `XPOOL_CONFIG`
4. Defaults

Not every setting accepts every source. Use the command's `--help` for
available CLI overrides and `.env.example` for common environment overrides.
Debug controls are environment-only, not TOML entries; for example, use
`XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE`, not a `[debug.graph_observer]` table.

The following is a selected TOML configuration overview, not a complete
reference:

| Setting | Purpose |
| --- | --- |
| `daemon.host` / `daemon.port` | Selects the local control-plane address, not the SGLang serving address. |
| `vendor.model_base_uri` | Sets the absolute local model root. |
| `models[].id` / `models[].path` | Identifies a model and optionally overrides its absolute local weight path. |
| `models[].ffn_tp_size` | Fixes the model's FFN tensor-parallel width; omission uses the number of FfnAgents. |
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
cannot share a device. All processes must use the same CUDA device numbering.

Each `[[models]]` entry selects local weights: an explicit `path` takes
precedence; otherwise the path is `vendor.model_base_uri / id`. For example,
`id = "Qwen/Qwen3-14B"` below `/srv/models` resolves to
`/srv/models/Qwen/Qwen3-14B`. These settings do not download weights. Pass the
same resolved path to SGLang's `--model-path`.

Run `uv run xpool config dump` to inspect resolved values and their sources as
JSON. It reads the selected configuration; it does not generate a TOML template.

The [Control Plane design](docs/designs/control-plane.md#configuration-and-integration)
owns configuration semantics and validation contracts.

Memory calibration is optional: leave `ffn.device_memory_calibration` unset to
use analytic admission. To generate a profile, set it to an absolute output
path and run `uv run xpool memory-profile` with MPS running and the daemon and
serving processes stopped. The profiler uses a fixed model-independent corpus
and does not load the configured model weights. At startup, an explicitly
configured profile must exist and match the deployment environment; missing,
malformed, or incompatible profiles fail rather than silently falling back.

## SGLang Integration

Adapters are selected from the model architecture declared in `config.json`.
Model IDs provide configuration identity and path resolution rather than acting
as an adapter allowlist. Current qualification covers Qwen3, DeepSeek-V2-Lite,
GLM-4.7-Flash, and Qwen3-MoE. The
[SGLang E2E manifest](tests/harness/sglang/manifest.toml) is the authoritative
source for serving, numerical, topology, and graph-mode cases.

## Validation and Development

`xtest run` is the canonical composition root. It runs native CTest,
Unit, Integration, and E2E stages in their accepted order, schedules GPU work
against explicit resource requirements, and retains artifacts under
`.xpool-cache/test-runs/`.

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi

# Complete resource-eligible suite.
uv run xtest run

# Selected canonical stages.
uv run xtest run --suite cext --suite integration
uv run xtest run --suite e2e --strict-requirements
```

See [tests/README.md](tests/README.md) for suite placement, requirements, and
commands, and [tests/harness/README.md](tests/harness/README.md) for process,
GPU lease, endpoint, and artifact ownership.

CMake uses ccache for C, C++, and CUDA when available and no compiler launcher
is already configured. To disable it for a build, add
`--config-settings-package xpool:cmake.define.XPOOL_ENABLE_CCACHE=OFF`
to the development-environment sync command above.

## Repository Guide

- [`docs/designs/README.md`](docs/designs/README.md) maps the current implemented
  architecture.
- [`docs/plans/README.md`](docs/plans/README.md) maps candidate workstreams and
  their technical relationships.
- [`CONTEXT.md`](CONTEXT.md) defines the CrossPool domain language.
- [`src/xpool/`](src/xpool/) contains configuration, runtime roles, the daemon,
  SGLang integration, and Python/native boundaries.
- [`src/cext-include/xpool/`](src/cext-include/xpool/) and
  [`src/cext/`](src/cext/) contain the C++/CUDA Transport and Fabric data plane.
- [`tests/`](tests/) contains the native, Unit, Integration, and E2E validation
  layers and their reusable harnesses.
- [`docs/code-style.md`](docs/code-style.md) defines repository-wide coding
  conventions.

## License

CrossPool is available under the [MIT License](LICENSE).
