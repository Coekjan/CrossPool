# xpool

xpool develops the control and data-plane infrastructure for
resource-disaggregated multi-model serving. It integrates SGLang with an
xpool-owned control plane, CUDA IPC transport, and an NVSHMEM Fabric for
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
low-latency interconnect. xpool provides the control and data-plane
infrastructure for this separation.

xpool uses SGLang as its serving engine. SGLang continues to own request
scheduling, attention, KV-cache management, CUDA graph selection, and output
postprocessing. xpool supplies the model adapters, process lifecycle, rank-local
transport, Fabric coordination, and graph-compatible tensor boundary used to
connect the GPU roles.

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
- **CUDA graph integration:** the adapter and shim test matrix exercises eager
  execution, Decode full CUDA graph replay, and Prefill piecewise CUDA graph
  replay when attention data parallelism is one.
- **Observable validation:** native and Python observers record graph, transport,
  and Fabric evidence, while the repository test runner owns GPU leases and
  descendant process cleanup.

## System Architecture

xpool has four process roles:

1. **SGLang Instance** owns one model process and its attention-side runtime.
   The xpool plugin binds the model, derives workload geometry, and routes the
   model's shim calls into a rank-local Transport arena.
2. **xpool daemon** is the host-only control plane. It owns registration,
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

External readiness is reported only after the configured processes are live,
Transport arenas are eligible, the Fabric generation is executable, every
Instance has crossed its initialization barrier, and the external CUDA MPS
controller is responsive.

## Requirements

- Linux on x86-64
- [uv](https://docs.astral.sh/uv/) 0.11.19 or newer
- An uv-managed Python 3.12 interpreter
- CUDA Toolkit 13.2 and CCCL 3.2
- NVIDIA GPUs matching the selected topology; the example configuration uses
  one attention-side GPU and one pooled-side GPU
- An externally managed CUDA MPS controller for runtime and GPU validation
- Local model weights for model-dependent validation

The native extension is built through uv and scikit-build-core. CUDA bindings,
Torch, SGLang, and the NVIDIA NVSHMEM runtime are direct project dependencies.

## Quick Start

Clone the repository and create machine-local configuration files:

```bash
git clone ${REPO_URL}
cd xpool
cp .env.example .env
cp configs/xpool.example.toml configs/dev.local.toml
```

For this minimal two-GPU example, edit `configs/dev.local.toml` to retain only
the `Qwen/Qwen3-14B` model, place its AtnAgent on GPU 0 and its FfnAgent on GPU
1, and set `vendor.model_base_uri` to the directory containing the `Qwen/`
subdirectory. Edit `.env` so `XPOOL_CONFIG` points to that file and configure
host-unique `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` paths. Keep
`SGLANG_PLUGINS=xpool`. Both files are ignored by Git.

Install the complete development environment and rebuild the native extension:

```bash
export UV_ENV_FILE="$PWD/.env"
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
uv run xpool config dump
```

Create the MPS directories configured in `.env`, then start the controller with
every GPU visible to PyTorch clients. The following path is an example; keep it
identical to the values in `.env`.

```bash
mkdir -p /tmp/xpool-mps-12345/{pipe,log}
CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)" \
  uv run nvidia-cuda-mps-control -d
printf 'get_default_active_thread_percentage\n' | uv run nvidia-cuda-mps-control
```

Start the xpool processes from separate terminals in the repository root. All
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

Once SGLang is healthy, wait for the complete xpool generation to become ready
and send one request through the real FFN path:

```bash
export UV_ENV_FILE="$PWD/.env"
until uv run xpool daemon check; do sleep 1; done

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

Runtime configuration is resolved in this order:

1. CLI arguments
2. Allowlisted environment variables
3. The TOML file selected by `XPOOL_CONFIG`
4. Registry defaults

The main configuration boundaries are:

| Setting | Purpose |
| --- | --- |
| `XPOOL_CONFIG` | Selects the runtime TOML file. |
| `SGLANG_PLUGINS=xpool` | Loads the xpool SGLang plugin. |
| `vendor.model_base_uri` | Sets the external model root. |
| `models[].id` / `models[].path` | Identifies a model and optionally overrides its absolute path. |
| `devices.atn_cuda_devices` | Places AtnAgent roles. |
| `devices.ffn_cuda_devices` | Places FfnAgent roles. |
| `scheduler.*` | Configures attention and executor concurrency and Fabric scheduling. |
| `ffn.loader.*` | Configures bounded checkpoint-reading parallelism. |
| `ffn.placement.*` | Configures placement solving and explicit device-memory margin. |
| `memory.calibration_path` | Selects an optional environment-qualified memory calibration profile. |

Model paths are resolved by `XpoolConfig.model_path_of(model_id)`. Start from
[`configs/xpool.example.toml`](configs/xpool.example.toml) and
[`.env.example`](.env.example); keep host-specific paths in an ignored
`*.local.toml` file.

Memory calibration is optional: analytic admission works without a profile.
When a device-local correction is useful, set an absolute
`memory.calibration_path` and run `uv run xpool memory-profile` before starting
the serving processes. The profiler uses a fixed model-independent corpus and
does not load the configured model weights.

## SGLang Integration

Adapters are selected from the model architecture declared in `config.json`.
Model IDs provide configuration identity and path resolution rather than acting
as an adapter allowlist. Current qualification covers Qwen3, DeepSeek-V2-Lite,
GLM-4.7-Flash, and Qwen3-MoE. The
[SGLang E2E manifest](tests/harness/sglang/manifest.toml) is the authoritative
source for serving, numerical, topology, and graph-mode cases.

## Validation and Development

`python -m tests` is the canonical composition root. It runs native CTest,
Unit, Integration, and E2E stages in their accepted order, schedules GPU work
against explicit resource requirements, and retains artifacts under
`.xpool-cache/test-runs/`.

```bash
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi

# Complete resource-eligible suite.
uv run python -m tests

# Selected canonical stages.
uv run python -m tests --suite cext --suite integration
uv run python -m tests --suite e2e --strict-requirements
```

See [tests/README.md](tests/README.md) for suite placement, requirements, and
commands, and [tests/harness/README.md](tests/harness/README.md) for process,
GPU lease, endpoint, and artifact ownership.

## Repository Guide

- [`docs/designs/README.md`](docs/designs/README.md) maps the current implemented
  architecture.
- [`docs/plans/README.md`](docs/plans/README.md) maps candidate workstreams and
  their technical relationships.
- [`CONTEXT.md`](CONTEXT.md) defines the xpool domain language.
- [`src/xpool/`](src/xpool/) contains configuration, runtime roles, the daemon,
  SGLang integration, and Python/native boundaries.
- [`src/cext-include/xpool/`](src/cext-include/xpool/) and
  [`src/cext/`](src/cext/) contain the C++/CUDA Transport and Fabric data plane.
- [`tests/`](tests/) contains the native, Unit, Integration, and E2E validation
  layers and their reusable harnesses.
- [`docs/code-style.md`](docs/code-style.md) defines repository-wide coding
  conventions.

## License

xpool is available under the [MIT License](LICENSE).
