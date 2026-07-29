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
   the resident Fabric path and its distributed executor slots; the first
   FfnAgent participant also hosts the Coordinator.

A model-layer request moves through the system as follows:

1. Every SGLang rank registers its resolved model workload with the daemon and
   attaches to a geometry-matching CUDA IPC arena.
2. The SGLang Instance publishes a rank-local request through
   `xpool.ops.ffn_shim`.
3. The AtnAgent consumes that mailbox operation and publishes a matching Fabric
   Submission.
4. The Coordinator forms an Invocation after all configured AtnAgents agree on
   its identity and geometry, then admits it to a distributed Executor.
5. Completion and result state return through the Fabric and Transport
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

The native extension is built through uv and scikit-build-core. CUDA, Torch,
SGLang, FlashInfer, and the NVIDIA NVSHMEM runtime are project dependencies.

## Quick Start

Clone the repository and create machine-local configuration files:

```bash
git clone ${REPO_URL}
cd xpool
cp .env.example .env
cp configs/xpool.example.toml configs/dev.local.toml
```

Edit `configs/dev.local.toml` to select the model root, model IDs, and CUDA
devices. Edit `.env` so `XPOOL_CONFIG` points to that file and configure
host-unique `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` paths. Both
files are ignored by Git.

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

Run the development validation layers:

```bash
uv run python -m tests --suite unit
uv run python -m tests --suite integration
```

Stop all SGLang and xpool processes before stopping the MPS controller:

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

Model paths are resolved by `XpoolConfig.model_path_of(model_id)`. Start from
[`configs/xpool.example.toml`](configs/xpool.example.toml) and
[`.env.example`](.env.example); keep host-specific paths in an ignored
`*.local.toml` file.

## SGLang Integration

Adapters are selected from the model architecture declared in `config.json`.
Model IDs provide configuration identity and path resolution rather than acting
as an adapter allowlist.

| Model family | Architecture | Adapter integration placements `(TP, DP)` |
| --- | --- | --- |
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | `(1, 1)`, `(2, 1)`, `(1, 2)` |
| Qwen3 | `Qwen3ForCausalLM` | `(1, 1)`, `(2, 1)` |
| GLM-4.7-Flash | `Glm4MoeLiteForCausalLM` | `(1, 1)`, `(2, 1)`, `(1, 2)` |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | `(1, 1)`, `(2, 1)`, `(1, 2)` |

The adapter and shim integration matrix exercises eager and Decode full CUDA
graph paths across the listed placements. Prefill piecewise CUDA graph coverage
applies when attention DP is one.

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

- [`PLAN.md`](PLAN.md) is the canonical architecture and implementation plan.
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
