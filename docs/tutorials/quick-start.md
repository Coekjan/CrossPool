# Quick Start: Qwen3-0.6B on Two GPUs

This guide starts one SGLang Instance with one attention GPU and one FFN GPU.
It exercises CrossPool FFN execution through the serving path. Use a Linux
host meeting the [repository requirements](../../README.md#requirements),
with the `Qwen/Qwen3-0.6B` checkpoint already stored locally. The two GPUs
must be available to CUDA IPC, NVSHMEM, and the externally managed CUDA MPS
controller.

## Configure the checkout

From the repository root, create ignored machine-local files:

```bash
uv --version
nvcc --version
cp .env.example .env
cp configs/xpool.example.toml configs/dev.local.toml
nvidia-smi --query-gpu=index,uuid,name --format=csv
```

The first two commands must report uv 0.12.17 or newer and CUDA Toolkit 13.2.
The Python environment supplies CMake and Ninja during the native build, while
the CUDA compiler remains a host prerequisite.

Choose two GPU UUIDs from the last command, in attention-then-FFN order. In
`.env`, keep `XPOOL_CONFIG=configs/dev.local.toml` and
`SGLANG_PLUGINS=xpool`, and set `CUDA_VISIBLE_DEVICES` to those two UUIDs in
that order. Their process-local CUDA indices are 0 and 1. Use the same `.env`
in every terminal; do not independently remap devices for different roles.

Create the MPS directories and display their expanded paths:

```bash
mkdir -p "/tmp/xpool-mps-$(id -u)"/{pipe,log}
printf 'CUDA_MPS_PIPE_DIRECTORY=%s/pipe\nCUDA_MPS_LOG_DIRECTORY=%s/log\n' \
  "/tmp/xpool-mps-$(id -u)" "/tmp/xpool-mps-$(id -u)"
```

Copy the two printed assignments into `.env`. Dotenv files do not evaluate
`$(id -u)`, so write the expanded absolute paths, not the command text.

In `configs/dev.local.toml`, keep `atn.devices = [0]` and
`ffn.devices = [1]`. Set `vendor.model_base_uri` to the absolute directory
containing `Qwen/`, remove the example's other `[[models]]` entries, and keep
only:

```toml
[[models]]
id = "Qwen/Qwen3-0.6B"
```

The resulting model path must contain `config.json`, for example
`/absolute/path/to/models/Qwen/Qwen3-0.6B/config.json`.

## Install and start MPS

Use the repository's uv-managed interpreter and pinned dependencies:

```bash
export UV_ENV_FILE="$PWD/.env"
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
uv run xpool config dump
```

The config dump should show exactly one model and the selected attention and
FFN device indices. Start MPS only if no controller already owns the selected
pipe directory; do not restart a controller serving other CUDA clients.

```bash
uv run nvidia-cuda-mps-control -d
printf 'get_default_active_thread_percentage\n' | uv run nvidia-cuda-mps-control
```

MPS starts its server lazily when the first CUDA client connects. The
controller must cover both UUIDs selected in `.env`.

## Start the serving processes

Open four terminals in the repository root. Run `export UV_ENV_FILE="$PWD/.env"`
in each terminal before its command. Start the first three processes, then
start SGLang after the agents have registered:

```bash
# Terminal 1: host control plane.
uv run xpool daemon serve
```

```bash
# Terminal 2: attention-side transport participant.
uv run xpool atnagent --cuda-device 0
```

```bash
# Terminal 3: FFN execution participant.
uv run xpool ffnagent --cuda-device 1
```

```bash
# Terminal 4: SGLang Instance. Use the path selected by vendor.model_base_uri.
uv run sglang serve \
  --model-path /absolute/path/to/models/Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000
```

In another terminal with `UV_ENV_FILE` set, wait for both CrossPool's System
Ready verdict and SGLang's HTTP health endpoint. System Ready alone does not
prove the public endpoint is healthy.

```bash
until uv run xpool daemon check; do sleep 1; done
until curl --fail --silent --show-error --max-time 5 http://127.0.0.1:30000/health; do sleep 1; done

curl -sS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain pooled GPU execution in one sentence.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
  }'
```

Stop SGLang first, then the AtnAgent and FfnAgent, and finally the daemon.
Stop the MPS controller only after every CUDA client using it has exited:

```bash
printf 'quit\n' | uv run nvidia-cuda-mps-control
```

## Next Step

To serve a second small model on these same two GPUs, follow
[Multi-LLM Serving](multi-llm-serving.md).
