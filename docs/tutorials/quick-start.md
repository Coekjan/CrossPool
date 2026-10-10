# Quick Start: Qwen3-0.6B on Two Devices

This guide starts one SGLang Instance with one attention device and one FFN device.
It exercises CrossPool FFN execution through the serving path. Use a Linux
host meeting the [repository requirements](../../README.md#requirements),
with the `Qwen/Qwen3-0.6B` checkpoint already stored locally. The two devices
must support CUDA IPC and NVSHMEM, with `nvidia-cuda-mps-control` on PATH.
The daemon manages attention-side MPS; FFN execution bypasses it.

## Configure the checkout

From the repository root, create ignored machine-local files:

```bash
uv --version
nvcc --version
cp .env.example .env
cp configs/xpool.example.toml configs/xpool.local.toml
cp configs/xkit.example.toml configs/xkit.local.toml
nvidia-smi --query-gpu=index,uuid,name --format=csv
```

The first two commands must report uv 0.12.17 or newer and CUDA Toolkit 13.2.
The Python environment supplies CMake and Ninja during the native build, while
the CUDA compiler remains a host prerequisite.

Choose two devices from the last command, in attention-then-FFN order. Numeric
selectors name the `nvidia-smi` indices shown by the query; runtime entries
normalize the ordered selection to full physical UUIDs before initialization.
In `.env`, keep `XPOOL_CONFIG=configs/xpool.local.toml`,
`XKIT_CONFIG=configs/xkit.local.toml` and
`SGLANG_PLUGINS=xpool`, and set `CUDA_VISIBLE_DEVICES` to the two selected UUIDs
in that order. Their deployment-visible indices are 0 and 1. Use the same
`.env` in every terminal; do not independently remap devices for different
roles.

Leave MPS pipe/log selection to the role entry points. They install the
attention endpoint or direct FFN bypass before driver initialization.

In `configs/xpool.local.toml`, keep `atn.devices = [0]` and
`ffn.devices = [1]`. Set `vendor.model_base_uri` to the absolute directory
containing `Qwen/`, remove the example's other `[[models]]` entries, and keep
only:

```toml
[[models]]
id = "Qwen/Qwen3-0.6B"
```

The resulting model path must contain `config.json`, for example
`/absolute/path/to/models/Qwen/Qwen3-0.6B/config.json`.

## Install

Use the repository's uv-managed interpreter and pinned dependencies:

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
export UV_ENV_FILE="$PWD/.env"
uv run xpool config dump
```

`UV_ENV_FILE` loads `.env` for `uv run`, not for `uv sync`. The sync command
builds for CMake's default CUDA architectures. If both selected devices are A100s,
you can instead limit that build to their architecture:

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool \
  --config-settings-package xpool:cmake.define.XPOOL_CUDA_ARCHITECTURES=80-real
```

Select every required architecture when the devices differ; `80-real` is only the
A100 example. The build option does not belong in `.env`.

The config dump should show exactly one model and the selected attention and
FFN device indices. The daemon starts its own attention controller when serving
begins. Independent deployments do not coordinate physical-device use; arrange
their placement before launch. Do not stop a controller serving other clients.

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
uv run xpool atnagent --device 0
```

```bash
# Terminal 3: FFN execution participant.
uv run xpool ffnagent --device 1
```

```bash
# Terminal 4: SGLang Instance. Use the path selected by vendor.model_base_uri.
uv run xpool exec -- sglang serve \
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
    "text": "Explain pooled device execution in one sentence.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
  }'
```

Finish or cancel any in-flight startup, then stop SGLang and wait for its workers
and helpers to exit. Start no new clients during retirement. Then send SIGTERM or
press Ctrl-C in the daemon terminal. The daemon retires the Agents and Fabric, stops
its MPS controller and removes its owned scope directory. Do not terminate
joined Agents independently. An early daemon signal waits for known Instance
exit; it does not initiate SGLang shutdown. Unconfirmed cleanup keeps the living
owner and diagnostics available for manual resolution.

## Next Step

To serve a second small model on these same two devices, follow
[Multi-LLM Serving](multi-llm-serving.md).
