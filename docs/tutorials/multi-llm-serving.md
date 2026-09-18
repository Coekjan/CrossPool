# Multi-LLM Serving on Two GPUs

This guide extends the [single-model quick start](quick-start.md) to serve
`Qwen/Qwen3-0.6B` and `Qwen/Qwen2.5-0.5B` together. Both Instances use the same
AtnAgent GPU and FfnAgent GPU. Complete the quick-start installation, model-root,
`.env`, and external MPS setup first.

## Configure both models

Stop the single-model SGLang process, AtnAgent, FfnAgent, and daemon before
editing their shared configuration. Leave the MPS controller running if other
CUDA clients use it.

Keep the same two GPU UUIDs in `.env` and the same device roles in
`configs/dev.local.toml`:

```toml
[atn]
devices = [0]

[ffn]
devices = [1]

[[models]]
id = "Qwen/Qwen3-0.6B"

[[models]]
id = "Qwen/Qwen2.5-0.5B"
```

Retain the other settings from the quick start, including
`vendor.model_base_uri`. Both model paths must contain `config.json` and their
local checkpoint files. These entries select models; they do not download
weights.

## Start the shared generation

From the repository root, run `export UV_ENV_FILE="$PWD/.env"` in every terminal.
Start one daemon, one AtnAgent, and one FfnAgent:

```bash
uv run xpool daemon serve
```

```bash
uv run xpool atnagent --cuda-device 0
```

```bash
uv run xpool ffnagent --cuda-device 1
```

Then start both SGLang Instances in separate terminals, before waiting for
either one to become healthy. Replace `/absolute/path/to/models` with the
`vendor.model_base_uri` selected above:

```bash
uv run sglang serve \
  --model-path /absolute/path/to/models/Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000
```

```bash
uv run sglang serve \
  --model-path /absolute/path/to/models/Qwen/Qwen2.5-0.5B \
  --host 127.0.0.1 \
  --port 31000
```

Use the same `.env` in all five terminals. Do not give the two Instances
different `CUDA_VISIBLE_DEVICES` mappings: they must agree with the daemon and
agents about the generation's device indices.

## Check both listeners

In another terminal with `UV_ENV_FILE` set, wait for System Ready and both HTTP
listeners. The daemon reports Serving Healthy only after both listeners pass
their startup health checks, but check each public endpoint before sending a
request:

```bash
until uv run xpool daemon check; do sleep 1; done
until curl --fail --silent --show-error --max-time 5 http://127.0.0.1:30000/health; do sleep 1; done
until curl --fail --silent --show-error --max-time 5 http://127.0.0.1:31000/health; do sleep 1; done
```

Send one request to each listener:

```bash
curl -sS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain shared GPU serving in one sentence.","sampling_params":{"temperature":0,"max_new_tokens":24}}'

curl -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain shared GPU serving in one sentence.","sampling_params":{"temperature":0,"max_new_tokens":24}}'
```

The two Instances keep separate logical KV contents and prefix caches while
sharing the attention GPU's physical KV Capacity Pool. Their model-specific
FFN weight shards coexist on the FFN GPU. This smoke check proves concurrent
two-model startup and inference on those two GPU roles; it does not exercise or
measure dynamic KV-capacity borrowing.

Stop both SGLang processes first, then the AtnAgent and FfnAgent, and finally
the daemon. Stop MPS only after every CUDA client using its controller has
exited.
