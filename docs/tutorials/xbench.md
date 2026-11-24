# Benchmark Two Models with xbench

Measure two models on two local devices. In `client` mode, you start vLLM and
SGLang yourself. In `hosted` mode, xbench starts the same servers through a
launcher script. In `owned` mode, xbench starts CrossPool.

## Prepare the catalogue

Follow the [Quick Start](quick-start.md) to install the development tools and
set up `.env`. The example uses `/work/CrossPool` for the checkout,
`/work/baselines` for the experiment, and `/work/models` for the checkpoints.
Set `vendor.model_base_uri` in the file selected by `XPOOL_CONFIG` to your
checkpoint directory so xbench can load both tokenizers.

Select two free devices in `.env` and use the same order for all three runs.
Run the xbench commands from one shell in the checkout.

```bash
cd /work/CrossPool
export UV_ENV_FILE="$PWD/.env"
export BENCH_ROOT=/work/baselines
export BENCH_CATALOG="$BENCH_ROOT/benches/benches.toml"
export XPOOL_CACHE_ROOT="$BENCH_ROOT/.xpool-cache"

model_set='Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B'
mkdir -p "$BENCH_ROOT/benches/launchers" "$BENCH_ROOT/configs/deployments/$model_set"
cp "configs/deployments/$model_set/atn1-ffn1-lanes2.toml" "$BENCH_ROOT/configs/deployments/$model_set/"
```

Use the following layout for the experiment directory.

```text
/work/baselines/
├── benches/
│   ├── benches.toml
│   └── launchers/start_baselines.py
└── configs/deployments/Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B/
    └── atn1-ffn1-lanes2.toml
```

Save the catalogue below as `$BENCH_CATALOG`. Each case warms up both models,
then schedules requests over five seconds at an average of one arrival per
second per model. The shared seed drives Poisson arrivals and random prompts
with log-normal input lengths and output limits.

```toml
[serving_cases.0a1d9f6a-aa19-44ef-aad2-2f4fe4348a68]
mode = "client"
description = "Measure manually started vLLM and SGLang baselines."
seed = 0
warmup_requests_per_target = 1

[serving_cases.0a1d9f6a-aa19-44ef-aad2-2f4fe4348a68.arrivals]
kind = "poisson"
duration_seconds = 5.0
rates = { "Qwen/Qwen3-0.6B" = 1.0, "Qwen/Qwen2.5-0.5B" = 1.0 }

[[serving_cases.0a1d9f6a-aa19-44ef-aad2-2f4fe4348a68.targets]]
model_id = "Qwen/Qwen3-0.6B"
base_url = "http://127.0.0.1:30000"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }

[[serving_cases.0a1d9f6a-aa19-44ef-aad2-2f4fe4348a68.targets]]
model_id = "Qwen/Qwen2.5-0.5B"
base_url = "http://127.0.0.1:31000"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }

[serving_cases.9bb07bb0-3718-4017-83a5-2edf175142b8]
mode = "hosted"
description = "Launch vLLM and standalone SGLang with one device each."
script = "launchers/start_baselines.py"
seed = 0
warmup_requests_per_target = 1

[serving_cases.9bb07bb0-3718-4017-83a5-2edf175142b8.arrivals]
kind = "poisson"
duration_seconds = 5.0
rates = { "Qwen/Qwen3-0.6B" = 1.0, "Qwen/Qwen2.5-0.5B" = 1.0 }

[[serving_cases.9bb07bb0-3718-4017-83a5-2edf175142b8.targets]]
model_id = "Qwen/Qwen3-0.6B"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }

[[serving_cases.9bb07bb0-3718-4017-83a5-2edf175142b8.targets]]
model_id = "Qwen/Qwen2.5-0.5B"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }

[serving_cases.567bf8c9-5f23-424a-a109-c28a74d5e12c]
mode = "owned"
description = "Launch both models with shared attention and FFN devices."
deployment = "atn1-ffn1-lanes2"
seed = 0
warmup_requests_per_target = 1

[serving_cases.567bf8c9-5f23-424a-a109-c28a74d5e12c.arrivals]
kind = "poisson"
duration_seconds = 5.0
rates = { "Qwen/Qwen3-0.6B" = 1.0, "Qwen/Qwen2.5-0.5B" = 1.0 }

[[serving_cases.567bf8c9-5f23-424a-a109-c28a74d5e12c.targets]]
model_id = "Qwen/Qwen3-0.6B"
graph_mode = "decode-full-prefill-breakable"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }

[[serving_cases.567bf8c9-5f23-424a-a109-c28a74d5e12c.targets]]
model_id = "Qwen/Qwen2.5-0.5B"
graph_mode = "decode-full-prefill-breakable"
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.002, sigma = 0.5 } }
output_tokens = { kind = "lognormal", median_fraction = 0.1, sigma = 0.5, max_output_input_ratio = 2.0 }
sampling = { temperature = 0.0, ignore_eos = true }
```

## Prepare the baseline environments

Install vLLM 0.34.0 in `/work/envs/vllm-baseline` and SGLang 0.5.20 in
`/work/envs/sglang-baseline`. Follow the installation guides for
[vLLM](https://docs.vllm.ai/en/v0.34.0/getting_started/installation/gpu/)
and [SGLang](https://docs.sglang.io/docs/get-started/install).

Keep CrossPool packages out of the standalone SGLang environment. Leave
`SGLANG_PLUGINS` unset in the baseline terminals to use SGLang's default plugins.

## Run a client benchmark

Run the two server commands in separate terminals. Replace `FIRST_DEVICE_UUID`
and `SECOND_DEVICE_UUID` with the first and second selected UUIDs from `.env`.

```bash
CUDA_VISIBLE_DEVICES=FIRST_DEVICE_UUID HF_HUB_OFFLINE=1 \
  uv run --no-project --no-env-file --no-python-downloads --python /work/envs/vllm-baseline \
  vllm serve /work/models/Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 30000 --served-model-name Qwen/Qwen3-0.6B
```

```bash
CUDA_VISIBLE_DEVICES=SECOND_DEVICE_UUID HF_HUB_OFFLINE=1 \
  uv run --no-project --no-env-file --no-python-downloads --python /work/envs/sglang-baseline \
  sglang serve /work/models/Qwen/Qwen2.5-0.5B \
  --host 127.0.0.1 --port 31000 --served-model-name Qwen/Qwen2.5-0.5B
```

Once both servers are ready, check their model names from the xbench shell.
If that shell uses an HTTP proxy, add `127.0.0.1` to `NO_PROXY` or `no_proxy`.

```bash
curl -fsS http://127.0.0.1:30000/v1/models
curl -fsS http://127.0.0.1:31000/v1/models
```

The catalogue uses each Model ID as the API model name. To reuse a server with a
different name, set the target's `served_model_name` to its returned `data[].id`.

```bash
client_run=$(uv run xbench run --catalog "$BENCH_CATALOG" --case 0a1d9f6a --repetitions 1)
```

Stop both servers in their terminals and wait for the devices to become
available before the hosted run.

## Launch vLLM and SGLang in hosted mode

Save the script as `$BENCH_ROOT/benches/launchers/start_baselines.py` and use
the same checkpoint and environment paths as in client mode. Its `requirements`
command asks for two devices and a separate address for each model. xbench
passes those addresses to `start` as `--model MODEL_ID@URL` arguments.

```python
#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

MODEL_ROOT = Path("/work/models")
VLLM_ENV = Path("/work/envs/vllm-baseline")
SGLANG_ENV = Path("/work/envs/sglang-baseline")
MODELS = (
    ("Qwen/Qwen3-0.6B", "vllm", VLLM_ENV),
    ("Qwen/Qwen2.5-0.5B", "sglang", SGLANG_ENV),
)
UV_RUN = ["uv", "run", "--no-project", "--no-env-file", "--no-python-downloads", "--python"]

parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(dest="command", required=True)
subcommands.add_parser("requirements")
start = subcommands.add_parser("start")
start.add_argument("--model", action="append", required=True)
args = parser.parse_args()

if args.command == "requirements":
    model_ids = [model for model, _, _ in MODELS]
    print(json.dumps({"device_count": len(model_ids), "listeners": [[model] for model in model_ids]}))
    raise SystemExit(0)

urls = dict(binding.split("@", 1) for binding in args.model)
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
commands = []
for (model, engine, venv), device in zip(MODELS, devices, strict=True):
    address = urlsplit(urls[model])
    environment = {"CUDA_VISIBLE_DEVICES": device, "HF_HUB_OFFLINE": "1"}
    argv = [
        *UV_RUN,
        str(venv),
        engine,
        "serve",
        str(MODEL_ROOT / model),
        "--host",
        address.hostname,
        "--port",
        str(address.port),
        "--served-model-name",
        model,
    ]
    commands.append((argv, environment))

try:
    records = [{"argv": argv, "environment": environment} for argv, environment in commands]
    Path("launch/commands.json").write_text(json.dumps({"commands": records}), encoding="utf-8")
except OSError as error:
    print(f"launch command record unavailable ({error})", file=sys.stderr)

os.environ.pop("SGLANG_PLUGINS", None)
for argv, environment in commands:
    os.posix_spawnp(argv[0], argv, os.environ | environment)

pid, status = os.wait()
raise SystemExit(f"serving process {pid} exited with status {os.waitstatus_to_exitcode(status)}")
```

The launcher starts one server on each device and waits in the foreground so it
can report an early exit. xbench stops both servers after measurement.

```bash
chmod +x "$BENCH_ROOT/benches/launchers/start_baselines.py"
uv run xbench list --catalog "$BENCH_CATALOG"
hosted_run=$(uv run xbench run --catalog "$BENCH_CATALOG" --case 9bb07bb0 --repetitions 1)
```

`list` shows two devices for the hosted and owned cases. The client case shows
zero because you manage its servers.

## Launch CrossPool in owned mode

The `atn1-ffn1-lanes2` deployment runs both models on a shared attention device
and a shared FFN device. xbench uses `XPOOL_CONFIG` to find the local checkpoints.

```bash
owned_run=$(uv run xbench run --catalog "$BENCH_CATALOG" --case 567bf8c9 --repetitions 1)
```

## Read the reports

Use the last part of each output directory as the run ID when generating reports.

```bash
uv run xbench report "${client_run##*/}" "${hosted_run##*/}" "${owned_run##*/}"
```

Open `cases/CASE_UUID/repetition-0001/report/report.md` in each run directory
for its summary, latency distributions and throughput plots. The case directory
also contains the generated prompts, arrivals and warmup requests. For export
options, see the [reporting guide](tooling.md#discover-evidence-and-render).
