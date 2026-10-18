# GPT-OSS MXFP4 manual qualification

This guide runs the implemented GPT-OSS checks on six A100 40GB devices,
restricted to physical device indices 1 through 6.
The checkpoint owner supplies the original MXFP4 Safetensors checkpoint.
The [retained qualification report](qualification.md) records passing results
for both real checkpoints, memory qualification and the strict repository suite.
The commands below reproduce those checks; synthetic component results alone
do not establish model readiness.

## 1. Select the checkout and devices

Run all commands from the same shell:

```bash
cd /home/LAB/zuozc26/CrossPool
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export XPOOL_CONFIG="$PWD/configs/gpt-oss-20b.local.toml"
nvidia-smi
```

Reserve these six devices for this run. The portable deployment uses two
attention devices with TP2 and four FFN devices with TP1 and one Executor Lane.
Configuration indices are relative to the visible view: attention indices
`[0, 1]` select physical devices 1 and 2, and FFN indices `[2, 3, 4, 5]`
select physical devices 3 through 6. Physical device 0 is excluded.
Run the following stages sequentially. `xtest` owns the daemon, serving workers,
MPS preparation, and shutdown; start the tests without a separately running
CrossPool daemon or serving instance on those devices.

## 2. Configure the checkpoint

The current checkpoint is at `/home/LAB/yezr/models/openai/gpt-oss-20b/`.
Supply `config.json`, the complete
`model.safetensors.index.json` and all its referenced shards, and the original
tokenizer/configuration files. Preserve the serialized MXFP4 blocks and scales.
The shard files must be regular files: this repository's checkpoint boundary
rejects symbolic links to shards.

The local file `configs/gpt-oss-20b.local.toml` has been prepared for this
checkout and is intentionally ignored by Git. A complete configuration for
these checkpoints is:

```toml
cache_root = "../.xpool-cache"

[daemon]
host = "127.0.0.1"
port = 9810

[logging]
level = "info"
color = true

[scheduler]
ffn_concurrency = 1
slo = { ttft_ms = 1000, tbt_ms = 50 }

[atn]
devices = [0, 1]

[ffn]
devices = [2, 3, 4, 5]
device_memory_calibration = "/home/LAB/zuozc26/CrossPool/.xpool-cache/qualification/gpt-oss-20261017-router/memory-calibration.json"
device_memory_extra_margin_bytes = 0

[vendor]
model_base_uri = "/home/LAB/yezr/models"

[[models]]
id = "openai/gpt-oss-20b"
path = "/home/LAB/yezr/models/openai/gpt-oss-20b"
atn_tp_size = 2
atn_dp_size = 1
ffn_tp_size = 1
```

On another checkout, adjust the absolute paths and the shell's working
directory. Keep runtime model paths in CrossPool configuration.

## 3. Build the matching native extension

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
uv run --no-sync xpool config dump
```

Rebuilding is required because the quantized graph resources change the native
ABI to 86, including the explicit Router GEMM entry. Inspect the dump for the
selected model, path, and device groups before continuing.

Generate the model-neutral memory calibration before loading real weights:

```bash
uv run --no-sync xpool memory-profile
```

This runs 18 fitting and three held-out worlds on the selected devices and
writes the configured calibration path. Keep a matching existing profile
instead of regenerating it when its ABI, allocator, software and device
environment are unchanged. An ABI-85 profile cannot be used with ABI 86.
The calibration accounts for NVSHMEM and library startup overhead; an
uncalibrated placement can exhaust a 40-GB device during communication setup.

## 4. Check checkpoint metadata before loading CUDA weights

```bash
uv run --no-sync python - <<'PY'
from xpool.config import init_global_config
from xpool.runtime.ffnagent import architecture, checkpoint

config = init_global_config()
assert len(config.models) == 1
model_id = config.models[0].id
model_path = config.model_path_of(model_id)
spec = architecture.load(model_id=model_id, model_path=model_path)
key_view = checkpoint.read_checkpoint_key_view(model_path)
assert (model_path / "tokenizer.json").is_file()
assert (model_path / "tokenizer_config.json").is_file()
print("model:", model_id)
print("path:", model_path)
print("architecture:", spec.architecture_name)
print("layers:", len(spec.layers))
print("checkpoint shards:", len(set(key_view.values())))
print("checkpoint metadata: OK")
PY
```

For 20b, expect `GptOssForCausalLM` and 24 layers. This checks source config,
shard presence, and exact FFN key coverage. The actual loading tests also
validate tensor shapes, packed/scales dtypes, scale values, and device layout.

## 5. Run synthetic MXFP4 component checks

These checks use generated weights and can run before the real checkpoint is
available:

```bash
uv run --no-sync xtest run \
  tests/suites/integration/runtime/ffnagent/test_mxfp4.py \
  tests/suites/integration/runtime/ffnagent/models/gpt_oss/test_router.py \
  --strict-requirements
```

They cover TP1/TP2 loading and computation, independent numerical decoding,
bias placement, invalid tails, allocation-free graph replay, native repeated
layer rebinding with separate Lane resources, and the original isolated
SGLang layer reference with current, legacy, and absent dtype metadata. Expect
nine passing tests, including 32- and 128-Expert Router rounding over every
power-of-two row capacity from 1 through 16384. The Router owns an explicit
1-MiB cuBLASLt scratch region within its accounted per-Lane workspace;
ordinary Torch GEMMs retain the FfnAgent zero-workspace policy. Reference
construction uses SGLang's resolved runtime dtype
and its loader's scoped default-dtype context; checkpoint metadata can omit the
dtype fields.

## 6. Compare real-checkpoint FFN results

```bash
uv run --no-sync xtest list --suite openai/gpt-oss-20b
uv run --no-sync xtest run --suite openai/gpt-oss-20b \
  -k test_ffn_numerical --strict-requirements
```

The inventory contains five cases in total. The numerical case compares layers
0 and 23 at 1, 32, and 512 rows with the original pinned SGLang FFN. Its
reference expands one layer at a time in the isolated reference process;
production FFN weights retain their packed representation.

Continue only after the numerical comparison passes. Preserve its artifacts
if there is a routing, activation, bias, or output mismatch.

## 7. Qualify serving graph modes

```bash
uv run --no-sync xtest run --suite openai/gpt-oss-20b \
  -k test_serving_graph --strict-requirements
```

Keep all three selected rows together: Eager, Decode Full, and Decode Full plus
Prefill Breakable. The runner checks Decode token agreement, first-prefill
logit KL, and observed graph execution. A successful HTTP response alone does
not satisfy the graph verdict.

For GPT-OSS, the installed launcher supplies Triton attention,
`--disable-hybrid-swa-memory`, `--context-length 8192`, and
`--max-running-requests 4`. Sliding-window masking remains active while the
existing Elastic KV storage pool is reused. This qualification does not cover
the model's full advertised context length or production throughput.

## 8. Cross the sliding-window boundary and reuse the prefix

```bash
uv run --no-sync xtest run --suite openai/gpt-oss-20b \
  -k test_sliding_window_and_prefix_reuse --strict-requirements
```

This sends the same 4097-token prefix twice, generates two tokens per request,
requires a positive cached-token count on the second request, and checks
combined Decode Full and Prefill Breakable graph evidence.

After diagnosing any individual failures, run all five model cases together
for the retained model-suite verdict:

```bash
uv run --no-sync xtest run --suite openai/gpt-oss-20b --strict-requirements
```

## 9. Retain the results

```bash
uv run --no-sync xtest report --list
# Replace RUN_ID with the exact ID printed for the desired run.
uv run --no-sync xtest report RUN_ID
```

Raw evidence is under `.xpool-cache/test-runs/<RUN_ID>/`. Keep `run.json`,
`results.json`, the failing task's `pytest.log` and `pytest.xml`, and its
daemon/server logs. Serving attempts also retain inference requests/responses
and graph/logit evidence. For an OOM, retain `nvidia-smi` output and startup
logs; six-device capacity, especially 120b capacity, is subject to measured
placement and memory demand.

An observed existing MPS endpoint directory is a startup failure. Preserve the
exact traceback and check for another live run; use the owning process's
normal shutdown rather than deleting a potentially live MPS directory.

## 10. Repeat for 120b and finish repository acceptance

The 120b checkpoint is at `/home/LAB/yezr/models/openai/gpt-oss-120b/`. Use a
separate local config with the same topology and replace both occurrences of
`openai/gpt-oss-20b` with `openai/gpt-oss-120b`. Export that file as
`XPOOL_CONFIG`, repeat the metadata check, and run:

```bash
uv run --no-sync xtest run --suite openai/gpt-oss-120b --strict-requirements
```

The 120b numerical case selects layers 0 and 35. Its checkpoint and resource
evidence are independent of the 20b result. Preserve its complete model-suite
verdict before claiming that the six-device topology can serve it.

The accepted plan additionally requires model-neutral memory calibration,
real-model baseline/pressure qualification, the complete strict repository
suite, and documentation alignment before claiming complete support. The
commands above provide numerical and serving qualification; they do not
replace those resource gates, whose passing evidence is retained in the
[qualification report](qualification.md). For the complete ordinary suite,
restore a local configuration containing its required existing Qwen/DeepSeek
checkpoints before running `uv run --no-sync xtest run --strict-requirements`.
The single-model GPT-OSS configuration cannot supply those unrelated fixtures.

This checkout also has `configs/gpt-oss-qualification.local.toml` listing both
checkpoints. To repeat their complete suites sequentially in one invocation:

```bash
export XPOOL_CONFIG="$PWD/configs/gpt-oss-qualification.local.toml"
uv run --no-sync xtest run \
  --suite openai/gpt-oss-120b --suite openai/gpt-oss-20b \
  --strict-requirements
```

To reproduce the complete ordinary repository suite using its prepared local
configuration:

```bash
export XPOOL_CONFIG="$PWD/configs/xpool.local.toml"
uv run --no-sync xtest run --strict-requirements
```

Keep `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6` exported for both invocations.
