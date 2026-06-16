# xpool v3 Project Plan

This document is the canonical design for the xpool v3 rebuild. It is
self-contained and replaces chat history or the v2 worktree as the source of
truth for new implementation work.

## Mission

xpool v3 implements intra-node colocated serving with the smallest practical
intrusion into SGLang. SGLang remains the attention-side serving program and
request scheduler. xpool replaces FFN calls with a graph-safe shim, arbitrates
shared GPU and communication-slot resources across multiple SGLang instances,
and owns the FFN transport/execution path.

The first production target is `deepseek-ai/DeepSeek-V2-Lite-Chat`. The project
is SGLang-only.

## Non-Goals

- Do not restore the v2 source architecture wholesale.
- Do not use SGLang to execute FFN work.
- Do not implement expert parallelism in the first design; FFN execution uses
  tensor parallelism.
- Do not make KV cache sharing part of the first runnable FFN-shim closure.
- Do not add Python NVSHMEM bindings unless a later accepted design requires
  Python-side NVSHMEM calls.

## Runtime Model

xpool has three runtime placements:

1. `xpool daemon` is a single global host control plane. It owns registration,
   policy configuration, readiness, health reporting, and global state for
   attention-side SGLang instance arbitration. It must not participate in
   request-time FFN progress or captured CUDA graph execution.
2. `xpool device-agent` is launched once per participating GPU. Each device
   agent owns one NVSHMEM rank, symmetric memory, CUDA IPC surfaces, and the
   resident persistent kernels on that GPU.
3. SGLang instances are normal SGLang server processes. Each instance loads the
   xpool SGLang plugin and receives monkeypatched FFN/MLP/MoE modules.

CUDA MPS is a required runtime precondition for the target deployment, because
one GPU may host multiple SGLang processes plus an xpool device agent with
resident device-side work. This is not a configurable runtime mode.
The daemon performs an MPS control-daemon preflight at startup and refuses to
serve when MPS is unreachable. It also runs a periodic MPS health monitor and
surfaces the cached state through `/health`; device agents repeat the same
preflight before resident GPU work.

## Shim Component

The shim component is not a single Python object. It is a coordinated ABI and
runtime path with two placements:

- **Shim frontend** lives inside each SGLang instance. It is installed by the
  SGLang plugin, replaces target FFN/MLP/MoE calls, and calls a CUDA extension
  op. The frontend writes a fixed-address request descriptor into IPC-mapped
  device memory and waits for completion/result state.
- **Shim device agent** lives in the attention-side device agent. It is a
  persistent kernel that polls per-instance queues, arbitrates resources,
  advances attention-to-FFN transport, and writes grants/results.

Both eager execution and CUDA graph capture/replay use the same device ABI.
Captured execution must not allocate memory, call host control-plane APIs,
perform Python control flow, or change descriptor addresses.

CUDA graph shape compatibility is derived from SGLang's actual graph capture
configuration and shim descriptors. xpool does not maintain an independent
decode-bucket list in repository config.

## Resource Policy

The daemon configures policy, but the attention-side shim device agent performs
the request-time grant on device.

The initial policy is intentionally conservative:

- For each attention GPU, only one SGLang instance may execute attention at a
  time.
- For each attention GPU, only one SGLang instance may own the communication
  slot at a time.
- Communication slots are explicit resources used to bound parallelism and
  enable pipeline scheduling.

## FFN Execution

FFN-side device agents own xpool FFN execution. The planned execution stack is:

- persistent-kernel scheduler,
- captured graph execution and lowering,
- declared host fallback for unsupported shapes,
- FlashInfer-backed kernels where available,
- tensor parallelism across participating FFN device agents.

Correctness is first proven against layer-level oracles before serving metrics
are claimed.

## NVSHMEM Transport

Every participating GPU owns one NVSHMEM rank through its device agent. The
daemon does not own an NVSHMEM rank. SGLang instances do not initialize NVSHMEM.

The transport uses NVSHMEM symmetric memory and device-side signal/put/get or
collective primitives between device agents. SGLang shims communicate with their
local device agent through CUDA IPC-mapped queues and descriptors.

## KV Sharing

KV cache sharing is phase 2. The phase-2 design must audit:

- SGLang 0.5.13 KV allocator and memory-pool internals,
- kvcached's SGLang autopatch and virtual memory allocator pattern,
- compatibility with xpool's per-device-agent worker ownership model.

Until that audit is complete, xpool v3 must not claim cross-model KV sharing.

## Public Commands

The package exposes one `xpool <subcommand>` CLI plus an SGLang general plugin.
SGLang 0.5.13 loads general plugins from the `sglang.srt.plugins` entry point
group in `sglang serve` and `python -m sglang.launch_server`; `SGLANG_PLUGINS`
is SGLang's comma-separated plugin whitelist.

```bash
uv run xpool daemon --config configs/xpool.example.toml --check
uv run xpool device-agent --config configs/xpool.example.toml --check
SGLANG_PLUGINS=xpool XPOOL_CONFIG=configs/xpool.example.toml uv run sglang serve ...
```

The `--check` mode validates configuration without starting resident GPU work.
Daemon and device-agent checks also report current MPS preflight state.

For repeated local development commands, copy `.env.example` to an untracked
`.env`, edit local values, and set `UV_ENV_FILE=/path/to/.env` before invoking
`uv run`.

## Configuration

The repository configuration is TOML. It defines:

- daemon bind address,
- scheduler concurrency limits,
- attention-side CUDA devices,
- FFN-side CUDA devices,
- target model id, absolute model path, and FFN tensor-parallel degree.

It does not configure device agents, NVSHMEM ranks, SGLang instances, model
family, hidden size, or attention topology directly. These are derived:

- one xpool device agent per configured CUDA device,
- one role per CUDA device,
- NVSHMEM ranks assigned in attention-device order followed by FFN-device order,
- one SGLang instance per configured model,
- model family, hidden size, KV heads, and attention layout through SGLang's
  model config resolution.

Runtime policy must be derived from configuration and model metadata, not from
hard-coded planner decisions.

The scheduler section uses concurrency terms, not slot-per-device terms:

```toml
[scheduler]
attention_concurrency = 1
transport_concurrency = 1
```

The device section is the source of truth for agent placement:

```toml
[devices]
attention_cuda_devices = [0]
ffn_cuda_devices = [1]
```

The two device lists must be non-empty, unique, and disjoint. A CUDA device may
host only one xpool role.

Each model entry uses a full model instance id and an absolute local path:

```toml
[[models]]
id = "deepseek-v2-lite-chat"
path = "/absolute/path/to/deepseek-ai/DeepSeek-V2-Lite-Chat"
tp = 1
```

`models[].tp` is the FFN tensor-parallel degree. It may be less than or equal to
the number of FFN device agents; the first implementation selects the first
`tp` FFN agents in configured order.

SGLang's `--tp-size` is treated as the attention world size. SGLang's
`ModelConfig.attention_arch` is the authoritative MLA/non-MLA boundary; xpool
does not infer MLA from model names or loose fields such as `kv_lora_rank`.
When SGLang reports MLA, xpool validates the positive MLA dimensions it needs
for compressed-cache layout and uses `attention_tp = 1`. When SGLang reports a
regular attention layout, xpool derives MHA/GQA/MQA from SGLang's total
attention heads and KV heads, then uses
`attention_tp = min(num_key_value_heads, attention_device_count)`. In both
cases `attention_dp = attention_device_count / attention_tp`, and SGLang
`--dp-size` matches the derived attention DP size when it is greater than one.

All TOML fields and CLI overrides for registered settings are declared in the
config registry and resolve in this order:

1. CLI arguments,
2. TOML config,
3. registry defaults.

Settings without a default must fail fast when none of their allowed sources
provides a value. xpool deployment policy must not be controlled by environment
variables. `XPOOL_CONFIG` is a bootstrap path used before TOML can be loaded,
not a registered config field. `SGLANG_PLUGINS` belongs to SGLang's plugin
loader and is documented in `.env.example`, not in xpool's config registry.
Loading the xpool SGLang plugin enables the shim; there is no separate xpool
enable flag or manually configured instance id. The instance id is derived from
the one-model-one-SGLang-instance mapping.

## ABI Surfaces

The shared Python/C++ ABI starts with versioned FFN request/result descriptors.
Both descriptors contain:

- ABI version and byte size,
- sequence number,
- status/error state,
- output slot id.

FFN request descriptors also contain:

- instance id,
- model id,
- layer id,
- forward mode,
- dtype,
- input/output/scratch device pointers,
- token count,
- hidden size.

FFN result descriptors also contain:

- error code,
- output device pointer.

Current Python tests check descriptor packing and native byte-size parity before
native runtime work is extended. Explicit enum, field-offset, and alignment
parity tests must be added before xpool relies on those properties as stable ABI
guarantees.

## Implementation Order

1. Land this plan and the repository skeleton.
2. Add Python package, `xpool <subcommand>` CLI, config validation, and daemon API
   skeleton.
3. Add device-agent launcher and MPS/runtime preflight checks.
4. Add SGLang plugin registration and FFN patch targets.
5. Add shared ABI headers and Python packing tests.
6. Prove single-GPU eager shim publish/wait with fake FFN results.
7. Prove CUDA graph capture/replay over the same shim ABI.
8. Prove one NVSHMEM rank per device agent transport with device-side progress.
9. Attach DeepSeek-V2-Lite FFN executor and correctness oracle tests.
10. Run SGLang E2E with multiple instances on the same attention GPU.
11. Start phase-2 KV sharing design and implementation.

## Validation

Repository validation uses:

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

C++ and CUDA code is managed by `CMakeLists.txt` and formatted with
`clang-format`.

Performance work must use Nsight Systems before hot-path optimization. TBT and
TPOT must be measured at generated-token boundaries, not derived from
end-to-end request latency.
