# xpool Project Plan

This document is the canonical design for the xpool rebuild. It is
self-contained and replaces chat history or the v2 worktree as the source of
truth for new implementation work.

## Mission

xpool implements intra-node colocated serving with the smallest practical
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
   xpool SGLang plugin from `xpool.integrations.sglang`. The plugin is
   model-neutral: public adapter contracts live in
   `xpool.integrations.sglang.adapter`, automatic adapter discovery lives in
   `xpool.integrations.sglang.registry`, and model-specific implementations
   live under `xpool.integrations.sglang.models`. The plugin loads the registry,
   registers adapter hooks, and runs plugin-level lifecycle checks around
   `ModelRunner.load_model`. Model-specific hook targets, construction
   compatibility, weight filtering, and post-load invariants live inside the
   owning adapter. For the first DeepSeek adapter, SGLang's original
   `ForCausalLM`, model body, decoder layer, attention, communicator, logits,
   and attention weight-loading logic remain the program skeleton.

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
  SGLang plugin through a model adapter, replaces target FFN/MLP/MoE classes
  during model construction, and calls a CUDA extension op. The frontend writes
  a fixed-address request descriptor into IPC-mapped device memory and waits for
  completion/result state. The DeepSeek adapter's FFN shim classes inherit the
  original SGLang FFN classes for `isinstance` compatibility but do not call the
  original FFN constructors, so dense FFN weights, MoE experts, and shared
  experts are never allocated in the attention-side SGLang process.
- **Shim device agent** lives in the attention-side device agent. It is a
  persistent kernel that polls per-instance queues, arbitrates resources,
  advances attention-to-FFN transport, and writes grants/results.

Both eager execution and CUDA graph capture/replay use the same device ABI.
Captured execution must not allocate memory, call host control-plane APIs,
perform Python control flow, or change descriptor addresses.

CUDA graph shape compatibility is derived from SGLang's actual graph capture
configuration and shim descriptors. xpool does not maintain an independent
decode-bucket list in repository config.

The first shim contract is deliberately narrow. A shim call represents one
decoder-layer FFN over a contiguous two-dimensional CUDA tensor
`[num_tokens, hidden_size]` and returns an out-of-place tensor with the same
shape and dtype. The first supported forward modes are normal SGLang decode and
extend/prefill. The SGLang plugin owns a model-neutral server-argument support
matrix and fails closed for SGLang runtime modes that are not yet represented in
the xpool shim ABI, including speculative execution, pipeline parallelism,
LoRA, quantized first-stage loading, expert parallelism, EPLB, DeepEP, expert
distribution recording, two-batch overlap, context parallel prefill, all-reduce
fusion, SGLang CPU/layer offload, hierarchical cache offload, decode KV offload,
mixed chunked prefill, PD disaggregation, diffusion LLM inference,
PD multiplexing, and reduce-scatter FFN output. Ordinary continuous batching
and ordinary chunked prefill remain allowed when SGLang presents shim calls as
exact `DECODE` or exact `EXTEND`; xpool rejects modes that can surface composite
forward modes such as `MIXED`, `SPLIT_PREFILL`, `DLLM_EXTEND`, or `PREBUILT`.
Model adapters may add model-construction checks, but they do not own global
SGLang runtime policy. Server-argument rules remain part of the SGLang plugin
layer because they describe global plugin ABI support, not model architecture.
These rules run on SGLang's resolved `ServerArgs`; they are not a replacement
for SGLang's own CLI parsing, defaults, and server-argument validation.

The SGLang shim frontend uses a single concrete `FfnShimModule` base class,
not a mixin/protocol pair. Model-specific classes inherit
`FfnShimModule, OriginalSGLangClass`, and `FfnShimModule.__init__` directly
initializes `nn.Module` to avoid running original FFN constructors. Shim layer
kinds use symmetric names: `DENSE` and `SPARSE`.

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

Until that audit is complete, xpool must not claim cross-model KV sharing.

## Public Commands

The package exposes one `xpool <subcommand>` CLI plus an SGLang general plugin.
SGLang 0.5.13 loads general plugins from the `sglang.srt.plugins` entry point
group in `sglang serve` and `python -m sglang.launch_server`; `SGLANG_PLUGINS`
is SGLang's comma-separated plugin whitelist.

The xpool plugin entry point is
`xpool.integrations.sglang.plugin:install`. It does not patch generic FFN
`forward()` functions and does not redirect SGLang `ModelRegistry` entries. The
plugin loads a model adapter registry, installs adapter hooks, and installs a
generic `ModelRunner.load_model` lifecycle hook. The registry auto-discovers all
zero-argument concrete `SglangModelAdapter` subclasses owned by modules under
`xpool.integrations.sglang.models`; new model support is added by adding a model
module, not by editing a global adapter list. The lifecycle hook validates
model-neutral SGLang server arguments once, binds the matching SGLang model path
to exactly one `models[]` entry from `XPOOL_CONFIG`, then lets the owning
adapter run model-specific checks. There is no `XPOOL_INSTANCE_ID`; the
instance id is derived from the one-model-one-SGLang-instance mapping in TOML.

The first DeepSeek adapter directly replaces SGLang's DeepSeek FFN classes with
xpool shim classes and hooks DeepSeek weight loading to skip
`model.layers.*.mlp.*` tensors. It does not expose a module-level adapter
singleton; the registry instantiates `DeepseekV2Adapter` through auto-discovery.
It does not support DeepSeek-V3/V3.2 until those architectures have explicit
adapter coverage and validation.

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
- target model id and absolute model path.

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

## Code Quality

Python checks use Ruff formatting, Ruff linting, Astral ty, and pytest through
the installed pre-commit hooks. Ruff enables `ANN401` so explicit `Any` in
function arguments is rejected. Ruff also enforces public Python docstrings for
classes, functions, methods, constructors, and documented Google-style
parameters. Field-level public API documentation is guarded by tests for
Pydantic fields, dataclass fields, enum members, and native ABI fields. Broad
`object` annotations are not banned with a custom AST test. The SGLang
integration layer is allowed and expected to import SGLang concrete types
directly; use local runtime guards or casts only around SGLang attributes that
are assigned dynamically and are not visible to the type checker.

C++ and CUDA public APIs use Doxygen comments. The root `Doxyfile` is a
warning-as-error gate for native headers and sources under `src/cext`, including
`.cu` and `.cuh` files mapped as C++ for documentation parsing. `clang-format`
remains the native formatter; Doxygen is the documentation completeness check.

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
```

Model entries do not carry tensor-parallel placement. The current topology uses
the full role-local device pools for every configured model: SGLang attention TP
is derived from `len(devices.attention_cuda_devices)`, and FFN TP is derived
from `len(devices.ffn_cuda_devices)`. Multiple configured models share the same
FFN device-agent pool; the runtime scheduler arbitrates ownership. If future
work needs model-specific FFN subsets, that should be introduced as an explicit
placement policy rather than a scalar `models[].tp` knob.

SGLang's `--tp-size` is treated as the attention world size. SGLang's
`ModelConfig.attention_arch` is the authoritative MLA/non-MLA boundary; xpool
does not infer MLA from model names or loose fields such as `kv_lora_rank`.
When SGLang reports MLA, xpool validates the positive MLA dimensions it needs
for compressed-cache layout and uses `attention_tp = 1`. When SGLang reports a
regular attention layout, xpool derives MHA/GQA/MQA from SGLang's total
attention heads and KV heads, then uses
`attention_tp = min(num_key_value_heads, attention_device_count)`. The first
shim ABI does not support SGLang DP attention, so xpool requires
`attention_device_count == attention_tp` and SGLang `--dp-size = 1`. Future
DP-aware shim support must update the config policy, SGLang server-argument
gate, and native descriptor semantics together.

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
- input/output/scratch arena offsets,
- token count,
- hidden size.

FFN result descriptors also contain:

- error code,
- output arena offset.

Descriptors must not carry raw CUDA pointers across processes. SGLang and the
local device agent may map the same CUDA IPC allocation at different virtual
addresses, so all cross-process references use arena offsets. Each lane owns a
device-resident sequence counter. The shim publish kernel increments that
counter at eager/runtime execution, including CUDA graph replay, and wait
kernels match both `result.sequence` and `DONE` status before copying output
back to the PyTorch return tensor.

Current Python tests check descriptor packing and native byte-size parity before
native runtime work is extended. Explicit enum, field-offset, and alignment
parity tests must be added before xpool relies on those properties as stable ABI
guarantees.

## Implementation Order

1. Land this plan and the repository skeleton.
2. Add Python package, `xpool <subcommand>` CLI, config validation, and daemon API
   skeleton.
3. Add device-agent launcher and MPS/runtime preflight checks.
4. Add model-neutral SGLang plugin registration and DeepSeek FFN class adapter.
5. Add shared ABI headers and Python packing tests.
6. Prove single-GPU eager shim publish/wait with a test-only loopback executor.
7. Prove CUDA graph capture/replay over the same shim ABI.
8. Prove one NVSHMEM rank per device agent transport with device-side progress.
9. Attach DeepSeek-V2-Lite FFN executor and correctness oracle tests.
10. Run SGLang E2E with multiple instances on the same attention GPU.
11. Start phase-2 KV sharing design and implementation.

The loopback executor in step 6 is only an ABI and CUDA graph validation tool.
It must not be used as serving evidence; SGLang E2E readiness requires the real
DeepSeek-V2-Lite FFN executor.

## Validation

Repository validation uses:

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
clang-format --dry-run --Werror src/cext/include/abi.hpp
doxygen Doxyfile
```

C++ and CUDA code is managed by `CMakeLists.txt` and formatted with
`clang-format`.

Performance work must use Nsight Systems before hot-path optimization. TBT and
TPOT must be measured at generated-token boundaries, not derived from
end-to-end request latency.
