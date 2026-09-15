# System Overview

CrossPool separates attention and KV-cache execution from FFN weight residency and
computation. The current deployment runs both roles on one host under CUDA
MPS. Multiple models may share every GPU assigned to a role; no model owns a
device exclusively.

This document defines the supported deployment boundary, process roles, and
cross-module interface ownership. See the [design map](README.md) for the
remaining current-state documents.

## Supported boundary

### Deployment assumptions

The supported production boundary is:

- one Linux host with NVIDIA GPUs capable of the selected execution;
- CUDA Toolkit 13.2 and CCCL 3.2;
- CUDA MPS running before GPU participants start;
- CUDA IPC mappings for rank-local Transport and NVSHMEM communication for
  Fabric, including directly accessible peer memory for FFN partial reduction;
- one identical resolved CrossPool configuration in every process;
- SGLang as the only serving-engine integration;
- BF16 or FP16 hidden-state payloads, with the current qualification models
  retaining BF16; and
- gated Dense and MoE decoder FFN layers.

The four qualification model families are dense Qwen3, Qwen3-MoE,
DeepSeek-V2-Lite, and GLM-4.7-Flash. Model adapters may support additional
compatible architectures, but readiness claims require direct evidence for
each new family.

GPU model names and interconnect labels do not define a hardware allowlist.
An untested topology is not excluded solely for lack of qualification, nor is
it guaranteed to work: the selected kernels, CUDA Graph features, and memory
access paths must be available on the deployment. Environment compatibility
checks for optional memory calibration restrict reuse of that profile, not
deployment to a fixed GPU model; see [Control Plane](control-plane.md#memory-admission).

The current implementation excludes cross-host IB, expert parallelism, Unary FFN,
same-Instance concurrent FFN requests, service-time placement changes, model
migration, rolling updates, mixed-ABI generations, and request-time fallback.
Except for the explicitly reserved `scheduler.atn_concurrency` setting described
in [Control Plane](control-plane.md), no dormant types or configuration switches
are retained for these features.

### Process roles

The daemon owns configuration-wide coordination. It registers processes,
builds one immutable Fabric generation, assigns Transport publications,
computes FFN placement, tracks readiness, observes heartbeats, and selects
fail-stop shutdown. It loads the native extension as the daemon role but does
not initialize CUDA or join NVSHMEM. Fabric UID creation is its only
business-level native operation.

Each serving Instance rank runs inside SGLang. It owns request scheduling,
attention, logical KV-cache and prefix-cache semantics, outer CUDA Graph
selection, output processing, and one rank-local Transport attachment.
CrossPool supplies stable elastic physical backing and coordinates capacity
across co-located Instances as described in
[Elastic KV-cache Pooling](elastic-kv-cache.md). The Instance invokes the sole
tensor API, `xpool.ops.ffn_shim`.

Each AtnAgent owns one configured CUDA device, rank-local CUDA IPC Transport
arenas, and one Fabric PE. It stages Instance input into Fabric, submits layer
work, receives the required FFN output form, and returns the result through the
Transport mailbox.

Each FfnAgent owns one configured CUDA device and one Fabric PE. It retains the
planned true-TP weight shards for all assigned model layers, materialized
operator resources, and one independently instantiated Lane GraphExec per
executor lane. The first FfnAgent PE also launches the Fabric Coordinator.

The immutable PE order is all AtnAgent PEs followed by all FfnAgent PEs. The
coordinator resides on the first FfnAgent PE. A failed participant cannot
recover into the retained generation.

## Public interfaces and ownership

The sole Tensor dispatcher operation is `xpool.ops.ffn_shim`. Its caller owns
output storage, so eager execution and enclosing CUDA Graph replay use the same
lifetime. Control-plane and resource lifecycle operations are typed pybind
functions rather than Torch operators.

The native binding tree follows ownership. The module root contains only ABI
identity, runtime role, and initialization. Shared FFN semantics live under
`xpool.native.ffn`; KV capacity channels live under `xpool.native.kv`; Debug
options, Transport, Fabric, FfnAgent, and each Devkit Observer own their values
and lifecycle in their corresponding submodules. Values are not re-exported at
the root, and generic `types`, `common`, or `protocol` buckets are not part of
the interface.

The C++ namespace tree mirrors domain ownership without introducing an
`xpool::native` namespace. ABI identity belongs to `xpool::abi`, shared FFN
semantics to `xpool::ffn`, concrete lifecycle to `xpool::transport`,
`xpool::fabric`, `xpool::ffnagent`, and `xpool::kv`, Observer evidence to
`xpool::devkit::<observer>`, and typed Hook Points to `xpool::hooks`.
Generic Host-side CUDA Graph mechanisms belong to `xpool::utils::graph`.

The daemon installs the same immutable Host Debug options as participants but
creates no CUDA Device mirror and joins no NVSHMEM PE. Participant lifecycle
functions reject the daemon role before initializing CUDA or touching Device
resources.

Native `RuntimeState` is the sole process-lifetime authority for the initialized
role. Python bootstrap queries that authority through the root binding and does
not retain a second role or CUDA-device identity.

Native validation follows the ownership seam. Aggregate Projection construction
validates immutable geometry, ordering, topology, referenced indices, and
resource relationships once. Runtime installation validates live CUDA
allocations, Graph handles, IPC mappings, NVSHMEM results, and
current-device accessibility. Public lifecycle methods reject invalid phase
transitions. Private helpers and owner-created views rely on established
invariants, while dynamic mailbox, publication, sequence, PE, and
external-runtime facts remain checked where they cross concurrency or trust
seams.

Fabric and Transport arena storage is C++/CUDA-owned. Python provides semantic
plan projections rather than byte offsets or duplicate layout constants.
Production allocation-size queries reuse the same native layout arithmetic as
allocation. Observer sizing reads immutable Host Debug options and reuses its
actual allocation geometry.

Python-to-native Projection values are construction-only and opaque after
validation. Tensor-bearing bindings accept owning Torch Tensors and extract
their Device addresses at the native trust seam; Python runtime owners retain
the Tensor lifetime. Opaque handle encodings derive their public character
lengths from the owning native value types rather than duplicate Python
constants.

The native ABI uses a stopped-world compatibility version. Shared closed-set
protocol concepts are fixed-underlying-type scoped enums in C++ and generated
`IntEnum` bindings in Python. Payload element types are `torch.dtype` in Python
and `c10::ScalarType` in native code. CUDA system-scope atomics operate directly
on trivially copyable fields through CCCL `cuda::atomic_ref` with acquire,
release, or acquire-release ordering appropriate to the publication. Generated
stubs are the only maintained Python typing source for the native interface.

Supported Python declarations are identified by explicit `__all__` exports and
documented at their owning declarations. Public Native declarations are
documented in `src/cext-include/xpool`; implementation helpers and tests carry
comments only for non-inferable ownership, lifecycle, synchronization, Graph,
numerical, or acceptance contracts. Ruff owns Python docstring shape, Doxygen
owns Native declaration coverage and markup, and clang-format owns Native
include grouping and comment layout.

The current implementation contains real Dense and MoE FFN execution,
true-TP weight ownership, Device-resident Transport and Fabric progress,
per-Lane GraphExec ownership, static placement, memory admission, elastic
attention-side KV backing, and SGLang serving integration. It contains no
alternate debug execution, request-time fallback, compatibility alias, or
mixed-ABI path.
