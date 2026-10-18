# GPT-OSS A100 qualification evidence

This report records the retained October 17, 2026 qualification of the
resident MXFP4 implementation. Both model suites, the real-model baseline
and pressure worlds, and the complete strict repository suite have passed.
Current designs and supported-model documentation reflect the accepted behavior.

## Environment and execution contract

- Physical devices: `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`; device 0 is excluded.
- Six NVIDIA A100-SXM4-40GB devices, SM80. Attention uses the first two
  visible devices with TP2; four direct FfnAgents use the remaining devices,
  FFN TP1 and one Executor Lane.
- Checkpoints: `/home/LAB/yezr/models/openai/gpt-oss-{20b,120b}`. Source
  configuration and index hashes, shard sizes and resolved paths are retained
  in `.xpool-cache/qualification/gpt-oss-20261017/checkpoints.json`.
- Native ABI 86; Torch 2.13.0, Triton 3.7.1, SGLang 0.5.20,
  sglang-kernel 0.4.7, nvidia-cublas 13.1.1.3, NVSHMEM 3.7.2,
  CUDA driver/runtime API version 13030.
- Build toolchain: managed Python 3.12.14, uv 0.12.17, NVCC 13.3.73,
  GCC 11.4.0, CMake 4.3.2 and Ninja 1.13.0. Exact command outputs are
  retained in `gpt-oss-20261017-router/toolchain.json`.
- MXFP4 values and E8M0 scales remain packed in device memory. SM80 Expert
  execution decodes tiles in the kernel and uses BF16 Tensor Core arithmetic
  with FP32 accumulation. Payloads and Expert biases are BF16.
- The biased BF16 Router uses caller-owned, aligned 1-MiB cuBLASLt scratch
  and preserves original SGLang Linear rounding. Ordinary Torch GEMMs retain
  the FfnAgent zero-workspace policy. The native library resolves cuBLASLt
  from Torch's pinned wheel even when native code is imported first.
- Installed serving uses Triton attention, an 8192-token context,
  `max_running_requests=4`, and ordinary Elastic KV storage with
  `disable_hybrid_swa_memory=true`. Sliding-window masks remain active.

Each model and resource world selects one checkpoint. These verdicts qualify
20b and 120b separately; simultaneous serving of both GPT-OSS models is outside
this test matrix.

The native binary SHA-256 is
`d0d4c5fffd49fbecc09f397611d7d83b962c5c2991addb315d6f747e16174aec`.
The source manifest is
`.xpool-cache/qualification/gpt-oss-20261017-router/source-identity.json`.
The working tree is based on `ff9a546d27cc5cfad5c63978cd096a792dc1e65f`;
the manifest, rather than that base commit alone, identifies the tested code.

## Checkpoint provenance

The retained Hugging Face download metadata records one fixed revision for
every file of each checkpoint:

| Model | Original repository revision | Safetensors shards | Total shard bytes |
| --- | --- | --- | --- |
| 20b | [`6cee5e81ee83917806bbde320786a8fb61efebee`](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | 3 | 13,761,316,904 |
| 120b | [`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a) | 15 | 65,248,893,184 |

An October 18 provenance audit matched every shard's recorded download
SHA-256 with the official repository's LFS metadata at that fixed revision.
It also matched both local configuration/index files with their recorded Git
blob IDs and the SHA-256 values captured before qualification. This establishes
the retained download provenance; the audit does not claim a fresh full-file
hash of all weight shards. Production consumes these original MXFP4 files
directly, with no converted checkpoint.

| Model | `config.json` SHA-256 | `model.safetensors.index.json` SHA-256 |
| --- | --- | --- |
| 20b | `3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce` | `0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1` |
| 120b | `933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566` | `ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e` |

Both source configs omit `dtype` and `torch_dtype`. The admitted effective
dtype is explicitly BF16; Expert values/scales retain their source byte
representation. Serving qualifies attention TP2/DP1 and FFN TP1; the shared
synthetic native regression separately covers FFN TP2 scale and bias ownership.

## Retained verdicts

| Evidence | Verdict | Run or artifact |
| --- | --- | --- |
| Canonical native build and generated stubs | Passed | `gpt-oss-20261017/native-build-router-wheel-headers.log` |
| MXFP4 component checks | 9 passed, 0 skipped | `20261017-095900-2622277-3258703630095564` |
| Model-neutral memory calibration | 21 worlds, 420 observations; passed | `gpt-oss-20261017-router/calibration.log` and `calibration-*.json` |
| 120b focused real-checkpoint FFN comparison | 6/6 samples passed | `20261017-100841-2650549-3259284428134865` |
| Complete 20b and 120b model suites | 10 passed, 0 skipped; both graph groups compared | `20261017-101056-2654722-3259419502469511` |
| Real-model baseline and pressure qualification | Four worlds, 80 observations; passed | `gpt-oss-20261017-router/resource-{20b,120b}-{baseline,pressure}/` |
| Complete strict repository suite | 1,373 passed, 0 skipped, 0 failed | `20261017-103959-2707808-3261161750575579` |

Run IDs resolve beneath `.xpool-cache/test-runs/`; other artifacts above
resolve beneath `.xpool-cache/qualification/`. Each installed test run retains
its original verdict, cleanup evidence, logs and numerical/graph observers.
The [manual guide](testing.md) gives reproduction commands and configuration.

The strict suite contains 83 native CTest cases, 684 unit cases, 594
integration cases and 12 end-to-end cases. It retains result code zero,
verified cleanup, a finished verdict and no infrastructure error. Both
concrete-model suites also retain finished verdicts and verified cleanup.
The ordinary suite covers transport, graph installation/rebinding, existing
FFN families, MPS lifecycle, serving TP/DP, Elastic KV pressure and owned
benchmark execution alongside the new MXFP4 regressions.

Static checks passed: Ruff formatting on 423 files, Ruff lint, ty,
clang-format on 131 native files, Doxygen and `git diff --check`.
The source manifest and installed native SHA were checked again after the
strict suite. No executable source or build dependency changed after these
accepted runs; final alignment changed documentation only.

## Issue acceptance mapping

The technical scope in [family issue #35](https://github.com/Coekjan/CrossPool/issues/35)
and its [20b](https://github.com/Coekjan/CrossPool/issues/36) and
[120b](https://github.com/Coekjan/CrossPool/issues/37) model tasks is covered as
follows. Upstream review and issue closure remain separate from these verdicts.

| Required evidence | Retained implementation and verdict |
| --- | --- |
| Exact activation, biases, routing and MXFP4 ingestion | Typed checkpoint/activation contracts, packed residency inventories, component regressions and independent real-layer reference for each checkpoint |
| Sliding/global attention state and graph execution | Ordinary Elastic KV with preserved masks/sinks; three serving modes and repeated 4097-token prefix for each model |
| Decode token identity | Eager and Decode Full produce identical eight-token sequences for each model |
| Prefill full-distribution KL and actual Breakable execution | Matching `[1, 201088]` logits, KL below 0.05 for both models and Graph Observer confirmation |
| Checkpoint, dtype, topology and hardware provenance | Fixed revisions and hashes above; explicit BF16/SM80 and attention TP2/FFN TP1 boundary |
| Frozen runtime/build acceptance and invalidated resource evidence | ABI-86 build, matching source/native hashes, 1,373 strict-suite passes, 21 calibration worlds and four real-model resource worlds |
| Installed serving and descendant cleanup | Both model suites finish with result code zero and verified cleanup through the existing harness |
| Existing deployment boundary and scoped support claims | SGLang/CUDA integration, existing attention MPS lifecycle, direct FFN execution and concrete-model catalogue entries limited to the qualified scope |

## Numerical results

The 120b comparison uses original pinned SGLang to load and expand one
reference layer at a time, independently of the production loader and decoder.
All selected Expert IDs agree. Thresholds remain the existing numerical policy.

| Layer | Rows | Output normalized RMSE | Maximum routing weight error |
| --- | --- | --- | --- |
| 0 | 1 | 0 | 2.98e-8 |
| 0 | 32 | 7.91e-5 | 5.96e-8 |
| 0 | 512 | 1.72e-5 | 1.19e-7 |
| 35 | 1 | 0 | 0 |
| 35 | 32 | 2.05e-5 | 4.47e-8 |
| 35 | 512 | 2.25e-5 | 5.96e-8 |

The 20b comparison also passes all six samples: layers 0 and 23 at 1, 32 and
512 rows. Its maximum output normalized RMSE is `2.45e-5`, and its maximum
routing weight error is `1.19e-7`.

Router component regressions compare original default-workspace CUDA Linear
with production routing at all power-of-two capacities from 1 to 16384,
for both 32 and 128 Experts. They include a misaligned parent workspace view,
CUDA Graph replay and unchanged allocated device bytes. The installed native
TP2 case alternates layers A, B, A with independent Lane scratch and checks
complete output against independently decoded Expert weights.

## Serving and KV results

All three serving modes pass for each model. Decode Full returns the same
eight token IDs as Eager. Prefill comparison uses the complete nonempty
`[1, 201088]` FP32 logit distribution and the existing forward-KL limit `0.05`:

| Model | Eager vs combined-mode Prefill KL | Second 4097-token request cached tokens |
| --- | --- | --- |
| 20b | 0.0053760093 | 4096 |
| 120b | 0.0159364007 | 4096 |

Both long-prefix requests return HTTP 200 and exactly two output tokens per
model. The first request has no cached tokens; the second has a positive cache
hit. Graph Observer evidence confirms combined Decode Full and Prefill
Breakable execution. These results establish the configured sliding-window
and ordinary Elastic KV path rather than full advertised-context coverage.

## Memory evidence

Exact retained FFN Tensor bytes, including packed values, scales, Expert
biases and Router weights/bias, follow checkpoint geometry:

| Model | Packed values | Scales | Expert biases | Router | Total bytes |
| --- | --- | --- | --- | --- | --- |
| 20b | 9,555,148,800 | 597,196,800 | 13,271,040 | 4,425,216 | 10,170,041,856 |
| 120b | 57,330,892,800 | 3,583,180,800 | 79,626,240 | 26,551,296 | 61,020,251,136 |

These are resident weight bytes, not startup peaks or serving admission
budgets. Real-model allocation inventories and stage observations are required
to validate packed residency and the complete memory estimate.

All four real-model worlds pass. Their allocation inventories contain the
expected 24 or 36 layers and exactly the resident bytes above; packed matrices
and scales are `torch.uint8`. Every observed startup stage has zero
underprediction, with a minimum stage headroom of 3,407,872 bytes.

| Model | World | Largest predicted stage, bytes | Largest observed stage, bytes | Pressure positioning below predicted peak, bytes |
| --- | --- | --- | --- | --- |
| 20b | Baseline | 18,069,532,732 | 17,876,123,648 | Not applicable |
| 20b | Pressure | 18,069,532,732 | 17,876,123,648 | 1,381,996–1,781,820 |
| 120b | Baseline | 41,385,413,068 | 41,152,413,696 | Not applicable |
| 120b | Pressure | 41,385,413,068 | 41,152,413,696 | 508,684–1,526,220 |

The resource worlds match the production serving geometry: Decode capacity
four, Prefill capacity 16384, attention TP2/DP1, FFN TP1 and one Lane. The
production placement solver assigns 20b's 24 layers to one FfnAgent and 120b's
36 layers to two FfnAgents in 17/19 groups. The other FfnAgents still join the
six-participant Fabric and retain their normal control resources. This is
evidence for the selected six-device topology, not qualification of a smaller
deployment.

The ABI-86 calibration uses 18 fit worlds and three disjoint held-out worlds.
Its minimum held-out headroom is 3,244,032 bytes. Nonzero coefficients are
3,407,872 base bytes, 10,092,544 bytes per Executor Lane,
31,522,816 bytes for the MoE implementation, and 3,307,929,600 bytes for a
joined FfnAgent. All other fitted coefficients are zero. The compatible local
profile is
`.xpool-cache/qualification/gpt-oss-20261017-router/memory-calibration.json`.

Real-model pressure worlds must use zero operator margin and hold a raw CUDA
reserve across weight materialization, Fabric join, graph capture, execution
installation and retained observation. Actual post-reserve free bytes must be
at or below the predicted peak and within one 16-MiB positioning quantum.
That quantum does not permit estimator underprediction.
The evidence runner requests the exact difference between initial free bytes
and the predicted peak. CUDA allocation rounding determines the measured
positioning error reported above. The reserve stays live through the complete
ordered measurement and native finalization. Raw positions, plans, inventories,
observations and verdicts are retained separately; `resource-summary.json`
collects their measured results in the qualification artifact directory.

## Performance diagnostics

The correctness runs retain one FFN invocation per representative layer and
row count, with observers enabled. The following ranges span the two layers
for each model and compare timestamps only within each participant's device
clock domain. They are diagnostic compute intervals, not benchmark estimates:

| Model | Rows | Observed FFN compute interval, ms |
| --- | --- | --- |
| 20b | 1 | 1.242–1.354 |
| 20b | 32 | 4.055–4.156 |
| 20b | 512 | 17.413–17.505 |
| 120b | 1 | 1.050–1.253 |
| 120b | 32 | 9.293–10.167 |
| 120b | 512 | 23.598–23.883 |

Single-request serving observations use an 81-token prompt and eight output
tokens. Latencies below are SGLang's `meta_info.e2e_latency`; the rate is
`completion_tokens / e2e_latency`, not aggregate concurrent serving throughput.

| Model | Mode | Request latency, s | Output tokens per request-second |
| --- | --- | --- | --- |
| 20b | Eager | 0.4475 | 17.88 |
| 20b | Decode Full | 0.3987 | 20.07 |
| 20b | Decode Full + Prefill Breakable | 0.3995 | 20.02 |
| 120b | Eager | 0.6769 | 11.82 |
| 120b | Decode Full | 0.6624 | 12.08 |
| 120b | Decode Full + Prefill Breakable | 0.7137 | 11.21 |

The repeated long-prefix request changes the observed latency from 3.413 to
0.104 seconds for 20b and from 5.293 to 0.152 seconds for 120b. No performance
threshold or tuning claim is attached to these samples.

## Diagnosed failures retained for comparison

- Uncalibrated 120b placement exhausted memory during Fabric join. An analytic
  weight fit omitted the measured opaque startup overhead; the deployment now
  selects a compatible, model-neutral calibration profile.
- Workspace-free Router GEMMs changed 120b Expert selection because original
  cuBLASLt may round Split-K partials to BF16. The fixed explicit-workspace
  Router passes the original independent numerical policy.
- Linking Toolkit cuBLASLt while using Torch-owned handles loaded two library
  versions when the CLI imported native code first. The build now links and
  searches Torch's pinned cuBLASLt library. Toolkit CUDA headers remain paired
  with NVCC and CCCL.
- The synthetic installed test assumed a separate BF16 product/bias boundary.
  Its independent reference now uses original CUDA Linear for Router rounding;
  Expert decoding and output tolerances are unchanged.
- Strict-suite preparation found an October 9 MPS endpoint directory for
  physical devices 4/5. Its controller PID was absent, no endpoint socket was
  live, all current MPS servers belonged to other controllers, and the control
  probe reported that no daemon existed. The directory was renamed intact to
  preserve its contents and release the endpoint address. The evidence is in
  `gpt-oss-20261017-router/mps-stale-endpoint-preservation.json`.

Failed runs are diagnostic evidence and do not count as accepted verdicts.
No performance threshold, full-context qualification or throughput guarantee
is established by these correctness tests.

## Code changes and documentation ownership

The completed implementation adds typed MXFP4 model specifications, bounded
packed checkpoint loading and TP slicing, SM80 tile-decoding Expert kernels,
and GPT-OSS activation and bias semantics. Quantized weights, scales and
biases participate in the existing graph resource binding, Control Probe and
memory estimate contracts. The attention shim removes the complete upstream
FFN before loading and retains ordinary Elastic KV storage with sliding masks.

The final qualification fixes add a native biased BF16 Router with explicit
per-Lane workspace, warm algorithm selection and allocation-free graph replay.
Native ABI 86 and generated stubs expose that operation. CMake and the direct
`nvidia-cublas` dependency select Torch's cuBLASLt runtime while keeping
Toolkit compilation headers. Regression references preserve original CUDA
Linear rounding, and the native observer admits valid idle FFN participants.
Portable GPT-OSS deployments now select four FFN devices with two attention
devices; their historical configuration filenames are retained.

Current contracts belong to [FFN Execution](../../designs/ffn-execution.md),
[Control Plane](../../designs/control-plane.md),
[Elastic KV Cache Pooling](../../designs/elastic-kv-cache.md),
[Overview](../../designs/overview.md) and
[Qualification](../../designs/qualification.md). Both models are marked
supported within the measured boundary in the
[model catalogue](../../supported-models.md).
The user selected retention of the completed task directory and its acceptance
materials for PR review. Current designs own its durable decisions; later
cleanup requires a separate user decision.
