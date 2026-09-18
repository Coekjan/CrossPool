"""SGLang server-argument compatibility policy for the CrossPool plugin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
)


@dataclass(frozen=True, slots=True)
class ServerArgRule:
    """A named SGLang server-argument compatibility rule.

    Attributes:
        label: Title-case feature name reported when the rule rejects a launch.
        supported: Predicate over published SGLang runtime configuration. It returns
            ``True`` only when the feature is representable by the current CrossPool
            shim ABI.
    """

    label: str
    supported: Callable[[], bool]


def validate_sglang_server_args() -> None:
    """Validate SGLang runtime modes against the CrossPool shim ABI.

    Raises:
        RuntimeError: If any configured SGLang feature selects a forward or
            memory-management mode outside the current CrossPool shim ABI.
    """

    unsupported = tuple(rule.label for rule in SGLANG_SERVER_ARG_RULES if not rule.supported())
    if unsupported:
        joined = ", ".join(unsupported)
        raise RuntimeError(f"xpool SGLang plugin does not support these SGLang features: {joined}")


SGLANG_SERVER_ARG_RULES: tuple[ServerArgRule, ...] = (
    ServerArgRule("Pipeline Parallelism", lambda: get_parallel().pp_size == 1),
    ServerArgRule("gRPC-only Serving", lambda: not get_serving().grpc_mode and not get_serving().smg_grpc_mode),
    ServerArgRule(
        "TLS Serving",
        lambda: (
            get_serving().ssl_keyfile is None
            and get_serving().ssl_certfile is None
            and get_serving().ssl_ca_certs is None
            and get_serving().ssl_keyfile_password is None
            and not get_serving().enable_ssl_refresh
        ),
    ),
    ServerArgRule("Attention Context Parallelism", lambda: get_parallel().attn_cp_size == 1),
    ServerArgRule("Speculative Decoding", lambda: get_spec().speculative_algorithm is None),
    ServerArgRule("LoRA", lambda: not get_lora().enable_lora and not get_lora().lora_paths),
    ServerArgRule("Quantization", lambda: get_model().quantization is None),
    ServerArgRule("Expert Parallelism", lambda: get_parallel().ep_size == 1),
    ServerArgRule("MoE Dense TP", lambda: get_parallel().moe_dense_tp_size in (None, 1)),
    ServerArgRule("MoE A2A Backend", lambda: get_exec().moe.moe_a2a_backend == "none"),
    ServerArgRule(
        "Speculative MoE A2A Backend",
        lambda: get_spec().speculative_moe_a2a_backend in (None, "none"),
    ),
    ServerArgRule("DeepEP Waterfill", lambda: not get_exec().moe.enable_waterfill),
    ServerArgRule("Elastic Expert Parallelism", lambda: get_exec().moe.elastic_ep_backend is None),
    ServerArgRule("EPLB", lambda: not get_exec().moe.enable_eplb),
    ServerArgRule("Expert Distribution Recorder", lambda: get_exec().moe.expert_distribution_recorder_mode is None),
    ServerArgRule("Two-Batch Overlap", lambda: not get_exec().overlap.enable_two_batch_overlap),
    ServerArgRule("Single-Batch Overlap", lambda: not get_exec().overlap.enable_single_batch_overlap),
    ServerArgRule("Torch Compile", lambda: not get_exec().graph.enable_torch_compile),
    ServerArgRule(
        "Decode CUDA Graph Backend",
        lambda: (
            get_exec().graph.cuda_graph_config is not None
            and get_exec().graph.cuda_graph_config.decode.backend in {Backend.DISABLED, Backend.FULL}
        ),
    ),
    ServerArgRule(
        "Prefill CUDA Graph Backend",
        lambda: (
            get_exec().graph.cuda_graph_config is not None
            and get_exec().graph.cuda_graph_config.prefill.backend in {Backend.DISABLED, Backend.BREAKABLE}
        ),
    ),
    ServerArgRule("Mixed Chunked Prefill", lambda: not get_schedule().enable_mixed_chunk),
    ServerArgRule("Attention TP Input Scattering", lambda: not get_parallel().enable_attn_tp_input_scattered),
    ServerArgRule("LayerNorm Sequence Parallelism", lambda: not get_parallel().enable_layernorm_sp),
    ServerArgRule("Prefill Context Parallelism", lambda: not get_parallel().enable_prefill_context_parallel),
    ServerArgRule("DSA Prefill Context Parallelism", lambda: not get_parallel().enable_dsa_prefill_context_parallel),
    ServerArgRule("FlashInfer All-Reduce Fusion", lambda: not get_exec().comm.enable_flashinfer_allreduce_fusion),
    ServerArgRule("AITER All-Reduce Fusion", lambda: not get_exec().comm.enable_aiter_allreduce_fusion),
    ServerArgRule("SGLang CPU Offload", lambda: get_exec().offload.cpu_offload_gb == 0),
    ServerArgRule("SGLang Layer Offload", lambda: get_exec().offload.offload_group_size <= 0),
    ServerArgRule("Startup Weight Loading", lambda: get_model().startup_weight_load_mode == "serial"),
    ServerArgRule("Unified Memory KV Allocator", lambda: not get_memory().enable_unified_memory),
    ServerArgRule("Page-Major KV Layout", lambda: not get_memory().enable_page_major_kv_layout),
    ServerArgRule("HND KV Layout", lambda: not envs.SGLANG_USE_HND_KVCACHE.get()),
    ServerArgRule("Disabled Radix Cache", lambda: not get_memory().disable_radix_cache),
    ServerArgRule("LMCache", lambda: not get_memory().enable_lmcache),
    ServerArgRule("FlexKV", lambda: not get_memory().enable_flexkv),
    ServerArgRule("Custom Radix Cache Backend", lambda: get_memory().radix_cache_backend is None),
    ServerArgRule("C++ Radix Tree", lambda: not envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get()),
    ServerArgRule(
        "Unified Radix TreeCore Backend",
        lambda: envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get() == "python",
    ),
    ServerArgRule("Hierarchical Cache", lambda: not get_memory().enable_hierarchical_cache),
    ServerArgRule("Decode KV Offload", lambda: not get_disagg().disaggregation_decode_enable_offload_kvcache),
    ServerArgRule("PD Disaggregation", lambda: get_disagg().disaggregation_mode == "null"),
    ServerArgRule("Diffusion LLM", lambda: get_exec().dllm.dllm_algorithm is None),
    ServerArgRule("PD Multiplexing", lambda: not get_disagg().enable_pdmux),
)
