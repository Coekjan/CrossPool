"""SGLang server-argument compatibility policy for the xpool plugin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True)
class ServerArgRule:
    """A named SGLang server-argument compatibility rule.

    Attributes:
        label: Title-case feature name reported when the rule rejects a launch.
        supported: Predicate over resolved SGLang ``ServerArgs``. It returns
            ``True`` only when the feature is representable by the current xpool
            shim ABI.
    """

    label: str
    supported: Callable[[ServerArgs], bool]


def validate_sglang_server_args(server_args: ServerArgs) -> None:
    """Reject SGLang runtime modes not represented in the xpool shim ABI yet.

    Args:
        server_args: Resolved SGLang server arguments from the active model runner.

    Raises:
        RuntimeError: If any configured SGLang feature can surface a forward or
            memory-management mode not represented by the current xpool shim ABI.
    """

    unsupported = tuple(rule.label for rule in SGLANG_SERVER_ARG_RULES if not rule.supported(server_args))
    if unsupported:
        joined = ", ".join(unsupported)
        raise RuntimeError(f"xpool SGLang plugin does not support these SGLang features yet: {joined}")


SGLANG_SERVER_ARG_RULES: tuple[ServerArgRule, ...] = (
    ServerArgRule("Pipeline Parallelism", lambda args: args.pp_size == 1),
    ServerArgRule("Attention Context Parallelism", lambda args: args.attn_cp_size == 1),
    ServerArgRule("Speculative Decoding", lambda args: disabled_string_option(args.speculative_algorithm)),
    ServerArgRule("LoRA", lambda args: not args.enable_lora and not args.lora_paths),
    ServerArgRule("Quantization", lambda args: args.quantization is None),
    ServerArgRule("Expert Parallelism", lambda args: args.ep_size == 1),
    ServerArgRule("MoE Dense TP", lambda args: args.moe_dense_tp_size in (None, 1)),
    ServerArgRule("MoE A2A Backend", lambda args: args.moe_a2a_backend == "none"),
    ServerArgRule(
        "Speculative MoE A2A Backend",
        lambda args: disabled_string_option(args.speculative_moe_a2a_backend),
    ),
    ServerArgRule("DeepEP Waterfill", lambda args: not args.enable_deepep_waterfill),
    ServerArgRule("Elastic Expert Parallelism", lambda args: args.elastic_ep_backend is None),
    ServerArgRule("EPLB", lambda args: not args.enable_eplb),
    ServerArgRule("Expert Distribution Recorder", lambda args: args.expert_distribution_recorder_mode is None),
    ServerArgRule("Two-Batch Overlap", lambda args: not args.enable_two_batch_overlap),
    ServerArgRule("Single-Batch Overlap", lambda args: not args.enable_single_batch_overlap),
    ServerArgRule("Torch Compile", lambda args: not args.enable_torch_compile),
    ServerArgRule("Piecewise CUDA Graph Compiler", lambda args: args.piecewise_cuda_graph_compiler == "eager"),
    ServerArgRule("Piecewise CUDA Graph Enforcement", lambda args: not args.enforce_piecewise_cuda_graph),
    ServerArgRule("Mixed Chunked Prefill", lambda args: not args.enable_mixed_chunk),
    ServerArgRule("Attention TP Input Scattering", lambda args: not args.enable_attn_tp_input_scattered),
    ServerArgRule("Prefill Context Parallelism", lambda args: not args.enable_prefill_context_parallel),
    ServerArgRule("DSA Prefill Context Parallelism", lambda args: not args.enable_dsa_prefill_context_parallel),
    ServerArgRule("FlashInfer All-Reduce Fusion", lambda args: not args.enable_flashinfer_allreduce_fusion),
    ServerArgRule("AITER All-Reduce Fusion", lambda args: not args.enable_aiter_allreduce_fusion),
    ServerArgRule("SGLang CPU Offload", lambda args: args.cpu_offload_gb == 0),
    ServerArgRule("SGLang Layer Offload", lambda args: args.offload_group_size <= 0),
    ServerArgRule("Hierarchical Cache", lambda args: not args.enable_hierarchical_cache),
    ServerArgRule("Decode KV Offload", lambda args: not args.disaggregation_decode_enable_offload_kvcache),
    ServerArgRule("PD Disaggregation", lambda args: args.disaggregation_mode == "null"),
    ServerArgRule("Diffusion LLM", lambda args: args.dllm_algorithm is None),
    ServerArgRule("PD Multiplexing", lambda args: not args.enable_pdmux),
)


def disabled_string_option(value: str | None) -> bool:
    """Return whether a nullable string-valued SGLang option is disabled.

    Args:
        value: Raw SGLang option value, usually ``None`` or a string sentinel.

    Returns:
        ``True`` when the option is absent or normalized to a disabled sentinel.
    """

    if value is None:
        return True
    normalized = str(value).strip().lower()
    return normalized in {"", "none", "null", "no", "false"}
