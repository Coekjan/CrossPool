from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from sglang.srt.server_args import ServerArgs


def server_args(**overrides: object) -> ServerArgs:
    values: dict[str, object] = {
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "speculative_algorithm": None,
        "enable_lora": None,
        "lora_paths": None,
        "quantization": None,
        "ep_size": 1,
        "moe_dense_tp_size": 1,
        "moe_a2a_backend": "none",
        "speculative_moe_a2a_backend": None,
        "enable_deepep_waterfill": False,
        "elastic_ep_backend": None,
        "enable_eplb": False,
        "expert_distribution_recorder_mode": None,
        "enable_two_batch_overlap": False,
        "enable_single_batch_overlap": False,
        "enable_mixed_chunk": False,
        "enable_dp_attention": False,
        "enable_attn_tp_input_scattered": False,
        "enable_prefill_context_parallel": False,
        "enable_dsa_prefill_context_parallel": False,
        "enable_flashinfer_allreduce_fusion": False,
        "enable_aiter_allreduce_fusion": False,
        "cpu_offload_gb": 0,
        "offload_group_size": -1,
        "enable_hierarchical_cache": False,
        "disaggregation_decode_enable_offload_kvcache": False,
        "disaggregation_mode": "null",
        "dllm_algorithm": None,
        "enable_pdmux": False,
    }
    values.update(overrides)
    return cast(ServerArgs, FakeServerArgs(**values))


@dataclass(slots=True)
class FakeServerArgs:
    tp_size: int
    dp_size: int
    pp_size: int | None
    speculative_algorithm: str | None
    enable_lora: bool | None
    lora_paths: Sequence[str] | None
    quantization: str | None
    ep_size: int | None
    moe_dense_tp_size: int | None
    moe_a2a_backend: str | None
    speculative_moe_a2a_backend: str | None
    enable_deepep_waterfill: bool
    elastic_ep_backend: str | None
    enable_eplb: bool
    expert_distribution_recorder_mode: str | None
    enable_two_batch_overlap: bool
    enable_single_batch_overlap: bool
    enable_mixed_chunk: bool
    enable_dp_attention: bool
    enable_attn_tp_input_scattered: bool
    enable_prefill_context_parallel: bool
    enable_dsa_prefill_context_parallel: bool
    enable_flashinfer_allreduce_fusion: bool
    enable_aiter_allreduce_fusion: bool
    cpu_offload_gb: float | int | None
    offload_group_size: int | None
    enable_hierarchical_cache: bool
    disaggregation_decode_enable_offload_kvcache: bool
    disaggregation_mode: str
    dllm_algorithm: str | None
    enable_pdmux: bool
