"""Parameter-free GPT-OSS FFN replacement and MXFP4 loading admission."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.gpt_oss import GptOssConfig, GptOssForCausalLM, GptOssSparseMoeBlock
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_exec, get_schedule

from xpool.integrations.sglang.adapter import SglangShimAdapter, filter_decoder_ffn_weights, model_runner_architectures
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind


class XpoolGptOssSparseMoeBlock(FfnShimModule, GptOssSparseMoeBlock):
    """Replace routing and Experts together, retaining no attention-side FFN weights."""

    def __init__(
        self, layer_id: int, config: GptOssConfig, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None:
        """Match the pinned block constructor while allocating only shim metadata."""

        if quant_config is None or quant_config.get_name() != "mxfp4":
            raise ValueError("xpool GPT-OSS requires MXFP4 quantization")
        FfnShimModule.__init__(self, layer_id=layer_id, hidden_size=config.hidden_size, layer_kind=LayerKind.MOE)

    def get_moe_weights(self) -> list[torch.Tensor]:
        """Expose the empty attention-side Expert parameter set."""

        return []


class GptOssShimAdapter(SglangShimAdapter):
    """Admit BF16 GPT-OSS with ordinary Elastic KV storage and Triton attention."""

    name = "gpt_oss"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace the GPT-OSS block and filter its entire FFN namespace."""

        return (
            SglangHook("sglang.srt.models.gpt_oss.GptOssSparseMoeBlock", XpoolGptOssSparseMoeBlock, HookType.REPLACE),
            SglangHook(
                "sglang.srt.models.gpt_oss.GptOssForCausalLM.load_weights", around_load_weights, HookType.AROUND
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Select only the original GPT-OSS causal-language-model architecture."""

        return "GptOssForCausalLM" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Reject unsupported cache storage, arithmetic, and attention backends."""

        config = model_runner.model_config
        if config.quantization != "mxfp4" or config.dtype is not torch.bfloat16:
            raise RuntimeError("xpool GPT-OSS requires an MXFP4 checkpoint and BF16 activations")
        if not get_schedule().disable_hybrid_swa_memory:
            raise RuntimeError("xpool GPT-OSS requires --disable-hybrid-swa-memory")
        execution = get_exec().kernel
        backends = (
            execution.attention_backend,
            execution.prefill_attention_backend or execution.attention_backend,
            execution.decode_attention_backend or execution.attention_backend,
        )
        if any(backend != "triton" for backend in backends):
            raise RuntimeError("xpool GPT-OSS on A100 requires Triton attention")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete parameter-free coverage and ordinary FULL FFN boundaries."""

        model = model_runner.model
        if not isinstance(model, GptOssForCausalLM):
            raise RuntimeError("xpool GPT-OSS model runner loaded an incompatible model")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.MOE,) * model.config.num_hidden_layers,
            allowed_shim_types=(XpoolGptOssSparseMoeBlock,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)


def around_load_weights(
    original_fn: Callable[..., None],
    model: GptOssForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
    weight_name_mapping: dict[str, str] | None = None,
) -> None:
    """Remove Router and Expert tensors before either upstream loader sees them."""

    if is_nextn or weight_name_mapping is not None:
        raise ShimUnavailableError("xpool GPT-OSS does not admit nextn or alternate checkpoint mappings")
    original_fn(model, filter_decoder_ffn_weights(weights), is_nextn=False, weight_name_mapping=None)
