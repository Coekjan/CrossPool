"""GPT-OSS FFN offload with upstream attention and hybrid storage unchanged."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.gpt_oss import GptOssForCausalLM, GptOssSparseMoeBlock
from sglang.srt.plugins.hook_registry import HookType
from transformers import GptOssConfig

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind


class XpoolGptOssSparseMoeBlock(FfnShimModule, GptOssSparseMoeBlock):
    """Parameter-free replacement; the FfnAgents retain Router and Expert storage."""

    def __init__(
        self,
        layer_id: int,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Preserve the original constructor while creating only the shared FFN shim."""

        FfnShimModule.__init__(self, layer_id=layer_id, hidden_size=config.hidden_size, layer_kind=LayerKind.MOE)

    def get_moe_weights(self) -> list[torch.Tensor]:
        """Expose no attention-side Expert tensors to upstream model inspection."""

        return []


class GptOssShimAdapter(SglangShimAdapter):
    """Own GPT-OSS's two model hooks and original-checkpoint compatibility checks."""

    name = "gpt_oss"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace the FFN class and retain the original non-FFN weight loader."""

        return (
            SglangHook(
                target="sglang.srt.models.gpt_oss.GptOssSparseMoeBlock",
                handler=XpoolGptOssSparseMoeBlock,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.gpt_oss.GptOssForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the original GPT-OSS causal-language-model architecture."""

        return "GptOssForCausalLM" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Require the original serialized MXFP4 model rather than runtime requantization."""

        if model_runner.model_config.quantization != "mxfp4":
            raise ShimUnavailableError("xpool GPT-OSS requires the original MXFP4 checkpoint")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require every decoder's FFN shim and original FULL communication boundary."""

        model = model_runner.model
        if not isinstance(model, GptOssForCausalLM):
            raise RuntimeError("xpool GPT-OSS model runner did not load a GptOssForCausalLM model")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.MOE,) * model.config.num_hidden_layers,
            allowed_shim_types=(XpoolGptOssSparseMoeBlock,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)


def around_load_weights(
    original_fn: Callable[[GptOssForCausalLM, Iterable[tuple[str, torch.Tensor]], bool, dict[str, str] | None], None],
    model: GptOssForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
    weight_name_mapping: dict[str, str] | None = None,
) -> None:
    """Exclude decoder FFN tensors while forwarding sinks, attention, embeddings and head."""

    if is_nextn:
        raise ShimUnavailableError("xpool GPT-OSS shim does not support MTP weight loading")
    original_fn(model, filter_decoder_ffn_weights(weights), is_nextn, weight_name_mapping)
