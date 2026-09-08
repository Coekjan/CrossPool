"""Qwen3 MoE SGLang FFN replacement."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeSparseMoeBlock
from sglang.srt.plugins.hook_registry import HookType
from transformers import Qwen3MoeConfig

from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind


class XpoolQwen3MoeSparseMoeBlock(FfnShimModule, Qwen3MoeSparseMoeBlock):
    """Parameter-free replacement for one sparse Qwen3-MoE decoder FFN."""

    def __init__(
        self,
        layer_id: int,
        config: Qwen3MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize only CrossPool shim state using SGLang's exact constructor surface."""

        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        hidden_size = getattr(config, "hidden_size", None)
        FfnShimModule.__init__(
            self,
            layer_id=layer_id,
            hidden_size=cast(int, hidden_size),
            layer_kind=LayerKind.MOE,
        )

    def get_moe_weights(self) -> list[torch.Tensor]:
        """Return the empty attention-side expert tensor set."""

        return []


class Qwen3MoeShimAdapter(SglangShimAdapter):
    """SGLang hooks and validation policy for Qwen3 MoE models."""

    name = "qwen3_moe"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace Qwen3-MoE FFNs and filter their attention-side weights."""

        return (
            SglangHook(
                target="sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock",
                handler=XpoolQwen3MoeSparseMoeBlock,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.qwen3_moe.Qwen3MoeForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match only the Qwen3-MoE causal-language-model architecture."""

        return "Qwen3MoeForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require exact Qwen3-MoE type, all-sparse shims, and FULL boundaries."""

        model = getattr(model_runner, "model", None)
        if not isinstance(model, Qwen3MoeForCausalLM):
            raise RuntimeError("xpool Qwen3-MoE model runner did not load a Qwen3MoeForCausalLM model")
        layer_count = getattr(model.config, "num_hidden_layers", None)
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool Qwen3-MoE model config has no positive integer num_hidden_layers")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.MOE,) * layer_count,
            allowed_shim_types=(XpoolQwen3MoeSparseMoeBlock,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def around_load_weights(
    original_fn: Callable[[Qwen3MoeForCausalLM, Iterable[tuple[str, torch.Tensor]], bool], None],
    model: Qwen3MoeForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_mtp: bool = False,
) -> None:
    """Filter Qwen3-MoE decoder FFN tensors and reject unsupported MTP loading."""

    if is_mtp:
        raise ShimUnavailableError("xpool Qwen3-MoE shim does not support MTP weight loading")
    original_fn(model, filter_decoder_ffn_weights(weights), is_mtp)
