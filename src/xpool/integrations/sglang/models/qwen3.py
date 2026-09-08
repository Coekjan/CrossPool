"""Dense Qwen3 SGLang FFN class replacement."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import Qwen3ForCausalLM, Qwen3MLP
from sglang.srt.plugins.hook_registry import HookType

from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")


class XpoolQwen3MLP(FfnShimModule, Qwen3MLP):
    """Parameter-free replacement for one dense Qwen3 MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize only the CrossPool shim state without allocating FFN weights."""

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool Qwen3 dense MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=LayerKind.DENSE,
        )


class Qwen3ShimAdapter(SglangShimAdapter):
    """SGLang hooks and fail-closed policy for dense Qwen3 models."""

    name = "qwen3"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace dense Qwen3 MLPs and filter their attention-side weights."""

        return (
            SglangHook(
                target="sglang.srt.models.qwen3.Qwen3MLP",
                handler=XpoolQwen3MLP,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.qwen3.Qwen3ForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match only the dense Qwen3 causal-language-model architecture."""

        return "Qwen3ForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete dense shim coverage and FULL MLP boundaries."""

        model = getattr(model_runner, "model", None)
        if not isinstance(model, Qwen3ForCausalLM):
            raise RuntimeError("xpool Qwen3 model runner did not load a Qwen3ForCausalLM model")
        layer_count = getattr(model.config, "num_hidden_layers", None)
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool Qwen3 model config has no positive integer num_hidden_layers")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * layer_count,
            allowed_shim_types=(XpoolQwen3MLP,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=False)
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def around_load_weights(
    original_fn: Callable[[Qwen3ForCausalLM, Iterable[tuple[str, torch.Tensor]]], None],
    model: Qwen3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Remove dense FFN tensors before invoking Qwen3's original loader."""

    original_fn(model, filter_decoder_ffn_weights(weights))
