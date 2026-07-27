"""Dense Qwen3 SGLang FFN class replacement."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Concatenate

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import Qwen3ForCausalLM, Qwen3MLP
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangModelAdapter,
    XpoolModelRuntime,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError

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
        """Initialize only the xpool shim state without allocating FFN weights."""

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool Qwen3 dense MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=FfnLayerKind.DENSE,
        )
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config
        self.prefix = prefix


class Qwen3Adapter(SglangModelAdapter):
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

    def bind_runtime(self, model_runner: ModelRunner) -> None:
        """Validate the complete result group and accepted Qwen topology."""

        super().bind_runtime(model_runner)
        binding = XpoolModelRuntime.require(model_runner).binding
        if binding.atn_dp_size != 1 or binding.enable_dp_attention:
            raise RuntimeError("xpool dense Qwen3 adapter supports attention DP size one only")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete dense shim coverage and FULL MLP boundaries."""

        model = getattr(model_runner, "model", None)
        if not isinstance(model, nn.Module):
            raise RuntimeError("xpool Qwen3 model runner has no loaded PyTorch model after load_model")
        expected_layer_count = getattr(model_runner.model_config.hf_config, "num_hidden_layers", None)
        if (
            not isinstance(expected_layer_count, int)
            or isinstance(expected_layer_count, bool)
            or expected_layer_count < 1
        ):
            raise RuntimeError("xpool Qwen3 model config has no positive integer num_hidden_layers")
        shims = self.require_ffn_shims(
            model,
            expected_layer_count=expected_layer_count,
            allowed_shim_types=(XpoolQwen3MLP,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=False)
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def around_load_weights[**P, R](
    original_fn: Callable[Concatenate[Qwen3ForCausalLM, Iterable[tuple[str, torch.Tensor]], P], R],
    model: Qwen3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Remove dense FFN tensors before invoking Qwen3's original loader."""

    return original_fn(model, filter_ffn_weights(weights), *args, **kwargs)


def filter_ffn_weights[W](weights: Iterable[tuple[str, W]]) -> Iterable[tuple[str, W]]:
    """Yield only tensors outside decoder-layer MLP subtrees."""

    for name, tensor in weights:
        if ".mlp." not in name:
            yield name, tensor
