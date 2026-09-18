"""Dense Qwen3 SGLang FFN class replacement."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from transformers import Qwen3Config

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.models.qwen2 import XpoolQwen2MLP
from xpool.native.ffn import LayerKind


class Qwen3ShimAdapter(SglangShimAdapter):
    """SGLang hooks and fail-closed policy for dense Qwen3 models."""

    name = "qwen3"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace dense Qwen3 MLPs and filter their attention-side weights."""

        return (
            SglangHook(
                target="sglang.srt.models.qwen3.Qwen3MLP",
                handler=XpoolQwen2MLP,
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

        model = model_runner.model
        if not isinstance(model, Qwen3ForCausalLM):
            raise RuntimeError("xpool Qwen3 model runner did not load a Qwen3ForCausalLM model")
        config = model.config
        if not isinstance(config, Qwen3Config):
            raise RuntimeError("xpool Qwen3 model runner did not load a Qwen3Config")
        layer_count = config.num_hidden_layers
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool Qwen3 model config has no positive integer num_hidden_layers")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * layer_count,
            allowed_shim_types=(XpoolQwen2MLP,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=False)


def around_load_weights(
    original_fn: Callable[[Qwen3ForCausalLM, Iterable[tuple[str, torch.Tensor]]], None],
    model: Qwen3ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Remove dense FFN tensors before invoking Qwen3's original loader."""

    original_fn(model, filter_decoder_ffn_weights(weights))
