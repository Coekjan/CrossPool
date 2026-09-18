"""Dense Qwen2 SGLang FFN class and decoder replacements."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP
from sglang.srt.plugins.hook_registry import HookType
from transformers import Qwen2Config

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")


class XpoolQwen2MLP(FfnShimModule, Qwen2MLP):
    """Parameter-free replacement for a gated Qwen2 or Qwen3 MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize CrossPool shim state while SGLang retains the FFN weights."""

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool Qwen dense MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=LayerKind.DENSE,
        )


class Qwen2ShimAdapter(SglangShimAdapter):
    """SGLang hooks and post-load policy for dense Qwen2 models."""

    name = "qwen2"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace Qwen2 MLP and decoder execution and filter FFN weights."""

        return (
            SglangHook(
                target="sglang.srt.models.qwen2.Qwen2MLP",
                handler=XpoolQwen2MLP,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.qwen2.Qwen2DecoderLayer.forward",
                handler=qwen2_decoder_forward,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.qwen2.Qwen2ForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the dense Qwen2 causal-language-model architecture."""

        return "Qwen2ForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete Dense shim coverage after Qwen2 model loading."""

        model = model_runner.model
        if not isinstance(model, Qwen2ForCausalLM):
            raise RuntimeError("xpool Qwen2 model runner did not load a Qwen2ForCausalLM model")
        config = model.config
        if not isinstance(config, Qwen2Config):
            raise RuntimeError("xpool Qwen2 model runner did not load a Qwen2Config")
        layer_count = config.num_hidden_layers
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool Qwen2 model config has no positive integer num_hidden_layers")
        self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * layer_count,
            allowed_shim_types=(XpoolQwen2MLP,),
        )


def qwen2_decoder_forward(
    self: Qwen2DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve Qwen2 attention and norms while passing ForwardBatch to the FFN shim."""

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, quant_linear=self.self_attn.qkv_proj)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual, quant_linear=self.self_attn.qkv_proj)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )
    # The FFN is remote and has no local gate_up_proj for RMSNorm's optional
    # downstream static-FP8 quantization fusion. The admitted model is BF16.
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
    return hidden_states, residual


def around_load_weights(
    original_fn: Callable[[Qwen2ForCausalLM, Iterable[tuple[str, torch.Tensor]]], set[str] | None],
    model: Qwen2ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str] | None:
    """Remove decoder FFN tensors before invoking Qwen2's original loader."""

    return original_fn(model, filter_decoder_ffn_weights(weights))
