"""Dense Llama SGLang FFN class and decoder replacements."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from sglang.srt.plugins.hook_registry import HookType
from transformers import LlamaConfig

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")


class XpoolLlamaMLP(FfnShimModule, LlamaMLP):
    """Parameter-free replacement for a gated Llama MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        use_dp_attention_reduce: bool = False,
    ) -> None:
        """Initialize CrossPool shim state without allocating local FFN weights."""

        del intermediate_size, quant_config, reduce_results, tp_rank, tp_size, use_dp_attention_reduce
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool Llama MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=LayerKind.DENSE,
        )


class LlamaShimAdapter(SglangShimAdapter):
    """SGLang hooks and post-load policy for dense Llama models."""

    name = "llama"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace Llama MLP and decoder execution and filter FFN weights."""

        return (
            SglangHook(
                target="sglang.srt.models.llama.LlamaMLP",
                handler=XpoolLlamaMLP,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.llama.LlamaDecoderLayer.forward",
                handler=llama_decoder_forward,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.llama.LlamaForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the dense Llama causal-language-model architecture."""

        return "LlamaForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete Dense shim coverage after Llama model loading."""

        model = model_runner.model
        if not isinstance(model, LlamaForCausalLM):
            raise RuntimeError("xpool Llama model runner did not load a LlamaForCausalLM model")
        config = model.config
        if not isinstance(config, LlamaConfig):
            raise RuntimeError("xpool Llama model runner did not load a LlamaConfig")
        layer_count = config.num_hidden_layers
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool Llama model config has no positive integer num_hidden_layers")
        self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * layer_count,
            allowed_shim_types=(XpoolLlamaMLP,),
        )


def llama_decoder_forward(
    self: LlamaDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve Llama attention and norms while passing ForwardBatch to the FFN shim."""

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
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
    return hidden_states, residual


def around_load_weights(
    original_fn: Callable[[LlamaForCausalLM, Iterable[tuple[str, torch.Tensor]]], object],
    model: LlamaForCausalLM,
    checkpoint_weights: Iterable[tuple[str, torch.Tensor]],
) -> object:
    """Remove decoder FFN tensors before invoking Llama's original loader."""

    return original_fn(model, filter_decoder_ffn_weights(checkpoint_weights))
