"""OPT decoder replacement around SGLang's original attention."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.opt import OPTAttention, OPTDecoder, OPTDecoderLayer, OPTForCausalLM, OPTModel
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import OPTConfig

from xpool.integrations.sglang.adapter import SglangShimAdapter, filter_decoder_ffn_weights, model_runner_architectures
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.native.ffn import LayerKind

OPT_FFN_WEIGHT_PATTERN = re.compile(r"^(?:model\.)?decoder\.layers\.\d+\.(?:fc1|fc2)(?:\.|$)")


class XpoolOptDecoderLayer(OPTDecoderLayer):
    """Build upstream attention and norms without allocating local FFN weights."""

    def __init__(
        self,
        config: OPTConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.mlp = FfnShimModule(layer_id=layer_id, hidden_size=config.hidden_size, layer_kind=LayerKind.DENSE)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class XpoolOptModel(OPTModel):
    """Expose the existing decoder to SGLang's Prefill Graph resolver."""

    @property
    def model(self) -> OPTDecoder:
        return self.decoder


class OptShimAdapter(SglangShimAdapter):
    """SGLang hooks and post-load validation for OPT."""

    name = "opt"

    def hooks(self) -> tuple[SglangHook, ...]:
        return (
            SglangHook(target="sglang.srt.models.opt.OPTModel", handler=XpoolOptModel, kind=HookType.REPLACE),
            SglangHook(
                target="sglang.srt.models.opt.OPTDecoderLayer", handler=XpoolOptDecoderLayer, kind=HookType.REPLACE
            ),
            SglangHook(
                target="sglang.srt.models.opt.OPTForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        return "OPTForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        model = model_runner.model
        if not isinstance(model, OPTForCausalLM) or not isinstance(model.config, OPTConfig):
            raise RuntimeError("xpool OPT model runner did not load an OPT model and config")
        self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * model.config.num_hidden_layers,
            allowed_shim_types=(FfnShimModule,),
        )


def around_load_weights(
    original_fn: Callable[[OPTForCausalLM, Iterable[tuple[str, torch.Tensor]]], None],
    model: OPTForCausalLM,
    checkpoint_weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Load upstream attention weights and freeze inference parameters."""

    original_fn(model, filter_decoder_ffn_weights(checkpoint_weights, pattern=OPT_FFN_WEIGHT_PATTERN))
    model.requires_grad_(False)
