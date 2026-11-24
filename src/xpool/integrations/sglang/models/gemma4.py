"""CrossPool FFN hooks for the Gemma 4 Dense decoder and its two wrappers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

import torch
from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_rmsnorm_residual_scalar
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.gemma4_causal import Gemma4Attention, Gemma4DecoderLayer, Gemma4TextModel
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import add_prefix
from transformers import Gemma4TextConfig, PretrainedConfig

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

_ARCHITECTURES = frozenset({"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"})
_FFN_WEIGHT = re.compile(r"model\.language_model\.layers\.\d+\.mlp\.(?:gate|up|down)_proj\.weight\Z")


class Gemma4ShimAdapter(SglangShimAdapter):
    """Disaggregate the Dense text MLPs inside both Gemma 4 wrappers."""

    name = "gemma4"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace the Gemma 4 decoder seams and filter raw wrapper checkpoints."""

        decoder = "sglang.srt.models.gemma4_causal.Gemma4DecoderLayer"
        return (
            SglangHook(f"{decoder}.__init__", gemma4_decoder_init, HookType.REPLACE),
            SglangHook(f"{decoder}.forward", gemma4_decoder_forward, HookType.REPLACE),
            SglangHook(
                "sglang.srt.models.gemma4_mm.Gemma4ForConditionalGeneration.load_weights",
                around_load_weights,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.models.gemma4_unified.Gemma4UnifiedForConditionalGeneration.load_weights",
                around_load_weights,
                HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the original SGLang wrapper architectures."""

        return not model_runner_architectures(model_runner).isdisjoint(_ARCHITECTURES)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Require native BF16 serving with eager prefill."""

        super().validate_before_load(model_runner)
        graph = get_exec().graph.cuda_graph_config
        if graph is None or graph.prefill.backend != Backend.DISABLED:
            raise ShimUnavailableError("xpool Gemma 4 requires --cuda-graph-backend-prefill disabled")
        if model_runner.model_config.dtype is not torch.bfloat16 or not model_runner.model_config.is_multimodal:
            raise ShimUnavailableError("xpool Gemma 4 requires the BF16 native multimodal wrapper")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Verify the text decoder owns one parameter-free shim at each ordinal."""

        model = model_runner.model
        if type(model) not in (Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration):
            raise ShimUnavailableError("xpool Gemma 4 requires a native conditional generation model")
        wrapper = cast(Gemma4ForConditionalGeneration, model)
        text_model = wrapper.language_model
        if not isinstance(text_model, Gemma4TextModel):
            raise ShimUnavailableError("xpool Gemma 4 wrapper has an invalid text decoder")
        if isinstance(wrapper, Gemma4UnifiedForConditionalGeneration) and not hasattr(wrapper, "lm_head_is_tied"):
            # The pinned Unified constructor omits this flag read by the shared forward (SGLang #35809).
            wrapper.lm_head_is_tied = wrapper.lm_head is text_model.embed_tokens
        layers = text_model.layers
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * text_model.config.num_hidden_layers,
            allowed_shim_types=(FfnShimModule,),
        )
        for layer, shim in zip(layers, shims, strict=True):
            if layer.mlp is not shim or shim.hidden_size != text_model.config.hidden_size or tuple(shim.parameters()):
                raise ShimUnavailableError(f"xpool Gemma 4 decoder layer {shim.layer_id} has an invalid FFN shim")


def gemma4_decoder_init(
    self: Gemma4DecoderLayer,
    layer_id: int,
    config: PretrainedConfig,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
) -> None:
    """Construct the pinned Dense decoder with parameter-free FFN state."""

    if quant_config is not None or config.enable_moe_block or config.hidden_activation != "gelu_pytorch_tanh":
        raise ShimUnavailableError("xpool Gemma 4 requires an unquantized Dense decoder with gated tanh GELU")

    torch.nn.Module.__init__(self)
    self.hidden_size = config.hidden_size
    self.hidden_size_per_layer_input = config.hidden_size_per_layer_input or 0
    self.layer_id = layer_id
    self.is_full_attention = config.layer_types[layer_id] == "full_attention"
    self.self_attn = Gemma4Attention(
        layer_id=layer_id,
        # The pinned decoder consumes the same normalized fields for the Unified text config.
        config=cast(Gemma4TextConfig, config),
        max_position_embeddings=config.max_position_embeddings,
        head_dim=config.head_dim if self.is_full_attention else config.swa_head_dim,
        quant_config=quant_config,
        prefix=add_prefix("self_attn", prefix),
    )
    self.add_module("mlp", FfnShimModule(layer_id=layer_id, hidden_size=self.hidden_size, layer_kind=LayerKind.DENSE))
    self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
    self.pre_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
    self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    self.has_ple = self.hidden_size_per_layer_input > 0
    self.per_layer_input_gate = None
    self.per_layer_projection = None
    self.post_per_layer_input_norm = None
    if self.has_ple:
        self.per_layer_input_gate = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size_per_layer_input,
            bias=False,
            prefix=add_prefix("per_layer_input_gate", prefix),
        )
        self.per_layer_projection = ReplicatedLinear(
            self.hidden_size_per_layer_input,
            self.hidden_size,
            bias=False,
            prefix=add_prefix("per_layer_projection", prefix),
        )
        self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    self.enable_moe_block = False
    self.router = None
    self.moe = None
    self.post_feedforward_layernorm_1 = None
    self.post_feedforward_layernorm_2 = None
    self.pre_feedforward_layernorm_2 = None
    self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
    self.prefix = prefix


def gemma4_decoder_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor | None,
    forward_batch: ForwardBatch,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Keep decoder attention and PLE local while dispatching the normalized MLP input."""

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)

    norm = self.post_feedforward_layernorm
    layer_scalar = self.get_buffer("layer_scalar")
    if not self.has_ple and hidden_states.is_cuda and hidden_states.dim() == 2:
        return (
            gemma_rmsnorm_residual_scalar(
                hidden_states, norm.weight.data, residual, layer_scalar, norm.variance_epsilon
            ),
            None,
        )

    hidden_states = norm(hidden_states) + residual
    if self.has_ple and per_layer_input is not None:
        # The constructor creates all three modules together when PLE is enabled.
        gate, _ = cast(ReplicatedLinear, self.per_layer_input_gate)(hidden_states)
        gated_input = torch.nn.functional.gelu(gate, approximate="tanh") * per_layer_input
        contribution, _ = cast(ReplicatedLinear, self.per_layer_projection)(gated_input)
        hidden_states = hidden_states + cast(Gemma4RMSNorm, self.post_per_layer_input_norm)(contribution)
    return hidden_states * layer_scalar, None


def around_load_weights(
    original_fn: Callable[[Gemma4ForConditionalGeneration, Iterable[tuple[str, torch.Tensor]]], set[str]],
    model: Gemma4ForConditionalGeneration,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Keep the decoder FFN weights with FfnAgents and delegate all other keys."""

    return original_fn(model, filter_decoder_ffn_weights(weights, pattern=_FFN_WEIGHT))
