"""Original multimodal Qwen3.5 decoder FFN and state implementation selection."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

import torch
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_disagg, get_forward, get_memory

from xpool.integrations.sglang.adapter import SglangShimAdapter, model_runner_architectures
from xpool.integrations.sglang.ecs.implementations.linear.gdn import GdnStateAdapter, state_hooks, validate_settings
from xpool.integrations.sglang.ecs.interfaces import SglangStateAdapter
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

LAYER_PREFIX = re.compile(r"^model(?:\.language_model)?\.layers\.(?P<layer_id>\d+)\.mlp$")
LANGUAGE_FFN_WEIGHT = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.(?:gate|up|down)_proj\.weight$")


class XpoolQwen35MLP(FfnShimModule, Qwen2MoeMLP):
    """Retain the dense constructor without local language-FFN parameters."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        if hidden_act != "silu" or quant_config is not None:
            raise ShimUnavailableError("xpool Qwen3.5 requires unquantized SiLU language FFNs")
        match = LAYER_PREFIX.fullmatch(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool Qwen3.5 cannot identify decoder FFN prefix {prefix!r}")
        FfnShimModule.__init__(
            self, layer_id=int(match.group("layer_id")), hidden_size=hidden_size, layer_kind=LayerKind.DENSE
        )


class Qwen35ShimAdapter(SglangShimAdapter):
    """Select the native multimodal wrapper and its concrete state option."""

    name = "qwen3_5"
    supports_dp_attention = True
    state_adapter_type: type[SglangStateAdapter] = GdnStateAdapter

    def hooks(self) -> tuple[SglangHook, ...]:
        return (
            SglangHook("sglang.srt.models.qwen3_5.Qwen2MoeMLP", XpoolQwen35MLP, HookType.REPLACE),
            SglangHook(
                "sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_one_request",
                around_validate_request,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5._maybe_enable_silu_fp4_quant_fusion", around_silu_fusion, HookType.AROUND
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer.forward", linear_decoder_forward, HookType.REPLACE
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.forward",
                attention_decoder_forward,
                HookType.REPLACE,
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5ForConditionalGeneration.load_weights",
                around_load_weights,
                HookType.AROUND,
            ),
            *state_hooks(),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        return "Qwen3_5ForConditionalGeneration" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        super().validate_before_load(model_runner)
        validate_settings()
        config = model_runner.model_config.hf_text_config
        if not isinstance(config, Qwen3_5TextConfig):
            raise ShimUnavailableError("xpool Qwen3.5 requires the original dense text configuration")
        if model_runner.model_config.dtype is not torch.bfloat16:
            raise ShimUnavailableError("xpool Qwen3.5 requires original BF16 execution")
        if get_disagg().encoder_only or get_disagg().language_only or get_memory().enable_session_radix_cache:
            raise ShimUnavailableError("xpool Qwen3.5 requires the complete wrapper and ordinary request lifecycle")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        model = model_runner.model
        if not isinstance(model, Qwen3_5ForConditionalGeneration) or model.visual is None:
            raise ShimUnavailableError("xpool Qwen3.5 requires its native visual and language modules")
        config = model.config
        if not isinstance(config, Qwen3_5TextConfig):
            raise ShimUnavailableError("xpool Qwen3.5 decoder configuration differs from its model binding")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * config.num_hidden_layers,
            allowed_shim_types=(XpoolQwen35MLP,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)


def around_silu_fusion(original_fn: Callable[[torch.nn.Module], None], mlp: torch.nn.Module) -> None:
    if not isinstance(mlp, XpoolQwen35MLP):
        original_fn(mlp)


def around_validate_request(
    original_fn: Callable[..., None],
    manager: TokenizerManager,
    obj: GenerateReqInput | EmbeddingReqInput,
    input_ids: list[int],
) -> None:
    if "Qwen3_5ForConditionalGeneration" in (manager.model_config.hf_config.architectures or ()) and isinstance(
        obj, GenerateReqInput
    ):
        params = obj.sampling_params
        if not isinstance(params, dict):
            raise ValueError("xpool Qwen3.5 requires normalized per-request sampling parameters")
        if (params.get("beam_width") or 1) > 1:
            raise ValueError("xpool Qwen3.5 does not support beam-search state forking")
        if obj.session_params is not None or obj.session_id is not None:
            raise ValueError("xpool Qwen3.5 does not support session-owned state")
    original_fn(manager, obj, input_ids)


def around_load_weights(
    original_fn: Callable[[Qwen3_5ForConditionalGeneration, Iterable[tuple[str, torch.Tensor]]], None],
    model: Qwen3_5ForConditionalGeneration,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    original_fn(model, ((name, tensor) for name, tensor in weights if LANGUAGE_FFN_WEIGHT.fullmatch(name) is None))


def decoder_forward(
    layer: Qwen3_5LinearDecoderLayer | Qwen3_5AttentionDecoderLayer,
    positions: torch.Tensor | None,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    forward_batch: ForwardBatch,
    captured_last_layer_outputs: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states, residual = layer.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        hidden_states,
        residual,
        forward_batch,
        captured_last_layer_outputs=captured_last_layer_outputs,
    )
    if not forward_batch.forward_mode.is_idle() and hidden_states.shape[0] > 0:
        if isinstance(layer, Qwen3_5LinearDecoderLayer):
            hidden_states = layer.linear_attn(hidden_states, forward_batch)
        else:
            hidden_states = layer.self_attention(
                positions=cast(torch.Tensor, positions), hidden_states=hidden_states, forward_batch=forward_batch
            )
    hidden_states, residual = layer.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
    scatter = layer.layer_communicator.should_use_reduce_scatter(forward_batch)
    with get_forward().scoped(fuse_mlp_allreduce=False, mlp_reduce_scatter=scatter):
        hidden_states = layer.mlp(hidden_states, forward_batch=forward_batch)
    return layer.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)


def linear_decoder_forward(
    self: Qwen3_5LinearDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    forward_batch: ForwardBatch,
    captured_last_layer_outputs: list[torch.Tensor] | None = None,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decoder_forward(self, None, hidden_states, residual, forward_batch, captured_last_layer_outputs)


def attention_decoder_forward(
    self: Qwen3_5AttentionDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    forward_batch: ForwardBatch,
    captured_last_layer_outputs: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decoder_forward(self, positions, hidden_states, residual, forward_batch, captured_last_layer_outputs)
