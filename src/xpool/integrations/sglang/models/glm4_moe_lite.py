"""GLM-4 MoE Lite SGLang FFN replacements."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.glm4_moe_lite import (
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteSparseMoeBlock,
    PretrainedConfig,
    QuantizationConfig,
)
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")


class XpoolGlm4MoeLiteMLP(FfnShimModule, Glm4MoeLiteMLP):
    """Parameter-free replacement for one dense GLM decoder FFN."""

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
        """Initialize only CrossPool shim state using SGLang's exact constructor surface."""

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool GLM dense MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=LayerKind.DENSE,
        )


class XpoolGlm4MoeLiteSparseMoeBlock(FfnShimModule, Glm4MoeLiteSparseMoeBlock):
    """Parameter-free replacement for one sparse GLM decoder FFN."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        is_nextn: bool = False,
    ) -> None:
        """Initialize only CrossPool shim state and reject unsupported NextN layers."""

        if is_nextn:
            raise ShimUnavailableError("xpool GLM shim does not support next-token draft FFN layers")
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


class Glm4MoeLiteShimAdapter(SglangShimAdapter):
    """SGLang hooks and validation policy for GLM-4 MoE Lite models."""

    name = "glm4_moe_lite"
    supports_dp_attention = True

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace GLM FFNs and filter their attention-side weights."""

        return (
            SglangHook(
                target="sglang.srt.models.glm4_moe_lite.Glm4MoeLiteMLP",
                handler=XpoolGlm4MoeLiteMLP,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.glm4_moe_lite.Glm4MoeLiteSparseMoeBlock",
                handler=XpoolGlm4MoeLiteSparseMoeBlock,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.glm4_moe_lite.Glm4MoeLiteForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match only the GLM-4 MoE Lite causal-language-model architecture."""

        return "Glm4MoeLiteForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require exact GLM model type, layer policy, shims, and FULL boundaries."""

        model = getattr(model_runner, "model", None)
        if not isinstance(model, Glm4MoeLiteForCausalLM):
            raise RuntimeError("xpool GLM model runner did not load a Glm4MoeLiteForCausalLM model")
        expected_layer_kinds = expected_mixed_layer_kinds(model.config, family="GLM")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=expected_layer_kinds,
            allowed_shim_types=(XpoolGlm4MoeLiteMLP, XpoolGlm4MoeLiteSparseMoeBlock),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def expected_mixed_layer_kinds(config: object, *, family: str) -> tuple[LayerKind, ...]:
    """Derive the pinned dense/sparse decoder policy from the loaded config."""

    layer_count = getattr(config, "num_hidden_layers", None)
    first_sparse_layer = getattr(config, "first_k_dense_replace", None)
    sparse_frequency = getattr(config, "moe_layer_freq", None)
    routed_experts = getattr(config, "n_routed_experts", None)
    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
        raise RuntimeError(f"xpool {family} model config has no positive integer num_hidden_layers")
    if routed_experts is None:
        return (LayerKind.DENSE,) * layer_count
    if not isinstance(routed_experts, int) or isinstance(routed_experts, bool) or routed_experts <= 0:
        raise RuntimeError(f"xpool {family} model config has invalid n_routed_experts")
    if not isinstance(first_sparse_layer, int) or isinstance(first_sparse_layer, bool) or first_sparse_layer < 0:
        raise RuntimeError(f"xpool {family} model config has invalid first_k_dense_replace")
    if not isinstance(sparse_frequency, int) or isinstance(sparse_frequency, bool) or sparse_frequency <= 0:
        raise RuntimeError(f"xpool {family} model config has invalid moe_layer_freq")
    return tuple(
        LayerKind.MOE if layer_id >= first_sparse_layer and layer_id % sparse_frequency == 0 else LayerKind.DENSE
        for layer_id in range(layer_count)
    )


def around_load_weights(
    original_fn: Callable[
        [Glm4MoeLiteForCausalLM, Iterable[tuple[str, torch.Tensor]], bool, dict[str, nn.Parameter] | None],
        None,
    ],
    model: Glm4MoeLiteForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
    params_dict: dict[str, nn.Parameter] | None = None,
) -> None:
    """Filter GLM decoder FFN tensors and reject unsupported NextN loading."""

    if is_nextn:
        raise ShimUnavailableError("xpool GLM shim does not support next-token draft weight loading")
    original_fn(model, filter_decoder_ffn_weights(weights), is_nextn, params_dict)
