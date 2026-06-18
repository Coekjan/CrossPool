"""DeepSeek-V2 SGLang FFN class replacements."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, TypeVar

import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    PretrainedConfig,
    QuantizationConfig,
)
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangModelAdapter,
    assert_ffn_shim_coverage,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnLayerKind, FfnShimModule, ShimUnavailableError

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")

P = ParamSpec("P")
ReturnT = TypeVar("ReturnT")
WeightT = TypeVar("WeightT")


@dataclass(slots=True)
class DeepseekShimMoeRunnerConfig:
    """Minimal runner config surface expected by SGLang MoE control flow.

    Attributes:
        inplace: Whether SGLang may treat MoE output handling as in-place.
    """

    inplace: bool = True


class DeepseekShimExperts:
    """Minimal MoE expert surface used by SGLang decoder-layer control flow."""

    def __init__(self) -> None:
        """Create a parameter-free expert placeholder.

        Side Effects:
            Attaches ``moe_runner_config`` so SGLang code that inspects MoE
            metadata can run without allocating attention-side expert weights.
        """

        self.moe_runner_config = DeepseekShimMoeRunnerConfig()


class XpoolDeepseekV2MLP(FfnShimModule, DeepseekV2MLP):
    """Dense DeepSeek MLP replaced by the xpool FFN shim."""

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
        swiglu_limit: float | None = None,
    ) -> None:
        """Initialize a DeepSeek dense MLP replacement.

        Args:
            hidden_size: Hidden-state width declared by SGLang.
            intermediate_size: Original dense FFN intermediate width retained
                for diagnostics and compatibility only.
            hidden_act: Activation name. Must be ``"silu"`` for the current shim.
            quant_config: Original SGLang quantization config, retained for compatibility.
            reduce_results: Original SGLang reduce-results flag, retained for compatibility.
            prefix: SGLang module prefix used to derive the decoder layer id.
            tp_rank: Original SGLang tensor-parallel rank, retained for compatibility.
            tp_size: Original SGLang tensor-parallel size, retained for compatibility.
            swiglu_limit: Original SGLang SwiGLU limit, retained for compatibility.

        Raises:
            ValueError: If SGLang requests an unsupported activation.
            ShimUnavailableError: If the layer id cannot be derived from ``prefix``.

        Side Effects:
            Initializes only the xpool shim base; the original dense FFN
            constructor is intentionally not called, so no FFN weights are allocated.
        """

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        match = LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ShimUnavailableError(f"xpool DeepSeek dense MLP shim cannot derive layer id from prefix {prefix!r}")
        FfnShimModule.__init__(
            self,
            layer_id=int(match.group("layer_id")),
            hidden_size=hidden_size,
            layer_kind=FfnLayerKind.DENSE,
        )
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config
        self.reduce_results = reduce_results
        self.prefix = prefix
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.swiglu_limit = swiglu_limit


class XpoolDeepseekV2MoE(FfnShimModule, DeepseekV2MoE):
    """Sparse DeepSeek MoE replaced by the xpool FFN shim."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        is_nextn: bool = False,
        is_deepseek_v4: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        """Initialize a DeepSeek MoE replacement.

        Args:
            config: SGLang/Hugging Face config containing hidden-size and MoE metadata.
            layer_id: Decoder layer id routed to the native FFN executor.
            quant_config: Original SGLang quantization config, retained for compatibility.
            prefix: SGLang module prefix retained for diagnostics.
            alt_stream: Original SGLang alternate CUDA stream, retained for compatibility.
            is_nextn: Whether the layer belongs to SGLang draft/next-token flow. Must be false.
            is_deepseek_v4: Original SGLang DeepSeek-V4 compatibility flag.
            dsa_enable_prefill_cp: Original SGLang DSA context-parallel flag.
            mla_enable_prefill_cp: Original SGLang MLA context-parallel flag.

        Raises:
            ValueError: If SGLang requests an unsupported activation.
            ShimUnavailableError: If the layer is next-token draft flow or config
                lacks an integer hidden size.

        Side Effects:
            Initializes only the xpool shim base and attaches minimal SGLang MoE
            compatibility attributes; original expert weights are not allocated.
        """

        if is_nextn:
            raise ShimUnavailableError("xpool DeepSeek shim does not support next-token draft FFN layers")
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        hidden_size = getattr(config, "hidden_size", None)
        if not isinstance(hidden_size, int):
            raise ShimUnavailableError("xpool DeepSeek shim requires integer config field hidden_size")
        FfnShimModule.__init__(
            self,
            layer_id=layer_id,
            hidden_size=hidden_size,
            layer_kind=FfnLayerKind.SPARSE,
        )
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn
        self.is_deepseek_v4 = is_deepseek_v4
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        self.num_fused_shared_experts = 0
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.experts = DeepseekShimExperts()

    def get_moe_weights(self) -> list[torch.Tensor]:
        """Return MoE weights exposed to SGLang runtime helpers.

        Returns:
            Empty list because attention-side shims intentionally carry no expert
            tensors.

        Side Effects:
            None.
        """

        # The shim carries no expert weights; return an empty list (not a dict) to match
        # SGLang's DeepseekV2MoE.get_moe_weights -> list[torch.Tensor] contract, so callers
        # that .append (e.g. the EPLB rebalancer) do not raise AttributeError.
        return []


class DeepseekV2Adapter(SglangModelAdapter):
    """SGLang hooks and validation policy for DeepSeek-V2 models."""

    name = "deepseek_v2"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Return SGLang hook declarations for DeepSeek-V2 FFN replacement.

        Returns:
            Hooks that replace dense MLP and MoE classes and wrap DeepSeek weight loading.
        """

        return (
            SglangHook(
                target="sglang.srt.models.deepseek_v2.DeepseekV2MLP",
                handler=XpoolDeepseekV2MLP,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.deepseek_v2.DeepseekV2MoE",
                handler=XpoolDeepseekV2MoE,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Return whether this adapter owns a SGLang model runner.

        Args:
            model_runner: SGLang model runner before or after model construction.

        Returns:
            ``True`` when the runner's Hugging Face architectures name a
            DeepSeek-V2 model.
        """

        return "DeepseekV2ForCausalLM" in model_runner_architectures(model_runner)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Validate that every DeepSeek decoder FFN was replaced by an xpool shim.

        Args:
            model_runner: SGLang model runner after ``load_model`` completes.

        Raises:
            RuntimeError: If the model is missing, not a PyTorch module, lacks a
                valid layer count, or has incomplete FFN shim coverage.

        Side Effects:
            Stamps ``xpool_ffn_shim_count`` on the model runner for diagnostics.
        """

        model = getattr(model_runner, "model", None)
        if model is None:
            raise RuntimeError("xpool DeepSeek model runner has no loaded model after load_model")
        if not isinstance(model, nn.Module):
            raise RuntimeError(f"xpool DeepSeek model runner loaded non-module model {type(model).__name__}")

        expected_layer_count = getattr(model_runner.model_config.hf_config, "num_hidden_layers", None)
        if (
            isinstance(expected_layer_count, bool)
            or not isinstance(expected_layer_count, int)
            or expected_layer_count < 1
        ):
            raise RuntimeError("xpool DeepSeek model config has no integer num_hidden_layers")
        shims = assert_ffn_shim_coverage(
            model,
            adapter_name=self.name,
            expected_layer_count=expected_layer_count,
            allowed_shim_types=(XpoolDeepseekV2MLP, XpoolDeepseekV2MoE),
        )
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def around_load_weights(
    original_fn: Callable[Concatenate[DeepseekV2ForCausalLM, Iterable[tuple[str, torch.Tensor]], P], ReturnT],
    model: DeepseekV2ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> ReturnT:
    """Filter DeepSeek FFN weights before invoking SGLang's weight loader.

    Args:
        original_fn: Original DeepSeek ``load_weights`` callable.
        model: DeepSeek model being loaded by SGLang.
        weights: Name/tensor pairs yielded by SGLang weight loading.
        *args: Positional arguments forwarded to ``original_fn``.
        **kwargs: Keyword arguments forwarded to ``original_fn``.

    Returns:
        Return value from ``original_fn``.

    Side Effects:
        Prevents SGLang's FFN weight loader branches from seeing ``mlp`` weights
        that the parameter-free xpool shims cannot consume.
    """

    return original_fn(model, filter_ffn_weights(weights), *args, **kwargs)


def filter_ffn_weights(weights: Iterable[tuple[str, WeightT]]) -> Iterable[tuple[str, WeightT]]:
    """Drop FFN weight tensors before SGLang's DeepSeek weight loader sees them.

    Args:
        weights: Original SGLang weight iterator.

    Yields:
        Non-FFN weight pairs that should still be loaded into the attention-side model.

    Side Effects:
        Lazily filters the iterator; it does not consume weights until SGLang's
        original loader iterates the returned iterable.

    The shim FFN modules carry no parameters, so they cannot absorb FFN weights.
    This filter is not redundant with the parameter-less shim: SGLang's
    ``DeepseekV2WeightLoaderMixin`` builds MoE expert parameter mappings from
    ``config.n_routed_experts`` and routes ``model.layers.*.mlp.*`` tensors into
    expert/shared-expert weight loaders that the shim does not provide. Removing the
    whole ``mlp`` subtree here keeps that MoE-specific weight-loading branch idle.
    """
    for name, tensor in weights:
        if name.startswith("model.layers.") and ".mlp." in name:
            continue
        yield name, tensor
