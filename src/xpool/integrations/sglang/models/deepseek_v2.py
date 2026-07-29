"""DeepSeek-V2 SGLang FFN class replacements."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

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

from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangModelAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError

LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp$")


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
    supports_dp_attention = True

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
        if not isinstance(model, DeepseekV2ForCausalLM):
            raise RuntimeError("xpool DeepSeek model runner did not load a DeepseekV2ForCausalLM model")
        config = model.config
        layer_count = getattr(config, "num_hidden_layers", None)
        first_sparse_layer = getattr(config, "first_k_dense_replace", None)
        sparse_frequency = getattr(config, "moe_layer_freq", None)
        routed_experts = getattr(config, "n_routed_experts", None)
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool DeepSeek model config has no positive integer num_hidden_layers")
        if routed_experts is None:
            expected_layer_kinds = (FfnLayerKind.DENSE,) * layer_count
        else:
            if not isinstance(routed_experts, int) or isinstance(routed_experts, bool) or routed_experts <= 0:
                raise RuntimeError("xpool DeepSeek model config has invalid n_routed_experts")
            if (
                not isinstance(first_sparse_layer, int)
                or isinstance(first_sparse_layer, bool)
                or first_sparse_layer < 0
            ):
                raise RuntimeError("xpool DeepSeek model config has invalid first_k_dense_replace")
            if not isinstance(sparse_frequency, int) or isinstance(sparse_frequency, bool) or sparse_frequency <= 0:
                raise RuntimeError("xpool DeepSeek model config has invalid moe_layer_freq")
            expected_layer_kinds = tuple(
                FfnLayerKind.SPARSE
                if layer_id >= first_sparse_layer and layer_id % sparse_frequency == 0
                else FfnLayerKind.DENSE
                for layer_id in range(layer_count)
            )
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=expected_layer_kinds,
            allowed_shim_types=(XpoolDeepseekV2MLP, XpoolDeepseekV2MoE),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=True)
        setattr(model_runner, "xpool_ffn_shim_count", len(shims))


def around_load_weights(
    original_fn: Callable[[DeepseekV2ForCausalLM, Iterable[tuple[str, torch.Tensor]], bool], None],
    model: DeepseekV2ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
    is_nextn: bool = False,
) -> None:
    """Filter DeepSeek FFN weights before invoking SGLang's weight loader.

    Args:
        original_fn: Original DeepSeek ``load_weights`` callable.
        model: DeepSeek model being loaded by SGLang.
        weights: Name/tensor pairs yielded by SGLang weight loading.
        is_nextn: Whether SGLang is loading a draft/NextN model.

    Returns:
        Return value from ``original_fn``.

    Side Effects:
        Prevents SGLang's FFN weight loader branches from seeing ``mlp`` weights
        that the parameter-free xpool shims cannot consume.
    """

    if is_nextn:
        raise ShimUnavailableError("xpool DeepSeek shim does not support next-token draft weight loading")
    original_fn(model, filter_decoder_ffn_weights(weights), is_nextn)
