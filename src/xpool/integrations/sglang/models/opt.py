"""Biased unary OPT SGLang FFN class and decoder replacements."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_context, get_forward
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import OPTConfig

from xpool.integrations.sglang.adapter import (
    SglangShimAdapter,
    filter_decoder_ffn_weights,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.native.ffn import LayerKind

OPT_LAYER_PREFIX_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)$")
OPT_DECODER_FFN_WEIGHT_PATTERN = re.compile(r"^(?:model\.)?decoder\.layers\.\d+\.(?:fc1|fc2)(?:\.|$)")
OPT_FFN_PARAM_PATTERN = re.compile(r"(?:^|\.)(?:fc1|fc2)\.(?:weight|bias)$")
# FlashInfer's cascade merge kernels only cover these head dimensions; OPT-2.7b
# uses 80 and must run on the Triton attention backend instead.
FLASHINFER_MERGE_HEAD_DIMS = (64, 128, 256)
logger = logging.getLogger(__name__)


def required_attention_backend(config: OPTConfig) -> str | None:
    """Return the non-default attention backend required by FlashInfer limits."""

    if config.num_attention_heads <= 0:
        return None
    head_dim = config.hidden_size // config.num_attention_heads
    return None if head_dim in FLASHINFER_MERGE_HEAD_DIMS else "triton"


class XpoolOPTFFN(FfnShimModule):
    """Parameter-free replacement for the inline OPT FC1/ReLU/FC2 block."""

    def __init__(self, *, layer_id: int, config: OPTConfig) -> None:
        """Initialize shim state without allocating local OPT FFN weights."""

        if str(config.activation_function).lower() != "relu":
            raise ShimUnavailableError(f"xpool OPT FFN shim supports only relu, got {config.activation_function!r}")
        if config.enable_bias is not True:
            raise ShimUnavailableError("xpool OPT FFN shim requires enable_bias=True")
        FfnShimModule.__init__(
            self,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            layer_kind=LayerKind.DENSE,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        gemm_output_zero_allocator: object | None = None,
    ) -> torch.Tensor:
        """Reject reduce-scatter fusion and dispatch the remote biased FFN."""

        if get_forward().mlp_reduce_scatter:
            raise ShimUnavailableError("xpool OPT FFN shim does not support SGLang reduce-scatter fusion")
        return super().forward(hidden_states, forward_batch, gemm_output_zero_allocator)


class XpoolOPTDecoderLayer(OPTDecoderLayer):
    """Replace the inline OPT FFN with a parameter-free CrossPool shim."""

    def __init__(
        self,
        config: OPTConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Build attention and norms while omitting local fc1/fc2 allocations."""

        match = OPT_LAYER_PREFIX_PATTERN.fullmatch(prefix)
        if match is None or int(match.group("layer_id")) != layer_id:
            raise ShimUnavailableError(f"xpool OPT decoder layer cannot derive layer id from prefix {prefix!r}")
        if quant_config is not None:
            raise ShimUnavailableError("xpool OPT shim does not support quantized attention")
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            bias=config.enable_bias,
            quant_config=None,
            prefix=add_prefix("self_attn", prefix),
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.mlp: XpoolOPTFFN = XpoolOPTFFN(layer_id=layer_id, config=config)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        """Preserve the pinned OPT pre-LN or post-LN branch around the shim."""

        return opt_decoder_forward(self, hidden_states, forward_batch)


def opt_decoder_forward(
    self: XpoolOPTDecoderLayer,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Mirror SGLang's OPT decoder layer with the remote FFN block."""

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


class OPTShimAdapter(SglangShimAdapter):
    """SGLang hooks and post-load policy for dense OPT models."""

    name = "opt"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Replace the OPT decoder layer and filter local FFN weights."""

        return (
            SglangHook(
                target="sglang.srt.models.opt.OPTDecoderLayer",
                handler=XpoolOPTDecoderLayer,
                kind=HookType.REPLACE,
            ),
            SglangHook(
                target="sglang.srt.models.opt.OPTForCausalLM.load_weights",
                handler=around_load_weights,
                kind=HookType.AROUND,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the dense OPT causal-language-model architecture."""

        return "OPTForCausalLM" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Reject unsupported OPT configuration before model construction."""

        config = model_runner.model_config.hf_config
        if not isinstance(config, OPTConfig):
            raise RuntimeError("xpool OPT runner did not resolve an OPTConfig")
        if str(config.activation_function).lower() != "relu":
            raise RuntimeError("xpool OPT adapter supports only relu activation")
        if config.enable_bias is not True:
            raise RuntimeError("xpool OPT adapter requires enable_bias=True")
        required_backend = required_attention_backend(config)
        if required_backend is not None:
            logger.warning(
                "OPT head_dim=%d is not supported by FlashInfer merge kernels; selecting the %s attention backend",
                config.hidden_size // config.num_attention_heads,
                required_backend,
            )
            get_context().override("xpool.opt", attention_backend=required_backend)

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require complete Dense shim coverage after OPT model loading."""

        model = model_runner.model
        if not isinstance(model, OPTForCausalLM):
            raise RuntimeError("xpool OPT model runner did not load an OPTForCausalLM model")
        config = model.config
        if not isinstance(config, OPTConfig):
            raise RuntimeError("xpool OPT model runner did not load an OPTConfig")
        layer_count = config.num_hidden_layers
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
            raise RuntimeError("xpool OPT model config has no positive integer num_hidden_layers")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * layer_count,
            allowed_shim_types=(XpoolOPTFFN,),
        )
        self.require_mlp_ownership(model, shims)
        if any(shim.hidden_size != config.hidden_size for shim in shims):
            raise RuntimeError("xpool OPT shim hidden size disagrees with config.hidden_size")
        leftover = [name for name, _ in model.named_parameters() if OPT_FFN_PARAM_PATTERN.search(name)]
        if leftover:
            raise RuntimeError("xpool OPT adapter left local FFN parameters: " + ", ".join(leftover))


def around_load_weights(
    original_fn: Callable[[OPTForCausalLM, Iterable[tuple[str, torch.Tensor]]], None],
    model: OPTForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Remove OPT decoder FFN tensors and freeze the inference-only model."""

    original_fn(model, filter_decoder_ffn_weights(weights, pattern=OPT_DECODER_FFN_WEIGHT_PATTERN))
    # Upstream OPTForCausalLM.forward is not decorated with torch.no_grad, so the
    # default loader leaves every parameter trainable. Full-decode CUDA Graph
    # capture runs with grad enabled and its KV writes would otherwise build an
    # autograd graph and fail inside the pool. Other SGLang models freeze their
    # parameters through copy_or_rebind_param; match that inference contract.
    model.requires_grad_(False)
    # SGLang resolves the prefill CUDA-graph body by descending ``.model`` until
    # it finds ``.layers``. OPT nests the decoder body at ``OPTModel.decoder``,
    # so expose it under the generic name without registering a duplicate module
    # alias in the state-dict namespace.
    language_model = getattr(model, "model", None)
    if isinstance(language_model, nn.Module) and not hasattr(language_model, "model"):
        decoder = getattr(language_model, "decoder", None)
        if isinstance(decoder, nn.Module):
            object.__setattr__(language_model, "model", decoder)
