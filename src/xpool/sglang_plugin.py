"""SGLang plugin entry point for xpool shim installation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from xpool.shim import forward_via_xpool

_ReturnT = TypeVar("_ReturnT")

FFN_FORWARD_TARGETS = (
    "sglang.srt.models.deepseek_v2.DeepseekV2MLP.forward",
    "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward",
    "sglang.srt.models.qwen2.Qwen2MLP.forward",
    "sglang.srt.models.glm4_moe_lite.Glm4MoeLiteMLP.forward",
    "sglang.srt.models.glm4_moe_lite.Glm4MoeLiteSparseMoeBlock.forward",
)


def install() -> None:
    """Register SGLang hooks for xpool FFN shim targets."""

    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    except Exception as exc:
        raise RuntimeError("xpool SGLang plugin requires SGLang's hook registry") from exc

    for target in FFN_FORWARD_TARGETS:
        HookRegistry.register(target, _around_ffn_forward, HookType.AROUND)


def _around_ffn_forward(
    original_fn: Callable[..., _ReturnT], module: object, *args: object, **kwargs: object
) -> _ReturnT:
    return forward_via_xpool(original_fn, module, *args, **kwargs)
