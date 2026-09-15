"""SGLang hook declarations and discovery."""

from xpool.integrations.sglang.hooks.registry import (
    SglangHook,
    SglangHookHandler,
    SglangHookSet,
    discover_sglang_hooks,
)

__all__ = [
    "SglangHook",
    "SglangHookHandler",
    "SglangHookSet",
    "discover_sglang_hooks",
]
