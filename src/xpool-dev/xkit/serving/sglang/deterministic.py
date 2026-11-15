"""Opt-in deterministic Decode reduction for SGLang serving comparisons."""

from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def install() -> None:
    """Install the explicitly selected policy before engine Graph capture.

    The current policy controls FA3 MLA Decode reduction. Other attention
    paths retain their implementation; this does not enable SGLang's global
    deterministic-inference mode or constrain independent FFN processes.
    """

    selected = {name.strip() for name in (envs.SGLANG_PLUGINS.get() or "").split(",")}
    if "xkit_deterministic" not in selected:
        return

    target = "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode"
    HookRegistry.register(target, around_decode, HookType.AROUND)
    original_apply_hooks = HookRegistry.apply_hooks

    def apply_hooks(registry: type[HookRegistry]) -> None:
        original_apply_hooks()
        # SGLang logs individual hook-application errors and continues.
        if target not in registry._patched:
            raise SystemExit(f"xkit deterministic hook was not applied: {target}")

    setattr(HookRegistry, "apply_hooks", classmethod(apply_hooks))


def around_decode[**P, R](
    original: Callable[Concatenate[FlashAttentionBackend, P], R],
    backend: FlashAttentionBackend,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Apply one Decode policy during eager execution and Graph capture."""

    if backend.fa_impl_ver != 3 or not backend.use_mla:
        return original(backend, *args, **kwargs)
    previous = backend.num_splits
    backend.num_splits = 1
    try:
        return original(backend, *args, **kwargs)
    finally:
        backend.num_splits = previous
