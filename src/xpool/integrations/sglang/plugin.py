"""SGLang plugin entry point for CrossPool."""

from __future__ import annotations

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry

from xpool.config import init_global_config
from xpool.integrations.sglang.hooks.registry import discover_sglang_hooks

XPOOL_REQUIRED_HOOK_TARGETS: set[str] = set()


def install() -> None:
    """Discover and install every CrossPool hook in an SGLang process."""

    try:
        envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.set(True)
        init_global_config()
        hooks = discover_sglang_hooks()
        for hook in hooks:
            HookRegistry.register(hook.target, hook.handler, hook.kind)
        XPOOL_REQUIRED_HOOK_TARGETS.update(hook.target for hook in hooks)
        install_apply_hooks_guard()
    except Exception as error:
        raise SystemExit(f"xpool SGLang plugin failed to install: {error}") from error


def install_apply_hooks_guard() -> None:
    """Install a fatal postcondition check around SGLang hook application."""

    original_apply_hooks = HookRegistry.apply_hooks

    def guarded_apply_hooks(registry: type[object]) -> object:
        result = original_apply_hooks()
        verify_required_hooks_applied()
        return result

    setattr(HookRegistry, "apply_hooks", classmethod(guarded_apply_hooks))


def verify_required_hooks_applied() -> None:
    """Raise fatally if SGLang skipped any CrossPool-required hook target."""

    patched_targets = getattr(HookRegistry, "_patched", None)
    if not isinstance(patched_targets, set):
        raise SystemExit("xpool SGLang plugin cannot verify HookRegistry patched targets")
    missing_targets = XPOOL_REQUIRED_HOOK_TARGETS.difference(str(target) for target in patched_targets)
    if missing_targets:
        joined = ", ".join(sorted(missing_targets))
        raise SystemExit(f"xpool SGLang plugin failed to apply required hooks: {joined}")
