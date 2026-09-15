"""Automatic discovery of repo-owned SGLang hook declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sglang.srt.plugins.hook_registry import HookType

from xpool.utils.discovery import discover_concrete_subclasses

HOOKS_PACKAGE = "xpool.integrations.sglang.hooks"

# SGLang owns the variadic handler contract; CrossPool only forwards handlers
# to HookRegistry.register.
type SglangHookHandler = Callable[..., object] | type


@dataclass(frozen=True, slots=True)
class SglangHook:
    """One SGLang hook registration requested by an integration component."""

    target: str
    handler: SglangHookHandler
    kind: HookType


class SglangHookSet(ABC):
    """One discoverable owner of a cohesive SGLang hook manifest."""

    @abstractmethod
    def hooks(self) -> Sequence[SglangHook]:
        """Return the hook declarations owned by this component."""


def discover_sglang_hooks(package_name: str = HOOKS_PACKAGE) -> tuple[SglangHook, ...]:
    """Discover zero-argument HookSets and flatten their declarations."""

    hooks: list[SglangHook] = []
    for hook_set_class in discover_concrete_subclasses(package_name, SglangHookSet):
        try:
            hook_set = hook_set_class()
        except TypeError as error:
            raise RuntimeError(
                f"SGLang HookSet {hook_set_class.__module__}.{hook_set_class.__name__} must be zero-argument"
            ) from error
        hooks.extend(hook_set.hooks())
    return tuple(hooks)
