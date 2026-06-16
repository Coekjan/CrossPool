"""SGLang-side shim frontend skeleton."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_ReturnT = TypeVar("_ReturnT")


class ShimUnavailableError(RuntimeError):
    """Raised when the xpool shim is enabled but native runtime is not ready."""


def forward_via_xpool(
    original_fn: Callable[..., _ReturnT], module: object, *args: object, **kwargs: object
) -> _ReturnT:
    """Fail-closed placeholder for the graph-safe FFN shim frontend."""

    module_name = type(module).__module__
    class_name = type(module).__qualname__
    raise ShimUnavailableError(
        f"xpool shim is enabled, but the native shim frontend is not implemented yet for {module_name}.{class_name}"
    )
