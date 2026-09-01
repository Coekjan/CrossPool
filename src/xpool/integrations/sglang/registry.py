"""Automatic registry for repo-owned SGLang model adapters."""

from __future__ import annotations

from collections.abc import Sequence

from xpool.integrations.sglang.adapter import SglangShimAdapter
from xpool.utils.discovery import discover_concrete_subclasses

MODELS_PACKAGE = "xpool.integrations.sglang.models"


def discover_sglang_model_adapters(package_name: str) -> tuple[SglangShimAdapter, ...]:
    """Discover and instantiate SGLang model adapters from a package.

    Args:
        package_name: Importable package containing model adapter modules.

    Returns:
        Stable, name-validated adapter instances.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If an adapter module cannot be imported in strict mode,
            or if an adapter cannot be constructed or duplicates a name.
    """

    adapters: list[SglangShimAdapter] = []
    for adapter_class in discover_concrete_subclasses(package_name, SglangShimAdapter):
        try:
            adapters.append(adapter_class())
        except TypeError as error:
            raise RuntimeError(
                f"SGLang adapter {adapter_class.__module__}.{adapter_class.__name__} must be zero-argument"
            ) from error
    return sort_and_validate_adapters(adapters)


def sort_and_validate_adapters(adapters: Sequence[SglangShimAdapter]) -> tuple[SglangShimAdapter, ...]:
    """Sort adapters deterministically and reject duplicate adapter names.

    Args:
        adapters: Discovered adapter instances.

    Returns:
        Tuple sorted by adapter class module and class name.

    Raises:
        RuntimeError: If two adapters expose the same ``name``.
    """

    sorted_adapters = tuple(sorted(adapters, key=lambda adapter: (type(adapter).__module__, type(adapter).__name__)))
    seen: dict[str, SglangShimAdapter] = {}
    for adapter in sorted_adapters:
        if adapter.name in seen:
            previous = seen[adapter.name]
            raise RuntimeError(
                "duplicate SGLang adapter name "
                f"{adapter.name!r}: {type(previous).__module__}.{type(previous).__name__} and "
                f"{type(adapter).__module__}.{type(adapter).__name__}"
            )
        seen[adapter.name] = adapter
    return sorted_adapters
