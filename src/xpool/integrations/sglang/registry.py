"""Automatic registry for repo-owned SGLang model adapters."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import cast

from xpool.integrations.sglang.adapter import SglangModelAdapter

MODELS_PACKAGE = "xpool.integrations.sglang.models"


def sglang_model_adapters() -> tuple[SglangModelAdapter, ...]:
    """Return model adapters installed by the xpool SGLang plugin."""

    return discover_sglang_model_adapters(MODELS_PACKAGE)


def discover_sglang_model_adapters(package_name: str) -> tuple[SglangModelAdapter, ...]:
    """Discover and instantiate SGLang model adapters from a package.

    Args:
        package_name: Importable package containing model adapter modules.

    Returns:
        Stable, name-validated adapter instances.

    Raises:
        ImportError: If the package or one of its modules cannot be imported.
        RuntimeError: If an adapter cannot be constructed or duplicates a name.
    """

    adapters: list[SglangModelAdapter] = []
    for module in iter_model_modules(package_name):
        for adapter_class in adapter_classes_in_module(module):
            adapters.append(instantiate_adapter(adapter_class))
    return sort_and_validate_adapters(adapters)


def iter_model_modules(package_name: str) -> tuple[ModuleType, ...]:
    """Import non-private model adapter modules from a package.

    Args:
        package_name: Importable package whose direct child modules should be scanned.

    Returns:
        Imported module objects for direct, non-package, non-private children.

    Raises:
        ImportError: If the package or a child module cannot be imported.
    """

    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    modules: list[ModuleType] = []
    for module_info in pkgutil.iter_modules(package_path, package.__name__ + "."):
        module_basename = module_info.name.rsplit(".", 1)[-1]
        if module_info.ispkg or module_basename.startswith("_"):
            continue
        modules.append(importlib.import_module(module_info.name))
    return tuple(modules)


def adapter_classes_in_module(module: ModuleType) -> tuple[type[SglangModelAdapter], ...]:
    """Return concrete adapter classes defined by one module.

    Args:
        module: Imported module to inspect.

    Returns:
        Concrete ``SglangModelAdapter`` subclasses whose ``__module__`` is the
        inspected module.
    """

    classes: list[type[SglangModelAdapter]] = []
    for _name, value in inspect.getmembers(module, inspect.isclass):
        if value is SglangModelAdapter:
            continue
        if value.__module__ != module.__name__:
            continue
        if not issubclass(value, SglangModelAdapter):
            continue
        if inspect.isabstract(value):
            continue
        classes.append(cast(type[SglangModelAdapter], value))
    return tuple(classes)


def instantiate_adapter(adapter_class: type[SglangModelAdapter]) -> SglangModelAdapter:
    """Instantiate a concrete adapter class.

    Args:
        adapter_class: Zero-argument ``SglangModelAdapter`` subclass.

    Returns:
        Adapter instance.

    Raises:
        RuntimeError: If the adapter requires constructor arguments.
    """

    try:
        return adapter_class()
    except TypeError as exc:
        raise RuntimeError(
            f"SGLang adapter {adapter_class.__module__}.{adapter_class.__name__} must be zero-argument"
        ) from exc


def sort_and_validate_adapters(adapters: Sequence[SglangModelAdapter]) -> tuple[SglangModelAdapter, ...]:
    """Sort adapters deterministically and reject duplicate adapter names.

    Args:
        adapters: Discovered adapter instances.

    Returns:
        Tuple sorted by adapter class module and class name.

    Raises:
        RuntimeError: If two adapters expose the same ``name``.
    """

    sorted_adapters = tuple(sorted(adapters, key=lambda adapter: (type(adapter).__module__, type(adapter).__name__)))
    seen: dict[str, SglangModelAdapter] = {}
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
