"""Automatic registry for repo-owned SGLang model adapters."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import cast

from xpool.integrations.sglang.adapter import SglangModelAdapter

MODELS_PACKAGE = "xpool.integrations.sglang.models"
logger = logging.getLogger(__name__)


def discover_sglang_model_adapters(package_name: str, *, strict: bool = True) -> tuple[SglangModelAdapter, ...]:
    """Discover and instantiate SGLang model adapters from a package.

    Args:
        package_name: Importable package containing model adapter modules.
        strict: Whether child module import failures should abort discovery.

    Returns:
        Stable, name-validated adapter instances.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If an adapter module cannot be imported in strict mode,
            or if an adapter cannot be constructed or duplicates a name.
    """

    adapters: list[SglangModelAdapter] = []
    for module in iter_model_modules(package_name, strict=strict):
        for adapter_class in adapter_classes_in_module(module):
            try:
                adapters.append(adapter_class())
            except TypeError as exc:
                raise RuntimeError(
                    f"SGLang adapter {adapter_class.__module__}.{adapter_class.__name__} must be zero-argument"
                ) from exc
    return sort_and_validate_adapters(adapters)


def iter_model_modules(package_name: str, *, strict: bool = True) -> tuple[ModuleType, ...]:
    """Import non-private model adapter modules from a package tree.

    Args:
        package_name: Importable package whose children should be scanned recursively.
        strict: Whether child import failures should abort discovery.

    Returns:
        Imported module objects for non-private children and subpackages.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If a child module cannot be imported in strict mode.
    """

    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    modules: list[ModuleType] = []
    for module_info in pkgutil.walk_packages(
        package_path,
        package.__name__ + ".",
        onerror=lambda failed_package: logger.warning(
            "Skipping SGLang adapter package %s after import failure",
            failed_package,
        ),
    ):
        module_parts = module_info.name.removeprefix(package.__name__ + ".").split(".")
        if any(part.startswith("_") for part in module_parts):
            continue
        try:
            modules.append(importlib.import_module(module_info.name))
        except Exception as exc:
            if strict:
                raise RuntimeError(f"failed to import SGLang adapter module {module_info.name}: {exc}") from exc
            logger.warning(
                "Skipping SGLang adapter module %s after import failure: %s",
                module_info.name,
                exc,
                exc_info=True,
            )
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
    for member in inspect.getmembers(module, inspect.isclass):
        value = member[1]
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
