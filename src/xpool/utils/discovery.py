"""Stable discovery of concrete classes in repo-owned package trees."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from typing import cast


def walk_package_modules(package_name: str) -> tuple[pkgutil.ModuleInfo, ...]:
    """Return public child modules in stable qualified-name order.

    Args:
        package_name: Importable package whose descendants should be scanned.

    Returns:
        Standard-library module records for public packages and leaves.

    Raises:
        ImportError: If the root package cannot be imported.
        RuntimeError: If package traversal cannot import a child package.
    """

    def fail_package_import(failed_package_name: str) -> None:
        raise RuntimeError(f"failed to import package {failed_package_name}")

    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    modules = (
        module_info
        for module_info in pkgutil.walk_packages(
            package_path,
            package.__name__ + ".",
            onerror=fail_package_import,
        )
        if not any(part.startswith("_") for part in module_info.name.removeprefix(package.__name__ + ".").split("."))
    )
    return tuple(sorted(modules, key=lambda module_info: module_info.name))


def discover_concrete_subclasses[T](
    package_name: str,
    base_class: type[T],
) -> tuple[type[T], ...]:
    """Import and return concrete subclasses defined by a package tree.

    Args:
        package_name: Importable package containing subclass definitions.
        base_class: Base class whose concrete descendants should be returned.

    Returns:
        Classes in stable module-name and class-name order.

    Raises:
        ImportError: If the root package cannot be imported.
        RuntimeError: If a discovered module cannot be imported.
    """

    classes: list[type[T]] = []
    for module_info in walk_package_modules(package_name):
        try:
            module = importlib.import_module(module_info.name)
        except Exception as error:
            raise RuntimeError(f"failed to import module {module_info.name}: {error}") from error
        for _, value in inspect.getmembers(module, inspect.isclass):
            if value is base_class or value.__module__ != module.__name__:
                continue
            if not issubclass(value, base_class) or inspect.isabstract(value):
                continue
            classes.append(cast(type[T], value))
    return tuple(sorted(classes, key=lambda value: (value.__module__, value.__name__)))
