"""Automatic registry for role-aware xpool devkit observers."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterable
from typing import cast

from xpool.abi import RuntimeRole
from xpool.bootstrap import get_runtime_role
from xpool.config import get_global_config

DEVKIT_PACKAGE = "xpool.devkit"
logger = logging.getLogger(__name__)
type ObserverInstaller = Callable[[], None]


def install() -> None:
    """Install enabled devkit observers applicable to this runtime role.

    Raises:
        RuntimeError: If bootstrap is incomplete or an enabled observer has an
            invalid role declaration or install entry point.
        ImportError: If the devkit package cannot be imported.

    Side Effects:
        Imports enabled observer modules and invokes applicable installers in
        stable module-name order.
    """

    for observer_name, installer in discover_devkit_observers(DEVKIT_PACKAGE):
        logger.debug("Installing xpool devkit observer %s", observer_name)
        installer()


def discover_devkit_observers(
    package_name: str,
    *,
    strict: bool = True,
) -> tuple[tuple[str, ObserverInstaller], ...]:
    """Discover enabled observers applicable to the current runtime role.

    Args:
        package_name: Importable package tree containing observer modules.
        strict: Whether enabled observer import failures abort discovery.

    Returns:
        Stable ``(qualified_name, installer)`` pairs.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If enabled modules collide by config name, fail import in
            strict mode, or expose invalid role or installer contracts.
    """

    config = get_global_config()
    runtime_role = get_runtime_role()
    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    observers: list[tuple[str, ObserverInstaller]] = []
    configured_module_by_name: dict[str, str] = {}
    for module_info in sorted(
        pkgutil.walk_packages(
            package_path,
            package.__name__ + ".",
            onerror=lambda failed_package_name: logger.warning(
                "Skipping devkit package %s after import failure",
                failed_package_name,
            ),
        ),
        key=lambda item: item.name,
    ):
        if module_info.ispkg:
            continue
        module_parts = module_info.name.removeprefix(package.__name__ + ".").split(".")
        if any(part.startswith("_") for part in module_parts):
            continue
        observer_name = module_parts[-1]
        observer_config = getattr(config.debug, observer_name, None)
        if getattr(observer_config, "enable", False) is not True:
            continue
        previous_module = configured_module_by_name.setdefault(observer_name, module_info.name)
        if previous_module != module_info.name:
            raise RuntimeError(
                f"xpool devkit observer name {observer_name} is ambiguous between "
                f"{previous_module} and {module_info.name}"
            )
        try:
            module = importlib.import_module(module_info.name)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"failed to import devkit observer module {module_info.name}: {exc}") from exc
            logger.warning(
                "Skipping devkit observer module %s after import failure: %s",
                module_info.name,
                exc,
                exc_info=True,
            )
            continue
        runtime_roles = getattr(module, "runtime_roles", None)
        if (
            not isinstance(runtime_roles, frozenset)
            or not runtime_roles
            or not all(isinstance(role, RuntimeRole) for role in runtime_roles)
        ):
            raise RuntimeError(
                f"xpool devkit observer {module.__name__} must expose a non-empty frozenset[RuntimeRole] runtime_roles"
            )
        if runtime_role not in runtime_roles:
            continue
        entry = getattr(module, "install", None)
        if not callable(entry):
            raise RuntimeError(f"xpool devkit observer {module.__name__} does not expose install()")
        observers.append((module_info.name, cast(ObserverInstaller, entry)))
    return tuple(observers)
