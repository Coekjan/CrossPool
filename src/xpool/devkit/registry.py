"""Role-aware discovery for enabled CrossPool Devkit observers."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import cast

from xpool.bootstrap import get_runtime_role
from xpool.config import get_global_config
from xpool.native import RuntimeRole
from xpool.utils.discovery import walk_package_modules

DEVKIT_PACKAGE = "xpool.devkit"
logger = logging.getLogger(__name__)
type ObserverInstaller = Callable[[], None]


def install(package_name: str = DEVKIT_PACKAGE) -> None:
    """Install enabled observers from one repo-owned package tree.

    Args:
        package_name: Importable package tree containing observer modules.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If enabled modules collide by config name, fail import,
            or expose invalid role or installer contracts.

    Side Effects:
        Imports enabled leaves and invokes role-matching installers in stable
        module-name order.
    """

    config = get_global_config()
    runtime_role = get_runtime_role()
    configured_module_by_name: dict[str, str] = {}
    for module_info in walk_package_modules(package_name):
        if module_info.ispkg:
            continue
        observer_name = module_info.name.rsplit(".", maxsplit=1)[-1]
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
        except Exception as error:
            raise RuntimeError(f"failed to import devkit observer module {module_info.name}: {error}") from error
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
        logger.debug("installing devkit observer=%s", module_info.name)
        cast(ObserverInstaller, entry)()
