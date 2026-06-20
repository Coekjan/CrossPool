"""Automatic registry for repo-owned devkit SGLang plugins."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterable
from typing import cast

from xpool.config import get_global_config

PLUGINS_PACKAGE = "xpool.devkit.sglang.plugins"
LOGGER = logging.getLogger(__name__)
PluginInstaller = Callable[[], None]


def install() -> None:
    """Install all enabled xpool devkit SGLang plugins.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
        RuntimeError: If an enabled plugin module does not expose a callable
            zero-argument ``install()`` entry point.

    Side Effects:
        Imports enabled plugin modules under ``xpool.devkit.sglang.plugins``
        and invokes their ``install()`` entry points. Disabled plugins are not
        imported.
    """

    for _plugin_name, installer in discover_sglang_devkit_plugins(PLUGINS_PACKAGE):
        installer()


def discover_sglang_devkit_plugins(
    package_name: str,
    *,
    strict: bool = True,
) -> tuple[tuple[str, PluginInstaller], ...]:
    """Discover enabled devkit SGLang plugins from a package tree.

    Args:
        package_name: Importable package containing devkit plugin modules.
        strict: Whether enabled child module import failures should abort
            discovery.

    Returns:
        Stable ``(plugin_name, installer)`` pairs for enabled plugins.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If an enabled plugin module cannot be imported in strict
            mode, or if an enabled plugin does not expose ``install()``.
    """

    config = get_global_config()
    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    plugins: list[tuple[str, PluginInstaller]] = []
    for module_info in sorted(
        pkgutil.walk_packages(
            package_path,
            package.__name__ + ".",
            onerror=lambda failed_package_name: LOGGER.warning(
                "Skipping devkit SGLang plugin package %s after import failure",
                failed_package_name,
            ),
        ),
        key=lambda item: item.name,
    ):
        module_parts = module_info.name.removeprefix(package.__name__ + ".").split(".")
        if any(part.startswith("_") for part in module_parts):
            continue
        plugin_name = module_parts[-1]
        plugin_config = getattr(config.debug, plugin_name, None)
        if getattr(plugin_config, "enable", False) is not True:
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"failed to import devkit SGLang plugin module {module_info.name}: {exc}") from exc
            LOGGER.warning(
                "Skipping devkit SGLang plugin module %s after import failure: %s",
                module_info.name,
                exc,
                exc_info=True,
            )
            continue
        entry = getattr(module, "install", None)
        if not callable(entry):
            raise RuntimeError(f"xpool devkit SGLang plugin {module.__name__} does not expose install()")
        plugins.append((plugin_name, cast(PluginInstaller, entry)))
    return tuple(plugins)
