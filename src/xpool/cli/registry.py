"""Automatic discovery and argparse registration for xpool CLI commands."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import pkgutil
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import cast

from xpool.cli.command import CliCommand, CliCommandGroup, RunnableCliCommand

CLI_PACKAGE = "xpool.cli.subcommands"
logger = logging.getLogger(__name__)


def discover_cli_commands(package_name: str = CLI_PACKAGE, *, strict: bool = True) -> tuple[CliCommand, ...]:
    """Discover and instantiate concrete xpool CLI commands from a subcommands package.

    Args:
        package_name: Importable subcommands package containing command modules.
        strict: Whether child module import failures should abort discovery.

    Returns:
        Stable, name-validated command instances.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If a command module cannot be imported in strict mode,
            a command class cannot be constructed, or command names are invalid.
    """

    commands: list[CliCommand] = []
    for module in iter_cli_modules(package_name, strict=strict):
        for command_class in command_classes_in_module(module):
            try:
                commands.append(command_class())
            except TypeError as exc:
                command_name = f"{command_class.__module__}.{command_class.__name__}"
                raise RuntimeError(f"xpool CLI command {command_name} must be zero-argument") from exc
    return sort_and_validate_commands(commands)


def register_cli_commands(subparsers: argparse._SubParsersAction, commands: Sequence[CliCommand]) -> None:
    """Register discovered CLI commands onto an argparse subparser collection.

    Args:
        subparsers: Top-level argparse subparser collection.
        commands: Commands returned by ``discover_cli_commands``.

    Side Effects:
        Mutates ``subparsers`` by adding command parsers and handlers.
    """

    children_by_parent: dict[str | None, list[CliCommand]] = {}
    for command in commands:
        children_by_parent.setdefault(command.parent, []).append(command)

    group_subparsers: dict[str | None, argparse._SubParsersAction] = {None: subparsers}

    def register_children(parent: str | None) -> None:
        parent_subparsers = group_subparsers[parent]
        for command in children_by_parent.get(parent, []):
            parser = parent_subparsers.add_parser(command.name, help=command.help)
            command.configure_parser(parser)
            if isinstance(command, CliCommandGroup):
                group_subparsers[command.name] = parser.add_subparsers(
                    dest=command.subparser_dest,
                    required=True,
                )
                register_children(command.name)
            elif isinstance(command, RunnableCliCommand):
                parser.set_defaults(handler=command.run)
            else:
                raise RuntimeError(f"unsupported xpool CLI command type: {type(command).__name__}")

    register_children(None)


def iter_cli_modules(package_name: str, *, strict: bool = True) -> tuple[ModuleType, ...]:
    """Import non-private CLI command modules from a subcommands package tree.

    Args:
        package_name: Importable subcommands package whose children should be scanned.
        strict: Whether child import failures should abort discovery.

    Returns:
        Imported module objects for command modules.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If a child module cannot be imported in strict mode.
    """

    package = importlib.import_module(package_name)
    package_path = cast(Iterable[str], getattr(package, "__path__"))
    modules: list[ModuleType] = []
    for module_info in sorted(
        pkgutil.walk_packages(
            package_path,
            package.__name__ + ".",
            onerror=lambda failed_package_name: logger.warning(
                "Skipping xpool CLI package %s after import failure",
                failed_package_name,
            ),
        ),
        key=lambda item: item.name,
    ):
        module_parts = module_info.name.removeprefix(package.__name__ + ".").split(".")
        if any(part.startswith("_") for part in module_parts):
            continue
        try:
            modules.append(importlib.import_module(module_info.name))
        except Exception as exc:
            if strict:
                raise RuntimeError(f"failed to import xpool CLI module {module_info.name}: {exc}") from exc
            logger.warning(
                "Skipping xpool CLI module %s after import failure: %s",
                module_info.name,
                exc,
                exc_info=True,
            )
    return tuple(modules)


def command_classes_in_module(module: ModuleType) -> tuple[type[CliCommand], ...]:
    """Return concrete command classes defined by one module.

    Args:
        module: Imported module to inspect.

    Returns:
        Concrete ``CliCommand`` subclasses whose ``__module__`` is the inspected
        module.
    """

    classes: list[type[CliCommand]] = []
    for member in inspect.getmembers(module, inspect.isclass):
        value = member[1]
        if value in (CliCommand, CliCommandGroup, RunnableCliCommand):
            continue
        if value.__module__ != module.__name__:
            continue
        if not issubclass(value, CliCommand):
            continue
        if inspect.isabstract(value):
            continue
        classes.append(cast(type[CliCommand], value))
    return tuple(classes)


def sort_and_validate_commands(commands: Sequence[CliCommand]) -> tuple[CliCommand, ...]:
    """Sort commands deterministically and reject invalid command trees.

    Args:
        commands: Discovered command instances.

    Returns:
        Tuple sorted by parent, order, module, and class name.

    Raises:
        RuntimeError: If names are missing, duplicate, or reference a missing
            parent group.
    """

    sorted_commands = tuple(
        sorted(
            commands,
            key=lambda command: (
                command.parent or "",
                command.order,
                type(command).__module__,
                type(command).__name__,
            ),
        )
    )
    seen_by_parent: dict[tuple[str | None, str], CliCommand] = {}
    groups_by_name: dict[str, CliCommandGroup] = {}
    for command in sorted_commands:
        if not command.name:
            raise RuntimeError(f"xpool CLI command {type(command).__module__}.{type(command).__name__} has no name")
        if not command.help:
            command_name = f"{type(command).__module__}.{type(command).__name__}"
            raise RuntimeError(f"xpool CLI command {command_name} has no help text")
        sibling_key = (command.parent, command.name)
        if sibling_key in seen_by_parent:
            previous = seen_by_parent[sibling_key]
            raise RuntimeError(
                "duplicate xpool CLI command "
                f"{command.name!r} under parent {command.parent!r}: "
                f"{type(previous).__module__}.{type(previous).__name__} and "
                f"{type(command).__module__}.{type(command).__name__}"
            )
        seen_by_parent[sibling_key] = command
        if isinstance(command, CliCommandGroup):
            if command.name in groups_by_name:
                previous_group = groups_by_name[command.name]
                raise RuntimeError(
                    "duplicate xpool CLI command group "
                    f"{command.name!r}: {type(previous_group).__module__}.{type(previous_group).__name__} and "
                    f"{type(command).__module__}.{type(command).__name__}"
                )
            groups_by_name[command.name] = command

    for command in sorted_commands:
        if command.parent is None:
            continue
        if command.parent not in groups_by_name:
            raise RuntimeError(
                f"xpool CLI command {type(command).__module__}.{type(command).__name__} "
                f"references missing parent group {command.parent!r}"
            )
    return sorted_commands
