"""Automatic discovery and argparse registration for xpool CLI commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from xpool.cli.command import CliCommand, CliCommandGroup, RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.utils.discovery import discover_concrete_subclasses

CLI_PACKAGE = "xpool.cli.subcommands"


def discover_cli_commands(package_name: str = CLI_PACKAGE) -> tuple[CliCommand, ...]:
    """Discover and instantiate concrete xpool CLI commands from a subcommands package.

    Args:
        package_name: Importable subcommands package containing command modules.

    Returns:
        Stable, name-validated command instances.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If a command module cannot be imported in strict mode,
            a command class cannot be constructed, or command names are invalid.
    """

    commands: list[CliCommand] = []
    for command_class in discover_concrete_subclasses(package_name, CliCommand):
        try:
            commands.append(command_class())
        except TypeError as error:
            command_name = f"{command_class.__module__}.{command_class.__name__}"
            raise RuntimeError(f"xpool CLI command {command_name} must be zero-argument") from error
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
            if isinstance(command, RunnableCliCommand):
                XpoolConfig.add_cli_args(parser)
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
