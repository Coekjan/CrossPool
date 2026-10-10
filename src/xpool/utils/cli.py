"""Shared command contracts for the CrossPool CLI."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, cast

from xpool.utils.config import ConfigModel
from xpool.utils.discovery import discover_concrete_subclasses

__all__ = [
    "CliCommand",
    "CliCommandGroup",
    "RunnableCliCommand",
    "discover_cli_commands",
    "register_cli_commands",
]


class CliCommand[C: ConfigModel](ABC):
    """Base class for argparse-backed CrossPool subcommands.

    Attributes:
        name: Command token used on the command line.
        help: Short help text shown by the parent parser.
        order: Stable sort key among siblings.
        parent: Optional parent command name for nested commands.
    """

    name: ClassVar[str]
    help: ClassVar[str]
    order: ClassVar[int] = 100
    parent: ClassVar[str | None] = None

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to ``parser``.

        Args:
            parser: Parser created for this command.

        Side Effects:
            Mutates ``parser`` by adding command arguments.
        """


class CliCommandGroup[C: ConfigModel](CliCommand[C]):
    """CLI command that owns nested subcommands instead of a direct handler.

    Attributes:
        subparser_dest: Attribute name used by argparse for the child command.
    """

    subparser_dest: ClassVar[str]


class RunnableCliCommand[C: ConfigModel](CliCommand[C], ABC):
    """CLI command that executes a handler after config resolution.

    Attributes:
        config_settings: Registered override names exposed by this command.
            None selects all; an empty tuple exposes no config flags.
    """

    config_settings: ClassVar[tuple[str, ...] | None] = None

    @abstractmethod
    def run(self, args: argparse.Namespace, config: C) -> int:
        """Run the command.

        Args:
            args: Parsed command-line arguments.
            config: Process-global CrossPool config resolved by the top-level CLI.

        Returns:
            Process-style exit code.

        Side Effects:
            Depends on the concrete command; may print diagnostics or run a
            resident process.
        """


def discover_cli_commands[C: ConfigModel](
    package_name: str,
    *,
    config_type: type[C],
) -> tuple[CliCommand[C], ...]:
    """Discover and instantiate concrete CrossPool CLI commands from a subcommands package.

    Args:
        package_name: Importable subcommands package containing command modules.
        config_type: Application-owned configuration type for this command family.

    Returns:
        Stable, name-validated command instances.

    Raises:
        ImportError: If the package itself cannot be imported.
        RuntimeError: If a command module cannot be imported,
            a command class cannot be constructed, or command names are invalid.
    """

    commands: list[CliCommand[C]] = []
    # Package discovery establishes the application-owned generic command family.
    command_classes = cast(tuple[type[CliCommand[C]], ...], discover_concrete_subclasses(package_name, CliCommand))
    for command_class in command_classes:
        try:
            commands.append(command_class())
        except TypeError as error:
            command_name = f"{command_class.__module__}.{command_class.__name__}"
            raise RuntimeError(f"xpool CLI command {command_name} must be zero-argument") from error
    return sort_and_validate_commands(commands)


def register_cli_commands[C: ConfigModel](
    subparsers: argparse._SubParsersAction,
    commands: Sequence[CliCommand[C]],
    *,
    config_type: type[C],
) -> None:
    """Register discovered CLI commands onto an argparse subparser collection.

    Args:
        subparsers: Top-level argparse subparser collection.
        commands: Commands returned by ``discover_cli_commands``.
        config_type: Configuration model supplying each command's registered options.

    Side Effects:
        Mutates ``subparsers`` by adding command parsers and handlers.
    """

    children_by_parent: dict[str | None, list[CliCommand[C]]] = {}
    for command in commands:
        children_by_parent.setdefault(command.parent, []).append(command)

    group_subparsers: dict[str | None, argparse._SubParsersAction] = {None: subparsers}

    def register_children(parent: str | None) -> None:
        parent_subparsers = group_subparsers[parent]
        for command in children_by_parent.get(parent, []):
            parser = parent_subparsers.add_parser(command.name, help=command.help)
            if isinstance(command, RunnableCliCommand):
                config_type.add_cli_args(parser, names=command.config_settings)
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


def sort_and_validate_commands[C: ConfigModel](commands: Sequence[CliCommand[C]]) -> tuple[CliCommand[C], ...]:
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
    seen_by_parent: dict[tuple[str | None, str], CliCommand[C]] = {}
    groups_by_name: dict[str, CliCommandGroup[C]] = {}
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
