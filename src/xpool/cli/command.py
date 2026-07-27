"""Shared command contracts for the xpool CLI."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar

from xpool.config import XpoolConfig


class CliCommand(ABC):
    """Base class for argparse-backed xpool subcommands.

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


class CliCommandGroup(CliCommand):
    """CLI command that owns nested subcommands instead of a direct handler.

    Attributes:
        subparser_dest: Attribute name used by argparse for the child command.
    """

    subparser_dest: ClassVar[str]


class RunnableCliCommand(CliCommand, ABC):
    """CLI command that executes a handler after config resolution."""

    @abstractmethod
    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Run the command.

        Args:
            args: Parsed command-line arguments.
            config: Process-global xpool config resolved by the top-level CLI.

        Returns:
            Process-style exit code.

        Side Effects:
            Depends on the concrete command; may print diagnostics or run a
            resident process.
        """
