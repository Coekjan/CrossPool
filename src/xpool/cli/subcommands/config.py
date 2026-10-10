"""Configuration inspection subcommands."""

from __future__ import annotations

import argparse

from pydantic import TypeAdapter

from xpool.config import ConfigSourceRecord, XpoolConfig
from xpool.utils.cli import CliCommandGroup, RunnableCliCommand


class ConfigCommand(CliCommandGroup[XpoolConfig]):
    """Command group for resolved configuration inspection."""

    name = "config"
    help = "inspect resolved configuration"
    order = 10
    subparser_dest = "config_command"


class ConfigDumpCommand(RunnableCliCommand[XpoolConfig]):
    """Dump resolved config and value provenance."""

    name = "dump"
    help = "dump resolved config and value sources"
    order = 10
    parent = "config"

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dump resolved config and value provenance."""

        print(TypeAdapter(tuple[ConfigSourceRecord, ...]).dump_json(config.sources, indent=2).decode())
        return 0
