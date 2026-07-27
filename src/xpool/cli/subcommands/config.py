"""Configuration inspection subcommands."""

from __future__ import annotations

import argparse
import json

from xpool.cli.command import CliCommandGroup, RunnableCliCommand
from xpool.config import XpoolConfig


class ConfigCommand(CliCommandGroup):
    """Command group for resolved configuration inspection."""

    name = "config"
    help = "inspect resolved configuration"
    order = 10
    subparser_dest = "config_command"


class ConfigDumpCommand(RunnableCliCommand):
    """Dump resolved config and value provenance."""

    name = "dump"
    help = "dump resolved config and value sources"
    order = 10
    parent = "config"

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dump resolved config and value provenance."""

        payload = [
            {
                "name": record["name"],
                "value": record["value"],
                "source": record["source"].value,
            }
            for record in config.sources
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
