"""Development configuration inspection."""

import argparse

from xkit.config import XpoolDevConfig
from xpool.config import XpoolConfig
from xpool.utils.cli import CliCommandGroup, RunnableCliCommand


class ConfigCommand(CliCommandGroup[XpoolDevConfig]):
    """Group resolved development settings and their provenance."""

    name = "config"
    help = "inspect development configuration"
    order = 50
    subparser_dest = "config_command"


class ConfigDumpCommand(RunnableCliCommand[XpoolDevConfig]):
    """Show settings, sources, cache and selected files."""

    name = "dump"
    help = "dump effective settings and value sources"
    parent = "config"
    config_settings = ("config_path", "xbench_repetitions")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        print(config.record().model_dump_json(indent=2))
        return 0
