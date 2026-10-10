"""Top-level CrossPool command-line parser and dispatcher."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence

from pydantic import ValidationError

from xpool.config import CONFIG_REGISTRY, ConfigError, ConfigSource, XpoolConfig, init_global_config
from xpool.runtime.agent import AgentError
from xpool.service.client import XpoolClientError, XpoolDaemonError
from xpool.utils.cli import discover_cli_commands, register_cli_commands


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CrossPool command-line entry point.

    Args:
        argv: Optional argument vector excluding the executable name. When
            omitted, ``argparse`` reads process arguments from ``sys.argv``.

    Returns:
        Process-style exit code: ``0`` for success, ``1`` for unhealthy
        daemon readiness results, ``2`` for CLI/config validation errors, and
        ``20`` for daemon deployment failure after verified resource retirement.

    Side Effects:
        May print config or daemon diagnostics, validation errors, or start a
        resident process depending on the selected subcommand. Wrapped
        commands replace this process and retain the target exit status.
    """

    argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="xpool", description="CrossPool control tool")
    subparsers = parser.add_subparsers(dest="command")

    register_cli_commands(
        subparsers,
        discover_cli_commands("xpool.cli.subcommands", config_type=XpoolConfig),
        config_type=XpoolConfig,
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        arg_values = vars(args)
        config = init_global_config(
            cli={
                setting.name: arg_values[setting.name]
                for setting in CONFIG_REGISTRY
                if setting.cli is not None
                and ConfigSource.CLI in setting.allowed_sources
                and arg_values.get(setting.name) is not None
            }
        )
        return args.handler(args, config)
    except (
        ConfigError,
        AgentError,
        XpoolClientError,
        XpoolDaemonError,
        OSError,
        tomllib.TOMLDecodeError,
        ValidationError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
