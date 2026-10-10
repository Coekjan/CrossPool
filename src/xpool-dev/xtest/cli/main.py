"""Test CLI bootstrap and dispatch."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xkit.config import XpoolDevConfig, init_global_config
from xpool.utils.cli import discover_cli_commands, register_cli_commands


def main(argv: Sequence[str] | None = None) -> int:
    """Install development policy and dispatch one command.

    Reports preserve source verdicts. Input, configuration and infrastructure
    failures return two; execution retains its owning runner's verdict.
    """
    parser = argparse.ArgumentParser(prog="xtest", description="CrossPool test command", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    register_cli_commands(
        commands,
        discover_cli_commands("xtest.cli.subcommands", config_type=XpoolDevConfig),
        config_type=XpoolDevConfig,
    )
    options, selectors = parser.parse_known_args(None if argv is None else list(argv))
    if selectors and options.command not in {"list", "run"}:
        parser.error(f"unrecognized arguments: {' '.join(selectors)}")
    options.pytest_selectors = tuple(selectors)
    try:
        config = init_global_config(cli=vars(options))
        if options.command in {"list", "run"} and not (Path.cwd() / "pyproject.toml").is_file():
            raise ValueError("xtest must run from the xpool repository root")
        return options.handler(options, config)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool test {options.command} failure: {error}", file=sys.stderr)
        return 2
