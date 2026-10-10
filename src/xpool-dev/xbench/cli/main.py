"""Benchmark CLI bootstrap and dispatch."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from xbench.harness.serving.runner import continue_benchmarks
from xkit.config import XpoolDevConfig, init_global_config
from xpool.utils.cli import discover_cli_commands, register_cli_commands


def main(argv: Sequence[str] | None = None) -> int:
    """Install development policy and dispatch one command.

    Reports preserve source verdicts. Input, configuration and infrastructure
    failures return two; execution retains its owning runner's verdict.
    """
    parser = argparse.ArgumentParser(prog="xbench", description="CrossPool benchmark command", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    register_cli_commands(
        commands,
        discover_cli_commands("xbench.cli.subcommands", config_type=XpoolDevConfig),
        config_type=XpoolDevConfig,
    )
    options = parser.parse_args(None if argv is None else list(argv))
    try:
        if options.command == "run" and options.continue_dir is not None:
            forbidden = {"config_path", "xbench_catalog", "xbench_repetitions", "cache_root"} & vars(options).keys()
            if options.cases or not vars(options)["all"] or forbidden:
                raise ValueError("run --all --continue is exclusive with case and configuration overrides")
            return continue_benchmarks(
                options.continue_dir, rerun_failure=options.rerun_failure, fast_fail=options.fast_fail
            )
        if options.command == "run" and options.rerun_failure:
            raise ValueError("--rerun-failure requires run --all --continue")
        config = init_global_config(cli=vars(options))
        return options.handler(options, config)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool benchmark {options.command} failure: {error}", file=sys.stderr)
        return 2
