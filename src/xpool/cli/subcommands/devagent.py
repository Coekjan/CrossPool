"""Resident devagent subcommand."""

from __future__ import annotations

import argparse

from xpool.cli.command import RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.runtime.devagent import DevagentError, create_devagent


class DevagentCommand(RunnableCliCommand):
    """Run the selected resident devagent process."""

    name = "devagent"
    help = "run a resident devagent"
    order = 30

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add devagent arguments to ``parser``."""

        XpoolConfig.add_cli_args(parser)
        parser.add_argument(
            "--cuda-device",
            type=int,
            help="Configured CUDA device index owned by this agent",
        )

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Run the selected resident devagent process."""

        if args.cuda_device is None:
            raise DevagentError("xpool devagent requires --cuda-device when running a resident agent")
        create_devagent(args.cuda_device).run()
        return 0
