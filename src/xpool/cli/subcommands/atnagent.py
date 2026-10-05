"""AtnAgent resident process command."""

from __future__ import annotations

import argparse

from xpool.cli.command import RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.runtime.agent import AgentError
from xpool.runtime.atnagent import AtnAgent


class AtnAgentCommand(RunnableCliCommand):
    """Run one configured AtnAgent process."""

    name = "atnagent"
    help = "run a resident AtnAgent"
    order = 30

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add AtnAgent command arguments."""

        parser.add_argument("--device", type=int, help="Configured ATN device index")

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Validate placement and run the selected AtnAgent."""

        if args.device is None:
            raise AgentError("xpool atnagent requires --device")
        if args.device not in config.atnagent_by_device:
            raise AgentError(f"device {args.device} is not configured for an AtnAgent")
        AtnAgent(device=args.device).run()
        return 0
