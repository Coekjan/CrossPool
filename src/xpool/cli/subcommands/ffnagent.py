"""FfnAgent resident process command."""

from __future__ import annotations

import argparse

from xpool.cli.command import RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.runtime.agent import AgentError
from xpool.runtime.ffnagent import FfnAgent


class FfnAgentCommand(RunnableCliCommand):
    """Run one configured FfnAgent process."""

    name = "ffnagent"
    help = "run a resident FfnAgent"
    order = 40

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add FfnAgent command arguments."""

        parser.add_argument("--device", type=int, help="Configured FFN device index")

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Validate placement and run the selected FfnAgent."""

        if args.device is None:
            raise AgentError("xpool ffnagent requires --device")
        if args.device not in config.ffnagent_by_device:
            raise AgentError(f"device {args.device} is not configured for an FfnAgent")
        FfnAgent(device=args.device).run()
        return 0
