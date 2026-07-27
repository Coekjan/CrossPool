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

        parser.add_argument("--cuda-device", type=int, help="Configured ATN CUDA device index")

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Validate placement and run the selected AtnAgent."""

        if args.cuda_device is None:
            raise AgentError("xpool atnagent requires --cuda-device")
        if args.cuda_device not in config.atnagent_by_cuda_device:
            raise AgentError(f"CUDA device {args.cuda_device} is not configured for an AtnAgent")
        AtnAgent(cuda_device=args.cuda_device).run()
        return 0
