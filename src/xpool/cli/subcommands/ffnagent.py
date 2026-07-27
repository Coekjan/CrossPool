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

        parser.add_argument("--cuda-device", type=int, help="Configured FFN CUDA device index")

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Validate placement and run the selected FfnAgent."""

        if args.cuda_device is None:
            raise AgentError("xpool ffnagent requires --cuda-device")
        if args.cuda_device not in config.ffnagent_by_cuda_device:
            raise AgentError(f"CUDA device {args.cuda_device} is not configured for an FfnAgent")
        FfnAgent(cuda_device=args.cuda_device).run()
        return 0
