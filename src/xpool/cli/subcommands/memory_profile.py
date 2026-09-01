"""Offline xpool memory-calibration command."""

from __future__ import annotations

import argparse

from xpool.cli.command import RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.memory import write_memory_calibration_profile
from xpool.runtime.agent import AgentError
from xpool.runtime.ffnagent.memory_profile import profile_ffn_memory


class MemoryProfileCommand(RunnableCliCommand):
    """Run the fixed FFN Calibration Corpus and publish its Profile."""

    name = "memory-profile"
    help = "profile xpool device-memory overhead"
    order = 50

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Produce and atomically publish one complete calibration Profile."""

        try:
            profile = profile_ffn_memory()
            write_memory_calibration_profile(profile)
        except RuntimeError as error:
            raise AgentError(f"xpool memory-profile failed: {error}") from error
        return 0
