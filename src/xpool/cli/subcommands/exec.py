"""Environment-only command execution for the attention MPS endpoint."""

from __future__ import annotations

import argparse
import os
from typing import NoReturn

from xpool.config import ConfigError, XpoolConfig
from xpool.utils.cli import RunnableCliCommand
from xpool.utils.device import normalize_environment, visible_uuids
from xpool.utils.mps import MpsEndpoint


class ExecCommand(RunnableCliCommand[XpoolConfig]):
    """Normalize complete visibility and prepare attention MPS, then exec the target."""

    name = "exec"
    help = "execute a command with the attention MPS environment"
    order = 30
    config_settings = ()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Accept one executable and opaque arguments, including its options."""

        parser.add_argument("executable", help="target executable")
        parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments passed unchanged to the target")

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> NoReturn:
        """Resolve TOML/environment placement and exec the target with inherited streams.

        The process identity and inherited process group remain unchanged.
        Invalid visibility raises ``ConfigError``; execution errors propagate
        as ``OSError``. This entry does not import or supervise the target.
        """

        try:
            normalize_environment()
            visibility = visible_uuids()
        except (ValueError, RuntimeError) as error:
            raise ConfigError(str(error)) from error
        if max(config.atn.devices) >= len(visibility):
            raise ConfigError("configured attention devices exceed the original process-visible device list")
        endpoint = MpsEndpoint(tuple(visibility[device] for device in config.atn.devices))
        os.environ.update(endpoint.environment())
        executable: str = args.executable
        arguments: list[str] = args.arguments
        os.execvp(executable, [executable, *arguments])
