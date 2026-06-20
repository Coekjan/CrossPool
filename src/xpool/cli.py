"""Command-line entry point for xpool."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence

import uvicorn
from pydantic import ValidationError

from xpool.config import (
    CONFIG_REGISTRY,
    ConfigError,
    ConfigSetting,
    ConfigSource,
    init_global_config,
)
from xpool.daemon import create_app
from xpool.device_agent import DeviceAgentLaunchPlan
from xpool.runtime.mps import MpsHealthMonitor, MpsPreflight

CLI_CONFIG_SETTINGS: tuple[ConfigSetting, ...] = tuple(
    setting for setting in CONFIG_REGISTRY if setting.cli is not None and ConfigSource.CLI in setting.allowed_sources
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xpool command-line entry point.

    Args:
        argv: Optional argument vector excluding the executable name. When
            omitted, ``argparse`` reads process arguments from ``sys.argv``.

    Returns:
        Process-style exit code: ``0`` for success, ``1`` for unhealthy check
        results, and ``2`` for CLI/config validation errors.

    Side Effects:
        May print JSON or validation errors, start a uvicorn daemon, or run MPS
        preflight depending on the selected subcommand.
    """

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except (ConfigError, OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_daemon(args: argparse.Namespace) -> int:
    arg_values = vars(args)
    config = init_global_config(
        cli_overrides={
            setting.name: arg_values[setting.name]
            for setting in CLI_CONFIG_SETTINGS
            if arg_values.get(setting.name) is not None
        }
    )
    mps = MpsPreflight.detect()
    if args.check:
        print(
            json.dumps(
                {
                    "config": config.model_dump(mode="json"),
                    "mps": mps.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if mps.healthy else 1

    if not mps.healthy:
        raise SystemExit(f"MPS preflight failed: {mps.message}")

    app = create_app(config, mps_monitor=MpsHealthMonitor(initial=mps))
    uvicorn.run(app, host=config.daemon.host, port=config.daemon.port)
    return 0


def _run_device_agent(args: argparse.Namespace) -> int:
    arg_values = vars(args)
    config = init_global_config(
        cli_overrides={
            setting.name: arg_values[setting.name]
            for setting in CLI_CONFIG_SETTINGS
            if arg_values.get(setting.name) is not None
        }
    )
    plan = DeviceAgentLaunchPlan.from_config(config)
    if args.check:
        payload = plan.model_dump(mode="json")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if plan.mps.healthy else 1

    raise SystemExit("xpool device-agent resident runtime is not implemented in the skeleton yet")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xpool", description="xpool control tool")
    subparsers = parser.add_subparsers(dest="command")

    daemon = subparsers.add_parser("daemon", help="run or check the daemon control plane")
    _add_config_args(daemon)
    daemon.add_argument("--check", action="store_true", help="Validate config and exit")
    daemon.set_defaults(handler=_run_daemon)

    device_agent = subparsers.add_parser("device-agent", help="run or check device-agent launch plan")
    _add_config_args(device_agent)
    device_agent.add_argument("--check", action="store_true", help="Validate launch plan and exit")
    device_agent.set_defaults(handler=_run_device_agent)

    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    for setting in CLI_CONFIG_SETTINGS:
        parser.add_argument(
            setting.cli or "",
            dest=setting.name,
            type=int if setting.parser == "int" else str,
            help=setting.description,
        )
