"""Command-line entry point for xpool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import uvicorn

from xpool.config import load_config
from xpool.daemon import create_app
from xpool.device_agent import DeviceAgentLaunchPlan
from xpool.runtime.mps import MpsHealthMonitor, MpsPreflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return args.handler(args)


def _run_daemon(args: argparse.Namespace) -> int:
    config = load_config(config_path=args.config, cli_overrides=_config_overrides(args))
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
        return 0

    if not mps.healthy:
        raise SystemExit(f"MPS preflight failed: {mps.message}")

    app = create_app(config, mps_monitor=MpsHealthMonitor(initial=mps))
    uvicorn.run(app, host=config.daemon.host, port=config.daemon.port)
    return 0


def _run_device_agent(args: argparse.Namespace) -> int:
    config = load_config(config_path=args.config, cli_overrides=_config_overrides(args))
    plan = DeviceAgentLaunchPlan.from_config(config)
    if args.check:
        print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    raise SystemExit("xpool device-agent resident runtime is not implemented in the skeleton yet")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xpool", description="xpool v3 control tool")
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
    parser.add_argument("--config", type=Path, help="Path to xpool TOML config")
    parser.add_argument("--daemon-host", dest="daemon_host")
    parser.add_argument("--daemon-port", dest="daemon_port", type=int)
    parser.add_argument("--attention-concurrency", dest="scheduler_attention_concurrency", type=int)
    parser.add_argument("--transport-concurrency", dest="scheduler_transport_concurrency", type=int)


def _config_overrides(args: argparse.Namespace) -> dict[str, object]:
    override_names = {
        "daemon_host",
        "daemon_port",
        "scheduler_attention_concurrency",
        "scheduler_transport_concurrency",
    }
    return {name: value for name, value in vars(args).items() if name in override_names and value is not None}
