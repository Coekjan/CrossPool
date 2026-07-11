"""Daemon control-plane subcommands."""

from __future__ import annotations

import argparse
import json

import uvicorn

from xpool.cli.command import CliCommandGroup, RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.daemon import create_daemon
from xpool.service.wire import ReadinessScope

type DaemonCheckPayload = dict[str, object]


class DaemonCommand(CliCommandGroup):
    """Command group for daemon process and readiness operations."""

    name = "daemon"
    help = "manage the daemon control plane"
    order = 20
    subparser_dest = "daemon_command"


class DaemonServeCommand(RunnableCliCommand):
    """Serve the daemon control-plane process."""

    name = "serve"
    help = "serve the daemon control plane"
    order = 10
    parent = "daemon"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add daemon-serve arguments to ``parser``."""

        XpoolConfig.add_cli_args(parser)

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Serve the daemon control-plane process."""

        app = create_daemon()
        uvicorn.run(app, host=config.daemon.host, port=config.daemon.port)
        return 0


class DaemonCheckCommand(RunnableCliCommand):
    """Check daemon readiness through the daemon API."""

    name = "check"
    help = "check daemon readiness"
    order = 20
    parent = "daemon"

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add daemon-check arguments to ``parser``."""

        XpoolConfig.add_cli_args(parser)
        parser.add_argument(
            "--scope",
            action="append",
            choices=tuple(scope.value for scope in ReadinessScope),
            default=[],
            help="participant readiness scope; repeat to select multiple scopes",
        )

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Check daemon readiness through the daemon API."""

        payload: DaemonCheckPayload
        client: XpoolClient | None = None
        try:
            client = XpoolClient()
            readiness = client.readiness(tuple(ReadinessScope(scope) for scope in args.scope))
        except (XpoolClientError, XpoolDaemonError) as exc:
            payload = {
                "ready": False,
                "readiness": None,
                "error": str(exc),
            }
            exit_code = 1
        else:
            payload = {
                "ready": readiness.ready,
                "readiness": readiness.model_dump(mode="json"),
                "error": None,
            }
            exit_code = 0 if readiness.ready else 1
        finally:
            if client is not None:
                client.close()
        payload |= {
            "daemon": {
                "host": config.daemon.host,
                "port": config.daemon.port,
            }
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return exit_code
