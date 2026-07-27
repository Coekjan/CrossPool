"""Daemon control-plane subcommands."""

from __future__ import annotations

import argparse
import json

import uvicorn

from xpool.cli.command import CliCommandGroup, RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.daemon import create_daemon
from xpool.service.daemon.app import DaemonFailure

type DaemonCheckPayload = dict[str, object]


class DaemonServer(uvicorn.Server):
    """Stop the Uvicorn lifecycle when the daemon watchdog fails."""

    def __init__(self, config: uvicorn.Config, failure: DaemonFailure) -> None:
        """Bind one Uvicorn server to the daemon failure latch."""

        super().__init__(config)
        self.failure = failure

    async def on_tick(self, counter: int) -> bool:
        """Combine Uvicorn's exit conditions with daemon-local failure."""

        return await super().on_tick(counter) or self.failure.failed


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

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Serve the daemon control-plane process."""

        app = create_daemon()
        server = DaemonServer(
            uvicorn.Config(app, host=config.daemon.host, port=config.daemon.port),
            app.state.daemon_failure,
        )
        server.run()
        return int(app.state.daemon_failure.failed)


class DaemonCheckCommand(RunnableCliCommand):
    """Check daemon readiness through the daemon API."""

    name = "check"
    help = "check daemon readiness"
    order = 20
    parent = "daemon"

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Check daemon readiness through the daemon API."""

        payload: DaemonCheckPayload
        client: XpoolClient | None = None
        try:
            client = XpoolClient()
            readiness = client.readiness()
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
