"""Daemon control-plane subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
from types import FrameType

import uvicorn

from xpool.cli.command import CliCommandGroup, RunnableCliCommand
from xpool.config import XpoolConfig
from xpool.service.client import XpoolClient, XpoolClientError, XpoolDaemonError
from xpool.service.daemon import create_daemon
from xpool.service.daemon.app import DaemonFailure
from xpool.service.daemon.control import ControlPlane

logger = logging.getLogger(__name__)

type DaemonCheckPayload = dict[str, object]


class DaemonServer(uvicorn.Server):
    """Keep the control listener usable throughout ordered daemon retirement."""

    def __init__(self, config: uvicorn.Config, failure: DaemonFailure, control: ControlPlane) -> None:
        """Bind signal and background-failure exit to the actual resource owner."""

        super().__init__(config)
        self.failure = failure
        self.control = control
        self.cleanup_task: asyncio.Task[None] | None = None

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        """Request one retirement without Uvicorn force-exit or signal replay."""

        self.control.begin_close()
        self.should_exit = True
        # Uvicorn's handler records signals for replay and enables force_exit on
        # a repeated SIGINT. Neither can bypass resource-owner cleanup.

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        """Retire partial startup under the same installed signal handlers."""

        with self.capture_signals():
            try:
                # Uvicorn skips shutdown when startup never sets ``started``;
                # its bind-error path also raises SystemExit after lifespan exit.
                await self._serve(sockets)
            except BaseException as error:
                self.failure.record(error)
                logger.exception("daemon failed pid=%s", os.getpid())
            finally:
                await asyncio.to_thread(self.control.close)

    async def on_tick(self, counter: int) -> bool:
        """Start owner retirement when the server or a background task stops."""

        should_exit = await super().on_tick(counter) or self.failure.failed
        if should_exit:
            self.control.begin_close()
        return should_exit

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        """Finish shared owner cleanup before Uvicorn closes HTTP listeners."""

        if self.cleanup_task is None:
            self.control.begin_close()
            self.cleanup_task = asyncio.create_task(asyncio.to_thread(self.control.close), name="xpool-daemon-cleanup")
        await asyncio.shield(self.cleanup_task)
        await super().shutdown(sockets=sockets)


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
            uvicorn.Config(app, host=config.daemon.host, port=config.daemon.port, access_log=False),
            app.state.daemon_failure,
            app.state.control_plane,
        )
        try:
            server.run()
        finally:
            logger.info("process stopped pid=%s", os.getpid())
        if not app.state.control_plane.closed:
            raise RuntimeError("daemon exited without verified resource cleanup")
        return 20 if app.state.daemon_failure.failed else 0


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
