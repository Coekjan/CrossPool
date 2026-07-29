"""Daemon process fail-stop behavior after watchdog loss."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import httpx

import xpool.service.daemon.control
from tests.harness.runner.child import PythonChildProcess
from tests.harness.runner.network import TcpEndpointReservation, TcpPortSpace
from tests.harness.support.config import write_minimal_config
from xpool.cli import main

DAEMON_EXIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class FailingWatchdogSpec:
    """Configuration for one real daemon with a failing second watchdog tick."""

    config_path: Path


def run_failing_watchdog_daemon(connection: Connection, spec: FailingWatchdogSpec) -> None:
    calls = 0

    def watchdog(control_plane: xpool.service.daemon.control.ControlPlane) -> None:
        nonlocal calls
        del control_plane
        calls += 1
        if calls == 2:
            raise RuntimeError("injected watchdog failure")

    setattr(xpool.service.daemon.control.ControlPlane, "watchdog", watchdog)
    connection.send(main(["daemon", "serve", "--config", str(spec.config_path)]))


def test_daemon_exits_nonzero_after_watchdog_failure(tmp_path: Path) -> None:
    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    config_path = write_minimal_config(tmp_path / "xpool.toml", daemon_port=endpoint.port)
    endpoint.release_for_spawn()
    process = PythonChildProcess.start(
        "failing-watchdog-daemon",
        run_failing_watchdog_daemon,
        FailingWatchdogSpec(config_path),
        log_path=tmp_path / "daemon.log",
    )
    try:
        deadline = time.monotonic() + DAEMON_EXIT_TIMEOUT_SECONDS
        health_observed = False
        while time.monotonic() < deadline and process.process.is_alive():
            try:
                response = httpx.get(f"http://127.0.0.1:{endpoint.port}/health", timeout=0.2)
            except httpx.HTTPError:
                time.sleep(0.05)
                continue
            if response.status_code == 200:
                health_observed = True
                break
        assert health_observed, process.tail()
        assert process.receive(int, timeout_seconds=DAEMON_EXIT_TIMEOUT_SECONDS) == 1
        process.wait(timeout_seconds=DAEMON_EXIT_TIMEOUT_SECONDS)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()
        endpoint.close()
