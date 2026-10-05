from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from multiprocessing.connection import Connection
from pathlib import Path

import httpx
import pytest

import xpool.cli.subcommands.daemon
import xpool.service.daemon.app
import xpool.service.daemon.control
from xkit.child import PythonChildProcess
from xkit.network import TcpEndpointReservation, TcpPortSpace
from xpool.cli import main
from xpool.fabric import FabricRole
from xpool.native import ABI_VERSION
from xpool.service.daemon.control import ControlPlane
from xpool.service.daemon.registration import AtnAgentRegistrationState
from xpool.utils.mps import MpsEndpoint, MpsProbeResult, MpsScope
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import (
    install_test_config,
    reset_global_config,
    synthetic_config,
    write_minimal_config,
)
from xtest.harness.support.service.daemon import atnagent_registration, instance_registration


@pytest.mark.usefixtures(reset_global_config.__name__)
def test_admitted_agent_can_publish_pre_join_retirement_boundary_after_close_begins() -> None:
    install_test_config(synthetic_config())
    agent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time; signal.signal(signal.SIGTERM, lambda *args: sys.exit(0)); "
            "print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert agent.stdout is not None and agent.stdout.readline().strip() == "ready"
    control = ControlPlane()
    scope = MpsScope(MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",)))
    control.mps_scope = scope
    identity = ProcUniqId(agent.pid)
    control.registrations.agent_startups[(FabricRole.ATNAGENT, 0)] = identity
    deadline = control.begin_close(deadline=time.monotonic() + 5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closed = executor.submit(control.close)
            control.register_atnagent(
                AtnAgentRegistrationState(device=0, abi_version=ABI_VERSION, pid=agent.pid, now=time.monotonic())
            )
            closed.result(timeout=5)
        assert agent.wait(timeout=1) == 0
        assert control.closed and scope.closed
        assert control.cleanup_deadline == deadline
    finally:
        if agent.poll() is None:
            agent.terminate()
            agent.wait(timeout=5)
        agent.stdout.close()


@dataclass(frozen=True, slots=True)
class ShutdownDaemonSpec:
    """CPU-only daemon setup with a shortened retirement clock."""

    config_path: Path
    uuids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShutdownDaemonResult:
    """Actual cleanup outcome retained independently of the deployment verdict."""

    exit_code: int
    deadline: float
    scope_closed: bool


def run_shutdown_daemon(connection: Connection, spec: ShutdownDaemonSpec) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda selected, role: None)
        patch.setattr(xpool.service.daemon.control, "MPS_CLEANUP_TIMEOUT_S", 0.4)
        app = xpool.cli.subcommands.daemon.create_daemon()
        control: ControlPlane = app.state.control_plane
        scope = MpsScope(MpsEndpoint(spec.uuids[:1]))
        # No controller or device work is created. Availability is the sole MPS
        # substitute; signals, HTTP and Instance identity observations are real.
        patch.setattr(scope, "probe", lambda: MpsProbeResult(True, None, "CPU shutdown check"))
        control.mps_scope = scope
        patch.setattr(control, "start", lambda: None)
        original_close = control.close

        def close(*, deadline: float | None = None) -> None:
            if not control.closed:
                connection.send(control.begin_close(deadline=deadline))
            original_close(deadline=deadline)

        def watchdog() -> None:
            if connection.poll() and connection.recv() == "fail":
                raise RuntimeError("injected watchdog failure")

        patch.setattr(control, "close", close)
        patch.setattr(control, "watchdog", watchdog)
        patch.setattr(xpool.cli.subcommands.daemon, "create_daemon", lambda: app)
        exit_code = main(["daemon", "serve", "--config", str(spec.config_path)])
        assert control.cleanup_deadline is not None
        assert scope.cleanup_deadline == control.cleanup_deadline
        connection.send(
            ShutdownDaemonResult(
                exit_code=exit_code,
                deadline=control.cleanup_deadline,
                scope_closed=scope.closed,
            )
        )


@pytest.mark.parametrize("failed", [False, True], ids=["signal", "watchdog-failure"])
def test_daemon_retains_control_until_instance_retires(failed: bool, tmp_path: Path) -> None:
    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    config_path = write_minimal_config(
        tmp_path / "xpool.toml", daemon_port=endpoint.port, atn_devices=(0,), ffn_devices=(1,)
    )
    uuids = ("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002")
    endpoint.release_for_spawn()
    daemon = PythonChildProcess(
        "shutdown-daemon",
        run_shutdown_daemon,
        ShutdownDaemonSpec(config_path, uuids),
        log_path=tmp_path / "daemon.log",
        import_paths=(Path(__file__).resolve().parents[5],),
    )
    serving = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        daemon.start()
        registration = instance_registration(pid=serving.pid)
        with httpx.Client(base_url=f"http://127.0.0.1:{endpoint.port}", timeout=1, trust_env=False) as client:
            startup_deadline = time.monotonic() + 10
            while True:
                try:
                    response = client.get("/health")
                except httpx.HTTPError:
                    assert daemon.process.is_alive(), daemon.tail()
                    assert time.monotonic() < startup_deadline, daemon.tail()
                    time.sleep(0.05)
                    continue
                assert response.status_code == HTTPStatus.OK
                break
            response = client.post("/instance/register", json=registration)
            assert response.status_code == HTTPStatus.NO_CONTENT, response.text
            daemon_pid = daemon.process.pid
            assert daemon_pid is not None
            if failed:
                daemon.send("fail")
            else:
                os.kill(daemon_pid, signal.SIGTERM)
            first_deadline = daemon.receive(float, timeout_seconds=5)
            os.kill(daemon_pid, signal.SIGINT)
            os.kill(daemon_pid, signal.SIGINT)
            # Expiry cannot replay a signal, force exit or close the scope.
            time.sleep(max(0, first_deadline - time.monotonic()) + 0.15)
            assert daemon.process.is_alive(), daemon.tail()
            assert serving.poll() is None
            assert client.get("/health").status_code == HTTPStatus.OK
            assert client.get("/config").status_code == HTTPStatus.OK
            assert client.post("/instance/register", json=registration).status_code == HTTPStatus.SERVICE_UNAVAILABLE
            assert (
                client.post("/atnagent/register", json=atnagent_registration(device=0)).status_code
                == HTTPStatus.SERVICE_UNAVAILABLE
            )
            serving.terminate()
            serving.wait(timeout=5)
        result = daemon.receive(ShutdownDaemonResult, timeout_seconds=5)
        daemon.wait(timeout_seconds=5)
        assert result == ShutdownDaemonResult(20 if failed else 0, first_deadline, True)
    finally:
        if serving.poll() is None:
            serving.terminate()
            serving.wait(timeout=5)
        # Only CPU stand-ins were created; no controller is eligible for kills.
        if daemon.process.is_alive():
            PythonChildProcess.terminate_all((daemon,))
        daemon.close()
        endpoint.close()


def test_daemon_bind_failure_retires_owned_resources(tmp_path: Path) -> None:
    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    config_path = write_minimal_config(
        tmp_path / "xpool.toml", daemon_port=endpoint.port, atn_devices=(0,), ffn_devices=(1,)
    )
    uuids = ("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002")
    # Keep the actual listener reservation held so Uvicorn takes its startup
    # failure path, which does not call the server's shutdown override.
    daemon = PythonChildProcess(
        "bind-failure-daemon",
        run_shutdown_daemon,
        ShutdownDaemonSpec(config_path, uuids),
        log_path=tmp_path / "daemon.log",
        import_paths=(Path(__file__).resolve().parents[5],),
    )
    try:
        daemon.start()
        first_deadline = daemon.receive(float, timeout_seconds=10)
        result = daemon.receive(ShutdownDaemonResult, timeout_seconds=5)
        daemon.wait(timeout_seconds=5)
        assert result == ShutdownDaemonResult(20, first_deadline, True)
    finally:
        if daemon.process.is_alive():
            PythonChildProcess.terminate_all((daemon,))
        daemon.close()
        endpoint.close()
