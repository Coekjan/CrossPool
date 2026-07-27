"""Atomic ownership of one daemon-plus-Agent E2E process tree."""

from __future__ import annotations

import errno
import signal
import socket
import time
from pathlib import Path
from types import TracebackType

import httpx

from tests.harness.network import TcpEndpointReservation
from tests.harness.process import OwnedProcessGroup, signal_process_group, wait_for_process_group
from tests.harness.sglang.e2e import E2eLaunch
from xpool.service.wire import ReadinessSnapshot, ReadinessStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
DAEMON_STARTUP_TIMEOUT_SECONDS = 60.0
AGENT_REGISTRATION_TIMEOUT_SECONDS = 30.0
AGENT_SHUTDOWN_TIMEOUT_SECONDS = 75.0
DAEMON_SHUTDOWN_TIMEOUT_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.1


class DaemonPortConflict(RuntimeError):
    """Raised when a failed daemon attempt leaves its endpoint occupied."""


class XpoolCluster:
    """Own one atomically started daemon and complete configured Agent set."""

    def __init__(
        self,
        launch: E2eLaunch,
        processes: list[OwnedProcessGroup],
        client: httpx.Client,
        daemon_startup_seconds: float,
    ) -> None:
        self.launch = launch
        self.processes = processes
        self.client = client
        self.daemon_startup_seconds = daemon_startup_seconds
        self.closed = False

    @classmethod
    def start(cls, launch: E2eLaunch, endpoint: TcpEndpointReservation) -> XpoolCluster:
        """Start one healthy daemon and complete live Agent registration set."""

        processes: list[OwnedProcessGroup] = []
        client = httpx.Client(base_url=daemon_url(launch), timeout=POLL_INTERVAL_SECONDS)
        daemon_healthy = False
        daemon_started_at = time.monotonic()
        try:
            endpoint.release_for_spawn()
            processes.append(spawn_process("daemon", ["daemon", "serve"], launch=launch))
            wait_for_daemon_health(client, processes)
            daemon_startup_seconds = time.monotonic() - daemon_started_at
            daemon_healthy = True
            for agent in launch.config.atnagents:
                processes.append(
                    spawn_process(
                        f"atnagent-{agent.cuda_device}",
                        ["atnagent", "--cuda-device", str(agent.cuda_device)],
                        launch=launch,
                    )
                )
            for agent in launch.config.ffnagents:
                processes.append(
                    spawn_process(
                        f"ffnagent-{agent.cuda_device}",
                        ["ffnagent", "--cuda-device", str(agent.cuda_device)],
                        launch=launch,
                    )
                )
            wait_for_agent_registrations(client, processes, launch=launch)
        except BaseException as error:
            diagnostics = process_diagnostics(processes)
            client.close()
            cleanup_failures = terminate_processes(processes)
            close_process_logs(processes)
            endpoint_diagnostic: str | None = None
            endpoint_is_occupied = False
            if not daemon_healthy and not cleanup_failures:
                try:
                    endpoint_is_occupied = endpoint_occupied(launch)
                except OSError as endpoint_error:
                    endpoint_diagnostic = f"daemon endpoint probe failed: {endpoint_error}"
            if endpoint_is_occupied:
                raise DaemonPortConflict(
                    f"daemon endpoint remained occupied after failed startup: {daemon_url(launch)}\n{diagnostics}"
                ) from error
            details = [str(error)]
            if cleanup_failures:
                details.append("cleanup failures: " + "; ".join(cleanup_failures))
            if endpoint_diagnostic is not None:
                details.append(endpoint_diagnostic)
            if diagnostics:
                details.append(diagnostics)
            raise RuntimeError("failed to start xpool E2E cluster:\n" + "\n".join(details)) from error
        return cls(launch, processes, client, daemon_startup_seconds)

    def diagnostics(self) -> str:
        """Return bounded status and log tails for every owned process."""

        return process_diagnostics(self.processes)

    def close(self) -> None:
        """Drain the cluster normally and apply bounded process-group fallback."""

        if self.closed:
            return
        self.closed = True
        self.client.close()
        failures = close_cluster_processes(self.processes)
        diagnostics = process_diagnostics(self.processes) if failures else ""
        close_process_logs(self.processes)
        if failures:
            raise RuntimeError("xpool E2E cluster cleanup failed: " + "; ".join(failures) + "\n" + diagnostics)

    def __enter__(self) -> XpoolCluster:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            self.close()
        except RuntimeError as cleanup_error:
            if exc_value is not None:
                raise RuntimeError(f"{exc_value}\n{cleanup_error}") from exc_value
            raise
        return False


def daemon_url(launch: E2eLaunch) -> str:
    """Return the effective daemon URL for one materialized launch."""

    host = launch.config.daemon.host
    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{launch.config.daemon.port}"


def spawn_process(name: str, arguments: list[str], *, launch: E2eLaunch) -> OwnedProcessGroup:
    """Start one xpool CLI process with task-local configuration and logging."""

    return OwnedProcessGroup.spawn_logged(
        name,
        ["xpool", *arguments],
        cwd=REPO_ROOT,
        env=dict(launch.environment),
        log_path=launch.config_path.parent / f"{name}.log",
    )


def wait_for_daemon_health(client: httpx.Client, processes: list[OwnedProcessGroup]) -> None:
    """Wait until the daemon health endpoint responds successfully."""

    started_at = time.monotonic()
    deadline = started_at + DAEMON_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        raise_for_exited_process(processes)
        try:
            response = client.get("/health")
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)
    elapsed = time.monotonic() - started_at
    raise RuntimeError(f"timed out waiting for daemon health after {elapsed:.3f}s")


def wait_for_agent_registrations(
    client: httpx.Client,
    processes: list[OwnedProcessGroup],
    *,
    launch: E2eLaunch,
) -> None:
    """Wait for every configured AtnAgent and FfnAgent registration to be live."""

    expected_atn_devices = {agent.cuda_device for agent in launch.config.atnagents}
    expected_ffn_devices = {agent.cuda_device for agent in launch.config.ffnagents}
    deadline = time.monotonic() + AGENT_REGISTRATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        raise_for_exited_process(processes)
        try:
            response = client.get("/ready")
            if response.is_success:
                readiness = ReadinessSnapshot.model_validate(response.json())
                online_atn_devices = {
                    entry.cuda_device for entry in readiness.atnagents if entry.status is ReadinessStatus.ONLINE
                }
                online_ffn_devices = {
                    entry.cuda_device for entry in readiness.ffnagents if entry.status is ReadinessStatus.ONLINE
                }
                if online_atn_devices == expected_atn_devices and online_ffn_devices == expected_ffn_devices:
                    return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError("timed out waiting for complete Agent registration")


def raise_for_exited_process(processes: list[OwnedProcessGroup]) -> None:
    """Reject startup as soon as any acquired process exits."""

    for process in processes:
        returncode = process.process.poll()
        if returncode is not None:
            raise RuntimeError(f"{process.name} exited during startup with code {returncode}")


def close_cluster_processes(processes: list[OwnedProcessGroup]) -> list[str]:
    """Request coordinated Agent shutdown, then stop the daemon."""

    agents = [process for process in processes if process.name.startswith(("atnagent-", "ffnagent-"))]
    daemons = [process for process in processes if process.name == "daemon"]
    failures = stop_process_set(agents, timeout_seconds=AGENT_SHUTDOWN_TIMEOUT_SECONDS, accepted_returncodes={0})
    failures.extend(
        stop_process_set(
            daemons,
            timeout_seconds=DAEMON_SHUTDOWN_TIMEOUT_SECONDS,
            accepted_returncodes={0, -signal.SIGTERM},
        )
    )
    return failures


def stop_process_set(
    processes: list[OwnedProcessGroup],
    *,
    timeout_seconds: float,
    accepted_returncodes: set[int],
) -> list[str]:
    """Signal a related process set together and wait under one shared deadline."""

    for process in processes:
        signal_process_group(process.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    failures: list[str] = []
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        if not wait_for_process_group(process.process, remaining):
            try:
                process.terminate()
            except RuntimeError as error:
                failures.append(str(error))
                continue
        returncode = process.process.returncode
        if returncode is not None and returncode not in accepted_returncodes:
            failures.append(f"{process.name} exited with code {returncode}")
    return failures


def terminate_processes(processes: list[OwnedProcessGroup]) -> list[str]:
    """Apply bounded process-group termination to every partially acquired process."""

    failures: list[str] = []
    for process in reversed(processes):
        try:
            process.terminate()
        except RuntimeError as error:
            failures.append(str(error))
    return failures


def close_process_logs(processes: list[OwnedProcessGroup]) -> None:
    """Close every process-owned pipe and log handle."""

    for process in processes:
        process.close()


def process_diagnostics(processes: list[OwnedProcessGroup]) -> str:
    """Return bounded status and log tails without parsing process output."""

    sections = []
    for process in processes:
        returncode = process.process.poll()
        status = "running" if returncode is None else f"exited({returncode})"
        sections.append(f"--- {process.name}: {status}; log={process.log_path} ---\n{process.tail()}")
    return "\n".join(sections)


def endpoint_occupied(launch: E2eLaunch) -> bool:
    """Return whether the exact daemon endpoint still rejects rebinding as in use."""

    host = launch.config.daemon.host
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, launch.config.daemon.port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return True
            raise
    return False
