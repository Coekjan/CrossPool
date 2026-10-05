"""Atomic ownership of one daemon-plus-Agent E2E process tree."""

from __future__ import annotations

import errno
import logging
import signal
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import httpx

from xkit.network import TcpEndpointConflict, TcpEndpointReservation
from xkit.process import OwnedProcessGroup, wait_for_process_group
from xkit.serving.readiness import ReadinessEvidence, ReadinessTimeout
from xkit.task import TaskCleanupScope, get_task_root
from xpool.config import XpoolConfig
from xpool.service.wire import ReadinessSnapshot, ReadinessStatus
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S
from xpool.utils.sighandler import defer_signal_exceptions

DAEMON_STARTUP_TIMEOUT_SECONDS = 60.0
AGENT_REGISTRATION_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 0.1
CONTROL_PLANE_HTTP_TIMEOUT_SECONDS = 1.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class XpoolClusterLaunch:
    """Concrete config and environment for one installed CrossPool process tree."""

    config: XpoolConfig
    config_path: Path
    environment: Mapping[str, str]
    cwd: Path


class XpoolCluster:
    """Own one atomically started daemon and complete configured Agent set."""

    def __init__(self, launch: XpoolClusterLaunch) -> None:
        """Create the retained owner before its first process is launched."""

        self.launch = launch
        self.processes: list[OwnedProcessGroup] = []
        self.client = httpx.Client(base_url=daemon_url(launch), timeout=CONTROL_PLANE_HTTP_TIMEOUT_SECONDS)
        self.daemon_startup_seconds = 0.0
        self.startup_attempted = False
        self.closed = False
        self.task_scope: TaskCleanupScope | None = None
        self.cleanup_lock = threading.Lock()
        self.cleanup_deadline: float | None = None
        self.cleanup_executor: ThreadPoolExecutor | None = None
        self.cleanup_future: Future[None] | None = None

    def start(
        self,
        endpoint: TcpEndpointReservation,
        *,
        startup_deadline: float | None = None,
        enclosing_scope: TaskCleanupScope | None = None,
    ) -> None:
        """Retain protection before starting the daemon and all configured Agents.

        Failed startup joins the same ordered close operation. The retained
        scope covers controller creation and partial rollback; ownership is
        released only after verified daemon cleanup.
        An enclosing System retains its own scope through complete shutdown;
        only standalone clusters register and complete a task scope here.
        """

        if self.closed or self.startup_attempted or self.cleanup_future is not None:
            raise RuntimeError("cluster startup was already attempted or retired")
        self.startup_attempted = True
        launch = self.launch
        processes = self.processes
        client = self.client
        daemon_healthy = False
        daemon_started_at = time.monotonic()
        try:
            root = get_task_root()
            if root is not None and enclosing_scope is None:
                with defer_signal_exceptions():
                    self.task_scope = root.register_scope()
                root.activate()
            self.check_startup_active()
            endpoint.release_for_spawn()
            with defer_signal_exceptions():
                processes.append(spawn_process("daemon", ["daemon", "serve"], launch=launch))
            readiness_directory = launch.config_path.parent / "readiness"
            wait_for_daemon_health(
                client,
                processes,
                url=f"{daemon_url(launch)}/health",
                evidence_path=readiness_directory / "daemon-health.json",
                startup_deadline=startup_deadline,
                check_startup_active=self.check_startup_active,
            )
            self.daemon_startup_seconds = time.monotonic() - daemon_started_at
            daemon_healthy = True
            for agent in launch.config.atnagents:
                self.check_startup_active()
                with defer_signal_exceptions():
                    processes.append(
                        spawn_process(
                            f"atnagent-{agent.device}",
                            ["atnagent", "--device", str(agent.device)],
                            launch=launch,
                        )
                    )
            for agent in launch.config.ffnagents:
                self.check_startup_active()
                with defer_signal_exceptions():
                    processes.append(
                        spawn_process(
                            f"ffnagent-{agent.device}",
                            ["ffnagent", "--device", str(agent.device)],
                            launch=launch,
                        )
                    )
            wait_for_agent_registrations(
                client,
                processes,
                launch=launch,
                url=f"{daemon_url(launch)}/ready",
                evidence_path=readiness_directory / "agent-registration.json",
                startup_deadline=startup_deadline,
                check_startup_active=self.check_startup_active,
            )
        except BaseException as error:
            diagnostics = process_diagnostics(processes)
            try:
                self.close()
            except Exception as cleanup_error:
                if not self.closed:
                    cleanup_error.add_note(diagnostics)
                    raise RuntimeError(
                        f"cluster startup failed ({error}); ordered cleanup failed: {cleanup_error}"
                    ) from error
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if not daemon_healthy:
                try:
                    endpoint.reacquire()
                except OSError as endpoint_error:
                    if endpoint_error.errno == errno.EADDRINUSE:
                        conflict = TcpEndpointConflict(((endpoint.host, endpoint.port),))
                        conflict.add_note(diagnostics)
                        raise conflict from error
                    endpoint_error.add_note("failed to reacquire the daemon endpoint after startup failure")
                    raise endpoint_error from error
            details = [str(error)]
            if diagnostics:
                details.append(diagnostics)
            raise RuntimeError("failed to start xpool E2E cluster:\n" + "\n".join(details)) from error

    def diagnostics(self) -> str:
        """Return bounded status and log tails for every owned process."""

        return process_diagnostics(self.processes)

    def check_startup_active(self) -> None:
        """Stop creation and readiness waiting at the owner's retirement edge."""

        root = get_task_root()
        if self.cleanup_deadline is not None or (root is not None and root.sealed):
            raise InterruptedError("cluster startup is retiring")

    def close(self, *, deadline: float | None = None) -> None:
        """Join one ordered cleanup operation without refreshing its deadline.

        SGLang or native Instance owners have already retired their clients.
        SIGTERM reaches only the daemon process; the daemon coordinates its
        Agents and controller while control remains available. Expiry or an
        unconfirmed daemon status retains this owner and its task proof. A
        caller interruption leaves the retained cleanup operation running.

        Args:
            deadline: Enclosing owner's absolute monotonic retirement envelope.
                Direct close reuses task retirement or starts the MPS budget.

        Raises:
            RuntimeError: A deployment failed after verified resource cleanup.
        """

        with self.cleanup_lock:
            if self.cleanup_future is None:
                root = get_task_root()
                if deadline is None and root is not None:
                    deadline = root.cleanup_deadline
                if self.cleanup_deadline is None:
                    self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S if deadline is None else deadline
                with defer_signal_exceptions():
                    self.cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xpool-cluster-close")
                    self.cleanup_future = self.cleanup_executor.submit(self.retire)
            future = self.cleanup_future
            executor = self.cleanup_executor
        try:
            future.result()
        finally:
            if future.done() and executor is not None:
                executor.shutdown(wait=True)
        failures = tuple(
            f"{process.name} exited with code {process.process.returncode}"
            for process in self.processes
            if process.process.returncode != 0
        )
        if failures:
            raise RuntimeError("xpool deployment failed after verified cleanup: " + "; ".join(failures))

    def retire(self) -> None:
        """Retain the actual handles until controlled daemon and domain retirement."""

        deadline = self.cleanup_deadline
        if deadline is None:
            raise RuntimeError("cluster retirement requires its retained deadline")
        daemons = tuple(process for process in self.processes if process.name == "daemon")
        notified = False
        expiry_reported = False
        last_diagnostic: tuple[type[Exception], str] | None = None
        while True:
            now = time.monotonic()
            if now >= deadline and not expiry_reported:
                logger.error("cluster cleanup expired; retaining owner and device grant; manual resolution required")
                expiry_reported = True
            try:
                if not notified and now < deadline:
                    for daemon in daemons:
                        if daemon.process.poll() is None:
                            daemon.process.send_signal(signal.SIGTERM)
                    notified = True
                if all(daemon.process.poll() in (0, 20) for daemon in daemons) and all(
                    wait_for_process_group(process.process, 0.0) for process in self.processes
                ):
                    self.client.close()
                    close_process_logs(self.processes)
                    self.closed = True
                    if self.task_scope is not None:
                        self.task_scope.complete()
                    return
                last_diagnostic = None
            except Exception as error:
                diagnostic = type(error), str(error)
                if diagnostic != last_diagnostic:
                    logger.error("cluster cleanup incomplete; retaining owner detail=%s", error)
                    last_diagnostic = diagnostic
            time.sleep(POLL_INTERVAL_SECONDS)

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


def daemon_url(launch: XpoolClusterLaunch) -> str:
    """Return the effective daemon URL for one materialized launch."""

    host = launch.config.daemon.host
    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{launch.config.daemon.port}"


def spawn_process(
    name: str,
    arguments: list[str],
    *,
    launch: XpoolClusterLaunch,
) -> OwnedProcessGroup:
    """Start one CrossPool CLI process with task-local configuration and logging."""

    return OwnedProcessGroup.spawn_logged(
        name,
        ["xpool", *arguments],
        cwd=launch.cwd,
        env=launch.environment,
        log_path=launch.config_path.parent / f"{name}.log",
    )


def wait_for_daemon_health(
    client: httpx.Client,
    processes: list[OwnedProcessGroup],
    *,
    url: str,
    evidence_path: Path,
    check_startup_active: Callable[[], None],
    startup_deadline: float | None = None,
) -> None:
    """Wait until the daemon health endpoint responds successfully."""

    started_at = time.monotonic()
    deadline = started_at + DAEMON_STARTUP_TIMEOUT_SECONDS if startup_deadline is None else startup_deadline
    evidence = ReadinessEvidence("daemon health", url)
    try:
        while time.monotonic() < deadline:
            check_startup_active()
            raise_for_exited_process(processes)
            try:
                response = (
                    client.get("/health")
                    if startup_deadline is None
                    else client.get(
                        "/health", timeout=min(CONTROL_PLANE_HTTP_TIMEOUT_SECONDS, deadline - time.monotonic())
                    )
                )
                evidence.record_response(response)
                if response.is_success:
                    return
            except httpx.HTTPError as error:
                evidence.record_error(error)
            time.sleep(POLL_INTERVAL_SECONDS)
        evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=processes)
        raise ReadinessTimeout(evidence)
    except BaseException as error:
        if not isinstance(error, ReadinessTimeout):
            evidence.record_error(error)
        raise
    finally:
        evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=processes)
        evidence.write(evidence_path)


def wait_for_agent_registrations(
    client: httpx.Client,
    processes: list[OwnedProcessGroup],
    *,
    launch: XpoolClusterLaunch,
    url: str,
    evidence_path: Path,
    check_startup_active: Callable[[], None],
    startup_deadline: float | None = None,
) -> None:
    """Wait for every configured AtnAgent and FfnAgent registration to be live."""

    expected_atn_devices = {agent.device for agent in launch.config.atnagents}
    expected_ffn_devices = {agent.device for agent in launch.config.ffnagents}
    started_at = time.monotonic()
    deadline = started_at + AGENT_REGISTRATION_TIMEOUT_SECONDS if startup_deadline is None else startup_deadline
    evidence = ReadinessEvidence("agent registration", url)
    try:
        while time.monotonic() < deadline:
            check_startup_active()
            raise_for_exited_process(processes)
            try:
                response = (
                    client.get("/ready")
                    if startup_deadline is None
                    else client.get(
                        "/ready", timeout=min(CONTROL_PLANE_HTTP_TIMEOUT_SECONDS, deadline - time.monotonic())
                    )
                )
                evidence.record_response(response)
                if response.is_success:
                    readiness = ReadinessSnapshot.model_validate(response.json())
                    online_atn_devices = {
                        entry.device for entry in readiness.atnagents if entry.status is ReadinessStatus.ONLINE
                    }
                    online_ffn_devices = {
                        entry.device for entry in readiness.ffnagents if entry.status is ReadinessStatus.ONLINE
                    }
                    if online_atn_devices == expected_atn_devices and online_ffn_devices == expected_ffn_devices:
                        return
            except (httpx.HTTPError, ValueError) as error:
                evidence.record_error(error)
            time.sleep(POLL_INTERVAL_SECONDS)
        evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=processes)
        raise ReadinessTimeout(evidence)
    except BaseException as error:
        if not isinstance(error, ReadinessTimeout):
            evidence.record_error(error)
        raise
    finally:
        evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=processes)
        evidence.write(evidence_path)


def raise_for_exited_process(processes: list[OwnedProcessGroup]) -> None:
    """Reject startup as soon as any acquired process exits."""

    for process in processes:
        returncode = process.process.poll()
        if returncode is not None:
            raise RuntimeError(f"{process.name} exited during startup with code {returncode}")


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
