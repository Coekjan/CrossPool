"""Complete local SGLang serving ownership shared by tests and benchmarks."""

from __future__ import annotations

import asyncio
import errno
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn, cast

import httpx

from xkit.network import TcpEndpointConflict, TcpEndpointReservation, TcpPortSpace
from xkit.serving.cluster import (
    CONTROL_PLANE_HTTP_TIMEOUT_SECONDS,
    XpoolCluster,
    XpoolClusterLaunch,
    daemon_url,
    raise_for_exited_process,
)
from xkit.serving.launch import ServingEndpoint, snapshot_cluster_launch
from xkit.serving.readiness import ReadinessEvidence, ReadinessTimeout
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease
from xkit.serving.sglang.launch import ServingLaunch
from xkit.serving.sglang.server import HTTP_TIMEOUT_SECONDS, SglangServerProcess
from xkit.task import TaskCleanupScope, get_task_root
from xpool.service.wire import ReadinessSnapshot
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S
from xpool.utils.sighandler import defer_signal_exceptions

__all__ = ["XpoolServingSystem"]

POLL_INTERVAL_SECONDS = 0.1


class XpoolServingSystem:
    """Own daemon, Agents, servers and reservations through local shutdown.

    This owner checks local process groups and released endpoint bindings.
    Startup publishes partial resources on this retained object. A background
    startup and ordered cleanup each have one retained operation; cancellation
    of an asyncio waiter does not abandon either operation. Only the enclosing
    supervisor proves the complete descendant domain empty.
    """

    def __init__(self) -> None:
        self.daemon_endpoint: TcpEndpointReservation | None = None
        self.server_endpoints: list[SglangEndpointFamilyLease] = []
        self.cluster: XpoolCluster | None = None
        self.servers: list[SglangServerProcess] = []
        self.endpoints: tuple[ServingEndpoint, ...] = ()
        self.startup_deadline: float | None = None
        self.startup_executor: ThreadPoolExecutor | None = None
        self.startup_future: Future[None] | None = None
        self.task_scope: TaskCleanupScope | None = None
        self.cleanup_lock = threading.Lock()
        self.cleanup_deadline: float | None = None
        self.cleanup_executor: ThreadPoolExecutor | None = None
        self.cleanup_future: Future[None] | None = None
        self.closed = False

    @property
    def launch(self) -> XpoolClusterLaunch:
        """Actual bound runtime snapshot owned by the started cluster."""

        if self.cluster is None:
            raise RuntimeError("serving cluster has not started")
        return self.cluster.launch

    def start(
        self,
        launch: ServingLaunch,
        *,
        workdir: Path,
        startup_timeout_seconds: float,
    ) -> None:
        """Reserve, snapshot and start every Instance under one startup deadline.

        Complete System Ready and every HTTP health check precede endpoint
        publication. A failure rolls back all acquired resources. Confirmed
        endpoint occupation after safe local cleanup raises TcpEndpointConflict;
        cleanup errors take precedence and prohibit a retry in the same domain.
        """

        self.begin_startup(startup_timeout_seconds)
        try:
            self.launch_resources(launch, workdir=workdir)
        except BaseException as error:
            self.rollback(error)

    async def start_async(
        self,
        launch: ServingLaunch,
        *,
        workdir: Path,
        startup_timeout_seconds: float,
    ) -> None:
        """Run synchronous serving startup while the caller's loop stays active.

        The caller already retains this System. Cancellation seals further
        resource creation, joins the actual worker and performs ordered rollback
        off the event loop before propagating the original cancellation.
        """

        self.begin_startup(startup_timeout_seconds)
        with defer_signal_exceptions():
            self.startup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xpool-serving-start")
            self.startup_future = self.startup_executor.submit(self.launch_resources, launch, workdir=workdir)
        waiter = asyncio.wrap_future(self.startup_future)
        try:
            await asyncio.shield(waiter)
        except BaseException as error:
            self.begin_close()
            await asyncio.to_thread(self.rollback, error)
        finally:
            # The cancelled shield waiter did not consume the worker's result.
            # Join its asyncio projection too, keeping the original verdict.
            await asyncio.gather(waiter, return_exceptions=True)
            if self.startup_future.done():
                self.startup_executor.shutdown(wait=True)

    def begin_startup(self, startup_timeout_seconds: float) -> None:
        """Retain task protection and a single startup envelope before creation."""

        if self.closed or self.startup_deadline is not None or self.cleanup_deadline is not None:
            raise RuntimeError("serving system startup was already attempted or retired")
        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("serving startup timeout must be positive and finite")
        self.startup_deadline = time.monotonic() + startup_timeout_seconds
        root = get_task_root()
        if root is not None:
            with defer_signal_exceptions():
                self.task_scope = root.register_scope()
            root.activate()

    def launch_resources(self, launch: ServingLaunch, *, workdir: Path) -> None:
        """Publish actual resources under the already retained startup operation."""

        deadline = self.startup_deadline
        if deadline is None:
            raise RuntimeError("serving startup requires its retained deadline")
        workdir.mkdir(parents=True, exist_ok=True)
        self.check_startup_active()
        port_space = TcpPortSpace.local()
        host = launch.config.daemon.host
        with defer_signal_exceptions():
            self.daemon_endpoint = TcpEndpointReservation.reserve(host, port_space=port_space, deadline=deadline)
        for model in launch.models:
            self.check_startup_active()
            with defer_signal_exceptions():
                self.server_endpoints.append(
                    SglangEndpointFamilyLease.acquire(
                        host,
                        dp_size=launch.config.model_by_id[model.model_id].atn_dp_size,
                        port_space=port_space,
                        deadline=deadline,
                    )
                )
        cluster_launch = snapshot_cluster_launch(
            launch.config,
            workdir=workdir / "launch",
            daemon_port=self.daemon_endpoint.port,
            environment={
                **launch.environment,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            cwd=launch.cwd,
        )
        self.check_startup_active()
        with defer_signal_exceptions():
            self.cluster = XpoolCluster(cluster_launch)
        self.check_startup_active()
        self.cluster.start(self.daemon_endpoint, startup_deadline=deadline, enclosing_scope=self.task_scope)
        for model, endpoint in zip(launch.models, self.server_endpoints, strict=True):
            self.check_startup_active()
            if time.monotonic() >= deadline:
                raise TimeoutError("serving startup deadline expired before all servers started")
            with defer_signal_exceptions():
                self.servers.append(
                    SglangServerProcess.start(launch=cluster_launch, model=model, endpoint=endpoint, workdir=workdir)
                )
        self.wait_for_readiness(deadline)
        self.endpoints = tuple(ServingEndpoint(server.model.model_id, server.url()) for server in self.servers)

    def check_startup_active(self) -> None:
        """Stop further acquisition when this owner or its task is retiring."""

        root = get_task_root()
        if self.cleanup_deadline is not None or (root is not None and root.sealed):
            raise InterruptedError("serving startup is retiring")

    def rollback(self, error: BaseException) -> NoReturn:
        """Join ordered rollback, preserving the operation error after safe cleanup."""

        diagnostics = self.diagnostics()
        try:
            self.close()
        except TcpEndpointConflict as conflict:
            if diagnostics:
                conflict.add_note(diagnostics)
            raise conflict from error
        except BaseException as cleanup_error:
            if diagnostics:
                cleanup_error.add_note(diagnostics)
            raise RuntimeError(f"serving startup rollback failed: {cleanup_error}") from error
        if diagnostics:
            error.add_note(diagnostics)
        raise error

    def check_alive(self) -> None:
        """Raise on any owned process exit; a closed system is not live."""

        if self.closed or self.cluster is None:
            raise RuntimeError("serving system is not live")
        raise_for_exited_process(self.cluster.processes)
        raise_for_exited_process([server.owner for server in self.servers])

    def wait_for_readiness(self, deadline: float) -> None:
        started_at = time.monotonic()
        cluster_launch = self.launch
        server_evidence = {
            server.model.model_id: ReadinessEvidence(f"SGLang {server.model.model_id} health", f"{server.url()}/health")
            for server in self.servers
        }
        system_evidence = ReadinessEvidence("system readiness", f"{daemon_url(cluster_launch)}/ready")
        evidence = system_evidence
        owners = [server.owner for server in self.servers]
        try:
            with httpx.Client(base_url=daemon_url(cluster_launch)) as client:
                while time.monotonic() < deadline:
                    self.check_startup_active()
                    self.check_alive()
                    healthy = True
                    for server in self.servers:
                        evidence = server_evidence[server.model.model_id]
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=owners)
                            raise ReadinessTimeout(evidence)
                        if not server.healthy(evidence, timeout_seconds=min(HTTP_TIMEOUT_SECONDS, remaining)):
                            healthy = False
                    evidence = system_evidence
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        response = client.get("/ready", timeout=min(CONTROL_PLANE_HTTP_TIMEOUT_SECONDS, remaining))
                        system_evidence.record_response(response)
                        readiness = ReadinessSnapshot.model_validate(response.json()) if response.is_success else None
                    except (httpx.HTTPError, ValueError) as error:
                        system_evidence.record_error(error)
                        readiness = None
                    if healthy and readiness is not None and readiness.ready:
                        return
                    time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
            evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=owners)
            raise ReadinessTimeout(evidence)
        except BaseException as error:
            if not isinstance(error, ReadinessTimeout):
                evidence.record_error(error)
            raise
        finally:
            directory = cluster_launch.config_path.parent / "readiness"
            elapsed = time.monotonic() - started_at
            for server in self.servers:
                evidence = server_evidence[server.model.model_id]
                evidence.finish(elapsed_seconds=elapsed, processes=(server.owner,))
                evidence.write(cast(Path, server.owner.log_path).parent / "health.json")
            system_evidence.finish(elapsed_seconds=elapsed, processes=owners)
            system_evidence.write(directory / "system-readiness.json")

    def diagnostics(self) -> str:
        """Return bounded process state and log tails for startup/inference errors."""

        sections = [server.diagnostics() for server in self.servers]
        if self.cluster is not None:
            sections.append(self.cluster.diagnostics())
        return "\n".join(section for section in sections if section)

    def close(self, *, deadline: float | None = None) -> None:
        """Close servers, then cluster, inspect bindings and release reservations.

        Concurrent callers join one retained operation. The first trigger seals
        startup and records an absolute deadline. Live or unconfirmed clients
        retain the daemon and all reservations. Successful return proves local
        retirement, not the enclosing supervisor's full descendant domain.
        """

        future = self.begin_close(deadline=deadline)
        try:
            future.result()
        finally:
            if future.done() and self.cleanup_executor is not None:
                self.cleanup_executor.shutdown(wait=True)

    def begin_close(self, *, deadline: float | None = None) -> Future[None]:
        """Seal startup and retain the shared close operation before waiting.

        Both synchronous and asyncio callers obtain this same concrete Future.
        Waiting cancellation cannot prevent dispatch or renew its deadline.
        """

        with self.cleanup_lock:
            if self.cleanup_future is None:
                if deadline is None and self.cluster is not None:
                    deadline = self.cluster.cleanup_deadline
                root = get_task_root()
                if deadline is None and root is not None:
                    deadline = root.cleanup_deadline
                if self.cleanup_deadline is None:
                    self.cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S if deadline is None else deadline
                if self.cluster is not None and self.cluster.cleanup_deadline is None:
                    # Seal in-progress Cluster startup without retiring its
                    # daemon before System-owned serving clients have exited.
                    self.cluster.cleanup_deadline = self.cleanup_deadline
                with defer_signal_exceptions():
                    self.cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xpool-serving-close")
                    self.cleanup_future = self.cleanup_executor.submit(self.retire)
            return self.cleanup_future

    async def close_async(self) -> None:
        """Join the same retained ordered close operation off the event loop."""

        future = self.begin_close()
        try:
            # gather consumes the projection's result even if its shield waiter
            # is cancelled. The actual cleanup Future remains on this owner.
            await asyncio.shield(asyncio.gather(asyncio.wrap_future(future), return_exceptions=True))
            future.result()
        finally:
            if future.done() and self.cleanup_executor is not None:
                self.cleanup_executor.shutdown(wait=True)

    def retire(self) -> None:
        """Join startup, retire every server, then release the daemon and bindings."""

        if self.startup_future is not None:
            try:
                self.startup_future.result()
            except BaseException:
                # Startup's waiter retains its verdict. Failed startup still
                # leaves these actual partial handles for ordered retirement.
                pass
        if self.startup_executor is not None:
            self.startup_executor.shutdown(wait=True)
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.servers))) as executor:
            futures = tuple(executor.submit(server.close, deadline=self.cleanup_deadline) for server in self.servers)
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
        if failures:
            raise RuntimeError("serving process cleanup failed: " + "; ".join(failures))
        deployment_error: Exception | None = None
        if self.cluster is not None:
            try:
                self.cluster.close(deadline=self.cleanup_deadline)
            except Exception as error:
                if not self.cluster.closed:
                    raise
                deployment_error = error
        try:
            occupied = self.classify_released_endpoints()
        finally:
            if self.daemon_endpoint is not None:
                self.daemon_endpoint.close()
            for endpoint in self.server_endpoints:
                endpoint.close()
            self.closed = True
            if self.task_scope is not None:
                self.task_scope.complete()
        if deployment_error is not None and self.endpoints:
            raise deployment_error
        if occupied:
            raise TcpEndpointConflict(occupied)

    def classify_released_endpoints(self) -> tuple[tuple[str, int], ...]:
        occupied: list[tuple[str, int]] = []
        if self.daemon_endpoint is not None and self.daemon_endpoint.listener is None:
            try:
                self.daemon_endpoint.reacquire()
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                occupied.append(self.daemon_endpoint.address)
        for endpoint in self.server_endpoints:
            if endpoint.tcp_released:
                occupied.extend((endpoint.family.host, port) for port in endpoint.reacquire_tcp())
        return tuple(occupied)
