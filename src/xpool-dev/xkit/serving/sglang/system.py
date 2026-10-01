"""Complete local SGLang serving ownership shared by tests and benchmarks."""

from __future__ import annotations

import errno
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

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
from xpool.service.wire import ReadinessSnapshot

__all__ = ["XpoolServingSystem"]

POLL_INTERVAL_SECONDS = 0.1


class XpoolServingSystem:
    """Own daemon, Agents, servers and reservations through local shutdown.

    This owner checks local process groups and released endpoint bindings.
    Only the enclosing supervisor proves the complete descendant domain empty
    and authorizes GPU lease release.
    """

    def __init__(self) -> None:
        self.daemon_endpoint: TcpEndpointReservation | None = None
        self.server_endpoints: list[SglangEndpointFamilyLease] = []
        self.cluster: XpoolCluster | None = None
        self.servers: list[SglangServerProcess] = []
        self.endpoints: tuple[ServingEndpoint, ...] = ()
        self.closed = False

    @property
    def launch(self) -> XpoolClusterLaunch:
        """Actual bound runtime snapshot owned by the started cluster."""

        if self.cluster is None:
            raise RuntimeError("serving cluster has not started")
        return self.cluster.launch

    @classmethod
    def start(
        cls,
        launch: ServingLaunch,
        *,
        workdir: Path,
        startup_timeout_seconds: float,
    ) -> XpoolServingSystem:
        """Reserve, snapshot and start every Instance under one startup deadline.

        Complete System Ready and every HTTP health check precede endpoint
        publication. A failure rolls back all acquired resources. Confirmed
        endpoint occupation after safe local cleanup raises TcpEndpointConflict;
        cleanup errors take precedence and prohibit a retry in the same domain.
        """

        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("serving startup timeout must be positive and finite")
        deadline = time.monotonic() + startup_timeout_seconds
        system = cls()
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            port_space = TcpPortSpace.local()
            host = launch.config.daemon.host
            system.daemon_endpoint = TcpEndpointReservation.reserve(host, port_space=port_space, deadline=deadline)
            for model in launch.models:
                system.server_endpoints.append(
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
                daemon_port=system.daemon_endpoint.port,
                environment={
                    **launch.environment,
                    "SGLANG_PLUGINS": "xpool",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                cwd=launch.cwd,
            )
            system.cluster = XpoolCluster.start(cluster_launch, system.daemon_endpoint, startup_deadline=deadline)
            for model, endpoint in zip(launch.models, system.server_endpoints, strict=True):
                if time.monotonic() >= deadline:
                    raise TimeoutError("serving startup deadline expired before all servers started")
                system.servers.append(
                    SglangServerProcess.start(launch=cluster_launch, model=model, endpoint=endpoint, workdir=workdir)
                )
            system.wait_for_readiness(deadline)
            system.endpoints = tuple(ServingEndpoint(server.model.model_id, server.url()) for server in system.servers)
            return system
        except BaseException as error:
            diagnostics = system.diagnostics()
            try:
                system.close()
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
            raise

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

    def close(self) -> None:
        """Close servers, then cluster, inspect bindings and release reservations.

        Release every local reservation even when process cleanup or endpoint
        inspection fails. Successful return is not a descendant-domain proof.
        """

        if self.closed:
            return
        self.closed = True
        failures: list[str] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(self.servers))) as executor:
                futures = tuple(executor.submit(server.close) for server in self.servers)
                for future in futures:
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(f"{type(error).__name__}: {error}")
            if self.cluster is not None:
                try:
                    self.cluster.close()
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
            if failures:
                raise RuntimeError("serving process cleanup failed: " + "; ".join(failures))
            occupied = self.classify_released_endpoints()
            if occupied:
                raise TcpEndpointConflict(occupied)
        finally:
            if self.daemon_endpoint is not None:
                self.daemon_endpoint.close()
            for endpoint in self.server_endpoints:
                endpoint.close()

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
