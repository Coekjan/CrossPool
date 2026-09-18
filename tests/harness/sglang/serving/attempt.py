"""Ownership of one complete public-command SGLang probe attempt."""

from __future__ import annotations

import errno
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from threading import Barrier

import httpx

from tests.harness.native.cluster import CONTROL_PLANE_HTTP_TIMEOUT_SECONDS, XpoolCluster, daemon_url
from tests.harness.native.readiness import ReadinessEvidence, ReadinessTimeout
from tests.harness.runner.network import TcpEndpointConflict, TcpEndpointReservation, TcpPortSpace
from tests.harness.sglang.manifest import E2eModel, E2eServingCase
from tests.harness.sglang.serving.alignment import ServingGraphArtifact, TokenOutput
from tests.harness.sglang.serving.endpoints import SglangEndpointFamilyLease
from tests.harness.sglang.serving.graph import GraphEvent, SglangGraphSettings, read_graph_events
from tests.harness.sglang.serving.launch import E2eLaunch, materialize, model_id_slug
from tests.harness.sglang.serving.server import SglangServerProcess, SglangServerResult
from xpool.config import LatencySloConfig, XpoolConfig
from xpool.service.wire import ReadinessSnapshot

PROBE_TIMEOUT_SECONDS = 30 * 60
POLL_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class ProbeRun:
    """Results and graph evidence from one complete SGLang attempt."""

    launch: E2eLaunch
    graph_settings: SglangGraphSettings
    results: tuple[SglangServerResult, ...]
    events: list[GraphEvent]
    daemon_startup_seconds: float
    duration_seconds: float

    def serving_graph_artifact(self, group: str) -> ServingGraphArtifact:
        """Project this completed probe to portable serving-graph evidence."""

        return ServingGraphArtifact(
            group=group,
            graph_settings=self.graph_settings,
            outputs=tuple(TokenOutput(result.model_id, result.output_ids) for result in self.results),
        )


def wait_for_system_readiness(launch: E2eLaunch, servers: list[SglangServerProcess]) -> None:
    """Wait for every HTTP server and complete system readiness."""

    started_at = time.monotonic()
    deadline = started_at + PROBE_TIMEOUT_SECONDS
    readiness_directory = launch.config_path.parent / "readiness"
    server_evidence = {
        server.model.model_id: ReadinessEvidence(
            f"SGLang {server.model.model_id} health",
            f"{server.url()}/health",
        )
        for server in servers
    }
    system_evidence = ReadinessEvidence("system readiness", f"{daemon_url(launch)}/ready")
    owners = [server.owner for server in servers]
    try:
        while time.monotonic() < deadline:
            if all(server.healthy(server_evidence[server.model.model_id]) for server in servers):
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            evidence = next(
                item
                for item in server_evidence.values()
                if item.last_status_code is None or item.last_status_code >= 400
            )
            evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=owners)
            raise ReadinessTimeout(evidence)

        with httpx.Client(base_url=daemon_url(launch), timeout=CONTROL_PLANE_HTTP_TIMEOUT_SECONDS) as client:
            while time.monotonic() < deadline:
                try:
                    response = client.get("/ready")
                    system_evidence.record_response(response)
                    readiness = ReadinessSnapshot.model_validate(response.json()) if response.is_success else None
                except (httpx.HTTPError, ValueError) as error:
                    system_evidence.record_error(error)
                    readiness = None
                if readiness is not None and readiness.ready:
                    return
                time.sleep(POLL_INTERVAL_SECONDS)
        system_evidence.finish(elapsed_seconds=time.monotonic() - started_at, processes=owners)
        raise ReadinessTimeout(system_evidence)
    except BaseException as error:
        if not isinstance(error, ReadinessTimeout):
            system_evidence.record_error(error)
        raise
    finally:
        elapsed = time.monotonic() - started_at
        for server in servers:
            evidence = server_evidence[server.model.model_id]
            evidence.finish(elapsed_seconds=elapsed, processes=(server.owner,))
            evidence.write(readiness_directory / f"sglang-{model_id_slug(server.model.model_id)}-health.json")
        system_evidence.finish(elapsed_seconds=elapsed, processes=owners)
        system_evidence.write(readiness_directory / "system-readiness.json")


@dataclass(slots=True)
class ProbeAttempt:
    """Own every resource acquired by one non-reusable E2E attempt."""

    case: E2eServingCase
    models: tuple[E2eModel, ...]
    serving_slo: LatencySloConfig
    base_config: XpoolConfig
    graph_settings: SglangGraphSettings
    workdir: Path
    daemon_endpoint: TcpEndpointReservation | None = None
    server_endpoints: list[SglangEndpointFamilyLease] = field(default_factory=list)
    cluster: XpoolCluster | None = None
    servers: list[SglangServerProcess] = field(default_factory=list)
    closed: bool = False

    def start_system(self) -> E2eLaunch:
        """Reserve endpoints, start every process, and reach readiness."""

        host = self.base_config.daemon.host
        port_space = TcpPortSpace.local()
        self.daemon_endpoint = TcpEndpointReservation.reserve(host, port_space=port_space)
        for placement in self.case.models:
            self.server_endpoints.append(
                SglangEndpointFamilyLease.acquire(
                    host,
                    dp_size=placement.atn_dp_size,
                    port_space=port_space,
                )
            )
        launch = materialize(
            self.case,
            models=self.models,
            serving_slo=self.serving_slo,
            base_config=self.base_config,
            workdir=self.workdir,
            daemon_port=self.daemon_endpoint.port,
            graph_settings=self.graph_settings,
        )
        self.cluster = XpoolCluster.start(launch, self.daemon_endpoint)
        for model, endpoint in zip(launch.models, self.server_endpoints, strict=True):
            self.servers.append(
                SglangServerProcess.start(
                    launch=launch,
                    model=model,
                    graph_settings=self.graph_settings,
                    endpoint=endpoint,
                    workdir=self.workdir,
                )
            )
        wait_for_system_readiness(launch, self.servers)
        return launch

    def run(
        self,
        workload: Callable[[list[SglangServerProcess]], tuple[SglangServerResult, ...]] | None = None,
    ) -> ProbeRun:
        """Execute one attempt through process cleanup and endpoint release.

        Startup endpoint occupation is the only retryable outcome. Process
        cleanup must succeed before endpoint classification, and every terminal
        path releases all endpoint leases before returning or raising.
        """

        self.workdir.mkdir(parents=True, exist_ok=False)
        started_at = time.monotonic()
        try:
            try:
                launch = self.start_system()
            except BaseException as startup_error:
                diagnostics = self.diagnostics()
                try:
                    occupied_addresses = self.close_processes_and_inspect_endpoints()
                except Exception as cleanup_error:
                    startup_error.add_note(
                        f"startup cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    if diagnostics:
                        startup_error.add_note(diagnostics)
                    raise AssertionError(f"SGLang E2E startup cleanup failed: {cleanup_error}") from startup_error
                if isinstance(startup_error, TcpEndpointConflict):
                    if diagnostics:
                        startup_error.add_note(diagnostics)
                    raise
                if occupied_addresses:
                    conflict = TcpEndpointConflict(occupied_addresses)
                    if diagnostics:
                        conflict.add_note(diagnostics)
                    raise conflict from startup_error
                if diagnostics:
                    startup_error.add_note(diagnostics)
                raise

            assert self.cluster is not None
            try:
                if workload is None:
                    request_barrier = Barrier(len(self.servers))
                    with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
                        results = tuple(executor.map(SglangServerProcess.result, self.servers, repeat(request_barrier)))
                else:
                    results = workload(self.servers)
            except BaseException as inference_error:
                diagnostics = self.diagnostics()
                try:
                    occupied_addresses = self.close_processes_and_inspect_endpoints()
                except Exception as cleanup_error:
                    inference_error.add_note(
                        f"inference cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                else:
                    if occupied_addresses:
                        inference_error.add_note(f"occupied endpoints after cleanup: {occupied_addresses}")
                if diagnostics:
                    inference_error.add_note(diagnostics)
                raise AssertionError(f"SGLang E2E inference failed: {inference_error}") from inference_error

            try:
                occupied_addresses = self.close_processes_and_inspect_endpoints()
            except Exception as cleanup_error:
                raise AssertionError(f"SGLang E2E cleanup failed: {cleanup_error}") from cleanup_error
            if occupied_addresses:
                raise AssertionError(f"SGLang E2E cleanup left occupied endpoints: {occupied_addresses}")
            return ProbeRun(
                launch=launch,
                graph_settings=self.graph_settings,
                results=results,
                events=read_graph_events(launch.observer_outdir),
                daemon_startup_seconds=self.cluster.daemon_startup_seconds,
                duration_seconds=time.monotonic() - started_at,
            )
        finally:
            try:
                self.close_endpoints()
            finally:
                self.closed = True

    def diagnostics(self) -> str:
        """Return bounded diagnostics for every acquired process."""

        sections = [server.diagnostics() for server in self.servers]
        if self.cluster is not None:
            sections.append(self.cluster.diagnostics())
        return "\n".join(section for section in sections if section)

    def close_processes(self) -> tuple[str, ...]:
        """Close SGLang servers together, then the CrossPool cluster."""

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.servers))) as executor:
            futures = tuple(executor.submit(server.close) for server in self.servers)
            for future in futures:
                try:
                    future.result()
                except RuntimeError as error:
                    failures.append(str(error))
        if self.cluster is not None:
            try:
                self.cluster.close()
            except RuntimeError as error:
                failures.append(str(error))
        return tuple(failures)

    def close_processes_and_inspect_endpoints(self) -> tuple[tuple[str, int], ...]:
        """Close every process before inspecting released endpoint ownership."""

        failures = self.close_processes()
        if failures:
            raise RuntimeError("SGLang E2E process cleanup failed: " + "; ".join(failures))
        return self.classify_released_endpoints()

    def classify_released_endpoints(self) -> tuple[tuple[str, int], ...]:
        """Reacquire released listeners and return only confirmed occupations."""

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

    def close_endpoints(self) -> None:
        """Close every endpoint reservation idempotently."""

        if self.daemon_endpoint is not None:
            self.daemon_endpoint.close()
        for endpoint in self.server_endpoints:
            endpoint.close()

    def close(self) -> None:
        """Release endpoints after process cleanup exactly once."""

        if self.closed:
            return
        self.closed = True
        failures = self.close_processes()
        self.close_endpoints()
        if failures:
            raise RuntimeError("SGLang E2E attempt cleanup failed: " + "; ".join(failures))
