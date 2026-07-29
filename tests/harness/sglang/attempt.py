"""Ownership of one complete public-command SGLang probe attempt."""

from __future__ import annotations

import errno
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from tests.harness.runner.network import TcpEndpointConflict, TcpEndpointReservation, TcpPortSpace
from tests.harness.sglang.cluster import CONTROL_PLANE_HTTP_TIMEOUT_SECONDS, XpoolCluster, daemon_url
from tests.harness.sglang.endpoints import SglangEndpointFamilyLease
from tests.harness.sglang.graph import GraphEvent, SglangGraphSettings, read_graph_events
from tests.harness.sglang.launch import E2eLaunch, materialize, model_id_slug
from tests.harness.sglang.manifest import E2eManifest, E2eServingCase
from tests.harness.sglang.parity import TokenOutput, TokenParityArtifact
from tests.harness.sglang.readiness import ReadinessEvidence, ReadinessTimeout
from tests.harness.sglang.server import SglangServerProcess, SglangServerResult
from xpool.config import LoopbackSite, XpoolConfig
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

    def token_parity_artifact(self, group: str) -> TokenParityArtifact:
        """Project this completed probe to portable token-parity evidence."""

        return TokenParityArtifact(
            group=group,
            graph_settings=self.graph_settings,
            outputs=tuple(TokenOutput(result.model_id, result.output_ids) for result in self.results),
        )


def wait_for_system_readiness(launch: E2eLaunch, servers: list[SglangServerProcess]) -> None:
    """Wait for every HTTP server and the selected loopback barrier."""

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

        if launch.loopback_site in {LoopbackSite.INSTANCE, LoopbackSite.ATNAGENT}:
            return
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

    manifest: E2eManifest
    case: E2eServingCase
    base_config: XpoolConfig
    graph_settings: SglangGraphSettings
    workdir: Path
    loopback_site: LoopbackSite
    daemon_endpoint: TcpEndpointReservation | None = None
    server_endpoints: list[SglangEndpointFamilyLease] = field(default_factory=list)
    cluster: XpoolCluster | None = None
    servers: list[SglangServerProcess] = field(default_factory=list)
    closed: bool = False

    def run(self) -> ProbeRun:
        """Execute one attempt through process cleanup and endpoint release.

        Startup endpoint occupation is the only retryable outcome. Process
        cleanup must succeed before endpoint classification, and every terminal
        path releases all endpoint leases before returning or raising.
        """

        self.workdir.mkdir(parents=True, exist_ok=False)
        started_at = time.monotonic()
        launch: E2eLaunch | None = None
        try:
            try:
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
                    self.manifest,
                    self.case,
                    base_config=self.base_config,
                    workdir=self.workdir,
                    daemon_port=self.daemon_endpoint.port,
                    loopback_site=self.loopback_site,
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
            except BaseException as startup_error:
                diagnostics = self.diagnostics()
                cleanup_failures = self.close_processes()
                if cleanup_failures:
                    details = [str(startup_error), "cleanup failures: " + "; ".join(cleanup_failures)]
                    if diagnostics:
                        details.append(diagnostics)
                    raise AssertionError("SGLang E2E startup cleanup failed:\n" + "\n".join(details)) from startup_error
                try:
                    occupied_addresses = self.classify_released_endpoints()
                except OSError as classification_error:
                    details = [str(startup_error), f"endpoint classification failed: {classification_error}"]
                    if diagnostics:
                        details.append(diagnostics)
                    raise AssertionError("SGLang E2E startup cleanup failed:\n" + "\n".join(details)) from startup_error
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

            assert launch is not None and self.cluster is not None
            try:
                with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
                    results = tuple(executor.map(SglangServerProcess.result, self.servers))
            except BaseException as inference_error:
                diagnostics = self.diagnostics()
                cleanup_failures = self.close_processes()
                details = [str(inference_error)]
                if cleanup_failures:
                    details.append("cleanup failures: " + "; ".join(cleanup_failures))
                else:
                    try:
                        occupied_addresses = self.classify_released_endpoints()
                    except OSError as classification_error:
                        details.append(f"endpoint classification failed: {classification_error}")
                    else:
                        if occupied_addresses:
                            details.append(f"occupied endpoints after cleanup: {occupied_addresses}")
                if diagnostics:
                    details.append(diagnostics)
                raise AssertionError("SGLang E2E inference failed:\n" + "\n".join(details)) from inference_error

            cleanup_failures = self.close_processes()
            if cleanup_failures:
                raise AssertionError("SGLang E2E cleanup failed: " + "; ".join(cleanup_failures))
            try:
                occupied_addresses = self.classify_released_endpoints()
            except OSError as classification_error:
                raise AssertionError(
                    f"SGLang E2E endpoint classification failed: {classification_error}"
                ) from classification_error
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
        """Close SGLang servers together, then the xpool cluster."""

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
