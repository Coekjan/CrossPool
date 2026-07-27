"""Host-side orchestration for one complete public-command SGLang attempt."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from tests.harness.network import TcpEndpointReservation, TcpPortSpace
from tests.harness.sglang.cluster import DaemonPortConflict, XpoolCluster, daemon_url
from tests.harness.sglang.e2e import E2eLaunch, materialize
from tests.harness.sglang.endpoints import SglangEndpointFamilyLease
from tests.harness.sglang.graph import GraphEvent, SglangGraphSettings, read_graph_events
from tests.harness.sglang.manifest import E2eManifest, E2eServingCase
from tests.harness.sglang.parity import TokenOutput, TokenParityArtifact
from tests.harness.sglang.server import SglangEndpointConflict, SglangServerProcess, SglangServerResult
from xpool.config import LoopbackSite, XpoolConfig
from xpool.service.wire import ReadinessSnapshot

PROBE_TIMEOUT_SECONDS = 30 * 60
POLL_INTERVAL_SECONDS = 0.1
PROBE_ATTEMPTS = 3


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


def run_probe(
    manifest: E2eManifest,
    case: E2eServingCase,
    *,
    base_config: XpoolConfig,
    graph_settings: SglangGraphSettings,
    workdir: Path,
    loopback_site: LoopbackSite = LoopbackSite.FFNAGENT,
) -> ProbeRun:
    """Run one complete rematerialized attempt, retrying bind collisions only."""

    conflicts: list[str] = []
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return run_probe_attempt(
                manifest,
                case,
                base_config=base_config,
                graph_settings=graph_settings,
                workdir=workdir / f"attempt-{attempt}",
                loopback_site=loopback_site,
            )
        except (DaemonPortConflict, SglangEndpointConflict) as error:
            conflicts.append(str(error))
    raise RuntimeError("SGLang E2E endpoint conflicts exhausted:\n" + "\n".join(conflicts))


def run_probe_attempt(
    manifest: E2eManifest,
    case: E2eServingCase,
    *,
    base_config: XpoolConfig,
    graph_settings: SglangGraphSettings,
    workdir: Path,
    loopback_site: LoopbackSite,
) -> ProbeRun:
    """Own endpoints, xpool roles, servers, requests, and cleanup for one attempt."""

    workdir.mkdir(parents=True, exist_ok=False)
    host = base_config.daemon.host
    port_space = TcpPortSpace.local()
    daemon_endpoint = TcpEndpointReservation.reserve(host, port_space=port_space)
    server_endpoints: list[SglangEndpointFamilyLease] = []
    cluster: XpoolCluster | None = None
    servers: list[SglangServerProcess] = []
    started_at = time.monotonic()
    try:
        for placement in case.models:
            server_endpoints.append(
                SglangEndpointFamilyLease.acquire(
                    host,
                    dp_size=placement.atn_dp_size,
                    port_space=port_space,
                )
            )
        launch = materialize(
            manifest,
            case,
            base_config=base_config,
            workdir=workdir,
            daemon_port=daemon_endpoint.port,
            loopback_site=loopback_site,
        )
        cluster = XpoolCluster.start(launch, daemon_endpoint)
        for model, endpoint in zip(launch.models, server_endpoints, strict=True):
            servers.append(
                SglangServerProcess.start(
                    launch=launch,
                    model=model,
                    graph_settings=graph_settings,
                    endpoint=endpoint,
                    workdir=workdir,
                )
            )
        wait_for_system_readiness(launch, servers)
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            results = tuple(executor.map(SglangServerProcess.result, servers))
    except BaseException as error:
        diagnostics = attempt_diagnostics(cluster, servers)
        cleanup_failures = cleanup_attempt(cluster, servers, daemon_endpoint, server_endpoints)
        if not cleanup_failures and isinstance(error, (DaemonPortConflict, SglangEndpointConflict)):
            error.add_note(diagnostics)
            raise
        details = [str(error)]
        if cleanup_failures:
            details.append("cleanup failures: " + "; ".join(cleanup_failures))
        if diagnostics:
            details.append(diagnostics)
        raise AssertionError("SGLang E2E probe failed:\n" + "\n".join(details)) from error

    cleanup_failures = cleanup_attempt(cluster, servers, daemon_endpoint, server_endpoints)
    if cleanup_failures:
        raise AssertionError("SGLang E2E cleanup failed: " + "; ".join(cleanup_failures))
    return ProbeRun(
        launch=launch,
        graph_settings=graph_settings,
        results=results,
        events=read_graph_events(launch.observer_outdir),
        daemon_startup_seconds=cluster.daemon_startup_seconds,
        duration_seconds=time.monotonic() - started_at,
    )


def wait_for_system_readiness(launch: E2eLaunch, servers: list[SglangServerProcess]) -> None:
    """Wait for every HTTP server and the loopback site's reachable barrier."""

    deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if all(server.healthy() for server in servers):
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise RuntimeError("timed out waiting for every SGLang /health endpoint")

    if launch.loopback_site in {LoopbackSite.INSTANCE, LoopbackSite.ATNAGENT}:
        return
    with httpx.Client(base_url=daemon_url(launch), timeout=POLL_INTERVAL_SECONDS) as client:
        while time.monotonic() < deadline:
            if not all(server.healthy() for server in servers):
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            try:
                response = client.get("/ready")
                readiness = ReadinessSnapshot.model_validate(response.json()) if response.is_success else None
            except (httpx.HTTPError, ValueError):
                readiness = None
            if readiness is not None and readiness.ready:
                return
            time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError("timed out waiting for final xpool readiness")


def cleanup_attempt(
    cluster: XpoolCluster | None,
    servers: list[SglangServerProcess],
    daemon_endpoint: TcpEndpointReservation,
    server_endpoints: list[SglangEndpointFamilyLease],
) -> list[str]:
    """Close servers together, then xpool roles and unconsumed reservations."""

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(servers))) as executor:
        futures = tuple(executor.submit(server.close) for server in servers)
        for future in futures:
            try:
                future.result()
            except RuntimeError as error:
                failures.append(str(error))
    if cluster is not None:
        try:
            cluster.close()
        except RuntimeError as error:
            failures.append(str(error))
    daemon_endpoint.close()
    for endpoint in server_endpoints:
        endpoint.close()
    return failures


def attempt_diagnostics(cluster: XpoolCluster | None, servers: list[SglangServerProcess]) -> str:
    """Return bounded diagnostics for every acquired process."""

    sections = [server.diagnostics() for server in servers]
    if cluster is not None:
        sections.append(cluster.diagnostics())
    return "\n".join(section for section in sections if section)
