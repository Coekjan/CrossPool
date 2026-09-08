"""FastAPI routes for the CrossPool daemon control plane."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from importlib.metadata import version
from time import monotonic

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from xpool import bootstrap
from xpool.config import XpoolConfig, get_global_config
from xpool.fabric import FabricPlan
from xpool.native import RuntimeRole
from xpool.service.daemon.control import ControlPlane
from xpool.service.daemon.registration import (
    AtnAgentRegistrationState,
    FfnAgentRegistrationState,
    InstanceRankId,
    InstanceRankRegistrationState,
)
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaUpsertRequest,
    AtnAgentTransportLeaseQuiesceResponse,
    FabricParticipantReport,
    FabricQuiesceRequest,
    FfnAgentRegistration,
    HeartbeatResponse,
    InstanceRankInitializedPublication,
    InstanceRankRegistration,
    ProcessRef,
    ReadinessSnapshot,
    ServingListener,
    XpoolDaemonErrorDetail,
)
from xpool.transport import TransportArenaHandle

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DaemonFailure:
    """Retain the first unrecoverable daemon-local background failure."""

    exception: BaseException | None = None

    @property
    def failed(self) -> bool:
        """Return whether an unrecoverable daemon failure was recorded."""

        return self.exception is not None

    def record(self, exception: BaseException) -> None:
        """Retain ``exception`` unless an earlier failure already won."""

        if self.exception is None:
            self.exception = exception


async def probe_serving_listener(client: httpx.AsyncClient, listener: ServingListener) -> bool:
    """Return whether one Instance listener answers its HTTP health check."""

    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(listener.host, listener.host)
    url = httpx.URL(scheme="http", host=host, port=listener.port, path="/health")
    try:
        response = await client.get(url)
    except httpx.RequestError:
        return False
    return response.is_success


def create_daemon() -> FastAPI:
    """Create the FastAPI daemon application for the process-global config.

    Returns:
        Configured daemon application.

    Side Effects:
        Initializes the process-wide native daemon role. The application
        lifespan owns the daemon watchdog and one-time serving-health monitor
        and records their unrecoverable failures.
    """

    bootstrap.init(None, RuntimeRole.DAEMON)
    control_plane = ControlPlane()
    daemon_failure = DaemonFailure()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        async def run_watchdog() -> None:
            try:
                while True:
                    await asyncio.to_thread(control_plane.watchdog)
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except BaseException as exception:
                logger.exception("daemon watchdog failed")
                daemon_failure.record(exception)

        async def monitor_serving_health() -> None:
            observed_targets = None
            healthy_instance_ids: set[str] = set()
            try:
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                    while True:
                        targets = await asyncio.to_thread(control_plane.capture_serving_health_targets)
                        if targets is None:
                            observed_targets = None
                            healthy_instance_ids.clear()
                        else:
                            if targets != observed_targets:
                                observed_targets = targets
                                healthy_instance_ids.clear()
                            pending = tuple(
                                (instance_id, listener)
                                for instance_id, listener in targets.listeners
                                if instance_id not in healthy_instance_ids
                            )
                            results = await asyncio.gather(
                                *(probe_serving_listener(client, listener) for _, listener in pending)
                            )
                            healthy_instance_ids.update(
                                instance_id
                                for (instance_id, _), healthy in zip(pending, results, strict=True)
                                if healthy
                            )
                            if len(healthy_instance_ids) == len(targets.listeners) and await asyncio.to_thread(
                                control_plane.confirm_serving_health,
                                targets,
                            ):
                                logger.info(
                                    "serving healthy generation=%s instance_count=%s",
                                    targets.generation.format(),
                                    len(targets.listeners),
                                )
                                for instance_id, listener in targets.listeners:
                                    logger.info(
                                        "serving listener instance=%s host=%s port=%s",
                                        instance_id,
                                        listener.host,
                                        listener.port,
                                    )
                        await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except BaseException as exception:
                logger.exception("serving health monitor failed")
                daemon_failure.record(exception)

        config = get_global_config()
        logger.info("process started host=%s port=%s pid=%s", config.daemon.host, config.daemon.port, os.getpid())
        tasks = (
            asyncio.create_task(run_watchdog(), name="xpool-daemon-watchdog"),
            asyncio.create_task(monitor_serving_health(), name="xpool-serving-health"),
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="xpool daemon", version=version("xpool"), lifespan=lifespan)
    app.state.control_plane = control_plane
    app.state.daemon_failure = daemon_failure

    @app.exception_handler(XpoolDaemonError)
    async def daemon_error_handler(request: Request, exc: XpoolDaemonError) -> JSONResponse:
        match exc.kind:
            case "conflict":
                status = HTTPStatus.CONFLICT
            case "not_found":
                status = HTTPStatus.NOT_FOUND
            case "not_ready":
                status = HTTPStatus.SERVICE_UNAVAILABLE
        return JSONResponse(
            status_code=int(status),
            content={"detail": XpoolDaemonErrorDetail.from_error(exc).model_dump(mode="json")},
        )

    @app.get("/health")
    async def health() -> Response:
        """Report that the daemon HTTP process is responsive."""

        return Response(status_code=HTTPStatus.OK)

    @app.get("/ready")
    async def ready() -> ReadinessSnapshot:
        """Return the projected readiness of MPS, registrations, Transport, and Fabric."""

        return await asyncio.to_thread(control_plane.readiness_snapshot)

    @app.get("/config")
    async def get_config() -> XpoolConfig:
        """Return the daemon's validated process-global configuration."""

        return get_global_config()

    @app.get("/fabric/plan")
    async def get_fabric_plan() -> FabricPlan:
        """Return the retained Fabric plan once a complete generation exists.

        Raises:
            503: No complete Fabric generation has been planned.
        """

        return await asyncio.to_thread(control_plane.require_fabric_plan)

    @app.post("/fabric/quiesce")
    async def request_fabric_quiesce(request: FabricQuiesceRequest) -> Response:
        """Authenticate an Agent participant and stop new Fabric admission.

        Returns:
            Empty 204 response after quiesce is accepted.

        Raises:
            404: The retained generation or participant does not exist.
            409: Generation identity or participant ownership conflicts.
            503: The generation cannot enter quiesce from its current phase.
        """

        await asyncio.to_thread(control_plane.request_fabric_quiesce, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/fabric/participant-reports")
    async def report_fabric_participant(request: FabricParticipantReport) -> Response:
        """Commit one owner-authenticated Fabric participant phase report.

        Returns:
            Empty 204 response after the report is committed.

        Raises:
            404: The reported generation or participant does not exist.
            409: Owner identity, report sequence, or phase transition conflicts.
            503: The generation is unavailable for participant reports.
        """

        await asyncio.to_thread(control_plane.record_fabric_participant, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/config/check")
    async def check_config(request: XpoolConfig) -> Response:
        """Require a participant's effective configuration to match the daemon.

        Returns:
            Empty 204 response when the configurations match.

        Raises:
            409: The participant configuration differs from the daemon.
        """

        await asyncio.to_thread(control_plane.check_config, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.get("/atnagents")
    async def list_atnagents() -> list[AtnAgentRegistration]:
        """List retained AtnAgent registrations."""

        return await asyncio.to_thread(control_plane.list_atnagents)

    @app.get("/instances")
    async def list_instances() -> list[InstanceRankRegistration]:
        """List retained Instance-rank registrations."""

        return await asyncio.to_thread(control_plane.list_instances)

    @app.get("/ffnagents")
    async def list_ffnagents() -> list[FfnAgentRegistration]:
        """List retained FfnAgent registrations."""

        return await asyncio.to_thread(control_plane.list_ffnagents)

    @app.post("/atnagent/register")
    async def register_atnagent(request: AtnAgentRegistration) -> Response:
        """Register one live AtnAgent as the owner of a configured CUDA device.

        Returns:
            Empty 204 response after registration.

        Raises:
            409: ABI, placement, or existing process ownership conflicts.
            503: A predecessor owner has not completed replacement cleanup.
        """

        await asyncio.to_thread(
            control_plane.register_atnagent,
            AtnAgentRegistrationState(
                cuda_device=request.cuda_device,
                abi_version=request.abi_version,
                pid=request.pid,
                now=monotonic(),
            ),
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/atnagent/{cuda_device}/heartbeat")
    async def heartbeat_atnagent(cuda_device: int, request: ProcessRef) -> HeartbeatResponse:
        """Refresh one authenticated AtnAgent registration and return desired state.

        Raises:
            404: No AtnAgent owns the requested CUDA device.
            409: The process identity does not own that registration.
        """

        return await asyncio.to_thread(control_plane.heartbeat_atnagent, cuda_device, request)

    @app.post("/ffnagent/register")
    async def register_ffnagent(request: FfnAgentRegistration) -> Response:
        """Register one live FfnAgent as the owner of a configured CUDA device.

        Returns:
            Empty 204 response after registration.

        Raises:
            409: ABI, placement, or existing process ownership conflicts.
            503: A predecessor owner has not completed replacement cleanup.
        """

        await asyncio.to_thread(
            control_plane.register_ffnagent,
            FfnAgentRegistrationState(
                cuda_device=request.cuda_device,
                cuda_total_memory_bytes=request.cuda_total_memory_bytes,
                cuda_free_memory_bytes=request.cuda_free_memory_bytes,
                abi_version=request.abi_version,
                pid=request.pid,
                now=monotonic(),
            ),
            request.model_specs,
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/ffnagent/{cuda_device}/heartbeat")
    async def heartbeat_ffnagent(cuda_device: int, request: ProcessRef) -> HeartbeatResponse:
        """Refresh one authenticated FfnAgent registration and return desired state.

        Raises:
            404: No FfnAgent owns the requested CUDA device.
            409: The process identity does not own that registration.
        """

        return await asyncio.to_thread(control_plane.heartbeat_ffnagent, cuda_device, request)

    @app.post("/atnagent/{cuda_device}/transport-arenas")
    async def upsert_atnagent_transport_arenas(
        cuda_device: int,
        request: AtnAgentTransportArenaUpsertRequest,
    ) -> Response:
        """Merge immutable Transport arena publications for one AtnAgent.

        Returns:
            Empty 204 response after publications are committed.

        Raises:
            404: The publishing AtnAgent is not registered.
            409: Publisher ownership or an immutable arena binding conflicts.
            503: The AtnAgent is unavailable for Transport publication.
        """

        await asyncio.to_thread(
            control_plane.upsert_atnagent_transport_arenas,
            cuda_device,
            request.bindings,
            request.publisher,
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/atnagent/{cuda_device}/transport-leases/quiesce")
    async def quiesce_atnagent_transport_leases(
        cuda_device: int,
        request: ProcessRef,
    ) -> AtnAgentTransportLeaseQuiesceResponse:
        """Stop admission and return the current lease-drain state for one AtnAgent.

        Raises:
            404: The AtnAgent registration does not exist.
            409: The process identity does not own the registration.
            503: Transport lease quiesce cannot currently progress.
        """

        return await asyncio.to_thread(control_plane.quiesce_atnagent_transport_leases, cuda_device, request)

    @app.post("/instance/register")
    async def register_instance(request: InstanceRankRegistration) -> Response:
        """Register one live Instance rank and its Transport and FFN profile contracts.

        Returns:
            Empty 204 response after registration.

        Raises:
            404: The configured Instance identity or rank is unknown.
            409: ABI, ffn_profile, placement, or process ownership conflicts.
            503: A predecessor registration has not completed cleanup.
        """

        await asyncio.to_thread(
            control_plane.register_instance,
            InstanceRankRegistrationState(
                instance=InstanceRankId(instance_id=request.instance_id, rank=request.rank),
                abi_version=request.abi_version,
                pid=request.pid,
                transport=request.transport,
                ffn_profile=request.ffn_profile,
                now=monotonic(),
            ),
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/instance/{instance_id:path}/deregister")
    async def deregister_instance(instance_id: str, rank: int, request: ProcessRef) -> Response:
        """Remove one authenticated Instance-rank registration.

        Returns:
            Empty 204 response after deregistration.

        Raises:
            404: The Instance rank is not registered.
            409: The process identity does not own that registration.
        """

        await asyncio.to_thread(control_plane.deregister_instance, instance_id, rank, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/instance/{instance_id:path}/initialized")
    async def publish_instance_initialized(
        instance_id: str,
        rank: int,
        request: InstanceRankInitializedPublication,
    ) -> Response:
        """Publish that one Instance rank initialized against the retained Fabric Plan.

        Returns:
            Empty 204 response after initialization is published.

        Raises:
            404: The Instance rank or Fabric generation does not exist.
            409: Owner, generation, or initialization facts conflict.
            503: Fabric is not ready to accept Instance-rank initialization.
        """

        await asyncio.to_thread(
            control_plane.publish_instance_initialized,
            instance_id,
            rank=rank,
            publication=request,
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/instance/{instance_id:path}/heartbeat")
    async def heartbeat_instance(instance_id: str, rank: int, request: ProcessRef) -> HeartbeatResponse:
        """Refresh one authenticated Instance-rank registration and return desired state.

        Raises:
            404: The Instance rank is not registered.
            409: The process identity does not own that registration.
        """

        return await asyncio.to_thread(control_plane.heartbeat_instance, instance_id, rank, request)

    @app.post("/instance/{instance_id:path}/transport-arena/acquire")
    async def acquire_instance_transport_arena(
        instance_id: str,
        rank: int,
        request: ProcessRef,
    ) -> TransportArenaHandle:
        """Acquire the admitted rank-local Transport arena lease for one Instance rank.

        Raises:
            404: The configured Instance or its Transport publication is unknown.
            409: Process, generation, or publication ownership conflicts.
            503: Registration, Fabric, or Transport admission is not ready.
        """

        return await asyncio.to_thread(
            control_plane.acquire_instance_transport_arena,
            instance_id,
            rank=rank,
            owner=request,
        )

    return app
