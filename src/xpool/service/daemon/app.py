"""FastAPI routes for the xpool daemon control plane."""

from __future__ import annotations

import asyncio
import time
from http import HTTPStatus
from importlib.metadata import version
from typing import Annotated

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse

from xpool.config import XpoolConfig, get_global_config
from xpool.service.daemon.mps import MpsStatusProvider, probe_mps_controller
from xpool.service.daemon.state import (
    AtnAgentRegistrationState,
    InstanceRegistrationState,
    InstanceUniqId,
    XpoolDaemonState,
)
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaDrainResponse,
    AtnAgentTransportArenaUpsertRequest,
    HeartbeatResponse,
    InstanceRegistration,
    ProcessHeartbeat,
    ProcessRef,
    ReadinessScope,
    ReadinessSnapshot,
    TransportArenaHandleRecord,
    XpoolDaemonErrorDetail,
)


def create_daemon(
    *,
    mps_status_provider: MpsStatusProvider = probe_mps_controller,
) -> FastAPI:
    """Create the FastAPI daemon application for the process-global config.

    Args:
        mps_status_provider: Bounded CUDA MPS controller probe used by
            readiness evaluation.

    Returns:
        Configured daemon application.
    """

    state = XpoolDaemonState(mps_status_provider=mps_status_provider)
    app = FastAPI(title="xpool daemon", version=version("xpool"))
    app.state.xpool_daemon_state = state

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
        return Response(status_code=HTTPStatus.OK)

    @app.get("/ready")
    async def ready(
        scope: Annotated[list[ReadinessScope] | None, Query()] = None,
    ) -> ReadinessSnapshot:
        selected = tuple(dict.fromkeys(scope or ())) or tuple(ReadinessScope)
        return await asyncio.to_thread(state.readiness_snapshot, selected)

    @app.get("/config")
    async def get_config() -> XpoolConfig:
        return get_global_config()

    @app.post("/config/check")
    async def check_config(request: XpoolConfig) -> Response:
        await asyncio.to_thread(state.check_config, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.get("/atnagents")
    async def list_atnagents() -> list[AtnAgentRegistration]:
        return await asyncio.to_thread(state.atnagent_registrations.views)

    @app.get("/instances")
    async def list_instances() -> list[InstanceRegistration]:
        return await asyncio.to_thread(state.instance_registrations.views)

    @app.post("/atnagent/register")
    async def register_atnagent(request: AtnAgentRegistration) -> Response:
        await asyncio.to_thread(
            state.register_atnagent,
            AtnAgentRegistrationState(
                cuda_device=request.cuda_device,
                abi_version=request.abi_version,
                pid=request.pid,
                now=time.monotonic(),
            ),
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/atnagent/{cuda_device}/heartbeat")
    async def heartbeat_atnagent(cuda_device: int, request: ProcessHeartbeat) -> HeartbeatResponse:
        return await asyncio.to_thread(state.heartbeat_atnagent, cuda_device, request)

    @app.post("/atnagent/{cuda_device}/transport-arenas")
    async def upsert_atnagent_transport_arenas(
        cuda_device: int,
        request: AtnAgentTransportArenaUpsertRequest,
    ) -> Response:
        await asyncio.to_thread(
            state.upsert_atnagent_transport_arenas,
            cuda_device,
            request.bindings,
            request.publisher,
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/atnagent/{cuda_device}/transport-arenas/drain")
    async def drain_atnagent_transport_arenas(
        cuda_device: int,
        request: ProcessRef,
    ) -> AtnAgentTransportArenaDrainResponse:
        return await asyncio.to_thread(state.drain_atnagent_transport_arenas, cuda_device, request)

    @app.post("/instance/register")
    async def register_instance(request: InstanceRegistration) -> Response:
        await asyncio.to_thread(
            state.register_instance,
            InstanceRegistrationState(
                instance=InstanceUniqId(instance_id=request.instance_id, rank=request.rank),
                abi_version=request.abi_version,
                pid=request.pid,
                transport=request.transport,
                now=time.monotonic(),
            ),
        )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/instance/{instance_id:path}/deregister")
    async def deregister_instance(instance_id: str, rank: int, request: ProcessRef) -> Response:
        await asyncio.to_thread(state.deregister_instance, instance_id, rank, request)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.post("/instance/{instance_id:path}/heartbeat")
    async def heartbeat_instance(instance_id: str, rank: int, request: ProcessHeartbeat) -> HeartbeatResponse:
        return await asyncio.to_thread(state.heartbeat_instance, instance_id, rank, request)

    @app.post("/instance/{instance_id:path}/transport-arena/acquire")
    async def acquire_instance_transport_arena(
        instance_id: str,
        rank: int,
        request: ProcessRef,
    ) -> TransportArenaHandleRecord:
        return await asyncio.to_thread(
            state.acquire_instance_transport_arena,
            instance_id,
            rank=rank,
            owner=request,
        )

    return app
