from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Mapping

import httpx
from fastapi import FastAPI

from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig, init_global_config
from xpool.service.daemon.app import create_daemon
from xpool.service.daemon.mps import MpsProbeResult
from xpool.service.daemon.state import HEARTBEAT_WARNING_WATERMARK_S, InstanceUniqId
from xpool.utils.procs import ProcUniqId


def create_app(config: XpoolConfig) -> FastAPI:
    """Create a daemon app with deterministic online MPS readiness."""

    init_global_config(config=config)
    return create_daemon(
        mps_status_provider=lambda: MpsProbeResult(True, "test MPS controller is online"),
    )


def instance_registration(
    *,
    instance_id: str = "deepseek-ai/DeepSeek-V2-Lite-Chat",
    rank: int = 0,
    abi_version: int = ABI_VERSION,
    element_size: int = 4,
    atn_tp_size: int = 1,
    pid: int | None = None,
) -> dict[str, object]:
    pid = process_pid(pid)
    return {
        "instance_id": instance_id,
        "rank": rank,
        "abi_version": abi_version,
        "pid": pid,
        "transport": transport_requirements(
            element_size=element_size,
            atn_tp_rank=rank,
            atn_tp_size=atn_tp_size,
        ),
    }


def devagent_registration(
    *,
    cuda_device: int,
    pid: int | None = None,
    abi_version: int = ABI_VERSION,
) -> dict[str, int | float]:
    pid = process_pid(pid)
    return {
        "cuda_device": cuda_device,
        "abi_version": abi_version,
        "pid": pid,
    }


def devagent_transport_arenas_path(cuda_device: int) -> str:
    return f"/devagent/{cuda_device}/transport-arenas"


def devagent_transport_arenas_drain_path(cuda_device: int) -> str:
    return f"/devagent/{cuda_device}/transport-arenas/drain"


def instance_transport_arena_acquire_path(instance_id: str, rank: int) -> str:
    return f"/instance/{instance_id}/transport-arena/acquire?rank={rank}"


def process_ref(registration: Mapping[str, object] | None = None) -> dict[str, object]:
    if registration is None:
        return {"pid": process_pid(None), "abi_version": ABI_VERSION}
    return {"pid": registration["pid"], "abi_version": registration["abi_version"]}


def devagent_transport_arenas(
    *arenas: tuple[str, int],
    publisher: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "publisher": process_ref(publisher),
        "bindings": devagent_transport_arena_bindings(*arenas),
    }


def devagent_transport_arena_bindings(*arenas: tuple[str, int]) -> list[dict[str, object]]:
    if not arenas:
        arenas = (("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),)
    return [
        {"instance_id": instance_id, "rank": rank, "handle": instance_transport_arena(rank=rank)}
        for instance_id, rank in arenas
    ]


def transport_requirements(
    *,
    element_size: int = 4,
    atn_tp_rank: int = 0,
    atn_tp_size: int = 1,
) -> dict[str, int]:
    return {
        "element_size": element_size,
        "hidden_size": 4,
        "max_tokens": 8,
        "atn_tp_rank": atn_tp_rank,
        "atn_tp_size": atn_tp_size,
        "atn_dp_rank": 0,
        "atn_dp_size": 1,
    }


def instance_transport_arena(*, rank: int) -> dict[str, object]:
    return {
        "handle": f"{rank:02x}" * 64,
    }


def process_pid(pid: int | None) -> int:
    if pid is None:
        return ProcUniqId.current().pid
    return pid


def expire_devagent_registration(app: FastAPI, *, cuda_device: int) -> None:
    state = app.state.xpool_daemon_state
    registration = state.devagent_registrations.registrations[cuda_device]
    registration.last_seen_at -= HEARTBEAT_WARNING_WATERMARK_S + 1.0


def expire_instance_registration(app: FastAPI, *, instance_id: str, rank: int) -> None:
    state = app.state.xpool_daemon_state
    registration = state.instance_registrations.registrations[InstanceUniqId(instance_id=instance_id, rank=rank)]
    registration.last_seen_at -= HEARTBEAT_WARNING_WATERMARK_S + 1.0


def start_sleeping_proc() -> tuple[subprocess.Popen[bytes], ProcUniqId]:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, ProcUniqId(process.pid)


def stop_proc(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def request(app: FastAPI, method: str, path: str, *, json: object | None = None) -> httpx.Response:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=json)

    return asyncio.run(run_request())
