from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from xpool.config import XpoolConfig
from xpool.daemon import create_app
from xpool.runtime.mps import MpsHealthMonitor, MpsPreflight


def test_daemon_registration_flow() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config, mps_monitor=_mps_monitor(healthy=True))

    health = _request(app, "GET", "/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["mps"]["healthy"] is True

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is False
    assert ready["mps_healthy"] is True

    agent_response = _request(
        app,
        "PUT",
        "/device-agents/register",
        json={"device_agent_id": "cuda0", "cuda_device": 0, "nvshmem_rank": 0, "pid": 1234},
    )
    assert agent_response.status_code == 200
    agent_response = _request(
        app,
        "PUT",
        "/device-agents/register",
        json={"device_agent_id": "cuda1", "cuda_device": 1, "nvshmem_rank": 1, "pid": 1235},
    )
    assert agent_response.status_code == 200

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is False
    assert ready["registered_device_agents"] == ["cuda0", "cuda1"]
    assert ready["registered_instances"] == []

    instance_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-v2-lite-chat",
            "model_id": "deepseek-v2-lite-chat",
            "attention_cuda_devices": [0],
            "pid": 2345,
        },
    )
    assert instance_response.status_code == 200

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is True
    assert ready["registered_device_agents"] == ["cuda0", "cuda1"]
    assert ready["registered_instances"] == ["deepseek-v2-lite-chat"]


def test_daemon_health_fails_when_mps_is_unhealthy() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config, mps_monitor=_mps_monitor(healthy=False))

    health = _request(app, "GET", "/health")
    assert health.status_code == 503
    assert health.json()["status"] == "unhealthy"
    assert health.json()["mps"]["healthy"] is False

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is False
    assert ready["mps_healthy"] is False


def _request(app: FastAPI, method: str, path: str, *, json: object | None = None) -> httpx.Response:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=json)

    return asyncio.run(run_request())


def _mps_monitor(*, healthy: bool) -> MpsHealthMonitor:
    status = MpsPreflight(
        required=True,
        control_binary="/usr/bin/nvidia-cuda-mps-control",
        pipe_directory=None,
        control_binary_found=True,
        control_daemon_reachable=healthy,
        healthy=healthy,
        checked_at_unix_s=1.0,
        message="MPS control daemon is reachable" if healthy else "MPS control daemon is not reachable",
    )
    return MpsHealthMonitor(detector=lambda: status, initial=status)
