from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from xpool.config import XpoolConfig
from xpool.daemon import create_app
from xpool.runtime.mps import MpsHealthMonitor, MpsPreflight


def test_daemon_registration_flow() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config, mps_monitor=_mps_monitor(healthy=True), process_alive=lambda _pid: True)

    health = _request(app, "GET", "/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["mps"]["healthy"] is True

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is False
    assert ready["mps_healthy"] is True
    assert ready["missing_device_agents"] == ["cuda0", "cuda1"]
    assert ready["missing_instances"] == ["deepseek-ai/DeepSeek-V2-Lite-Chat"]

    config_view = _request(app, "GET", "/config").json()
    assert [agent["id"] for agent in config_view["derived"]["device_agents"]] == ["cuda0", "cuda1"]
    assert [instance["id"] for instance in config_view["derived"]["serving_instances"]] == [
        "deepseek-ai/DeepSeek-V2-Lite-Chat"
    ]

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
    assert ready["missing_device_agents"] == []
    assert ready["missing_instances"] == ["deepseek-ai/DeepSeek-V2-Lite-Chat"]

    instance_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 2345,
        },
    )
    assert instance_response.status_code == 200

    ready = _request(app, "GET", "/ready").json()
    assert ready["ready"] is True
    assert ready["registered_device_agents"] == ["cuda0", "cuda1"]
    assert ready["registered_instances"] == ["deepseek-ai/DeepSeek-V2-Lite-Chat"]
    assert ready["missing_device_agents"] == []
    assert ready["missing_instances"] == []

    duplicate_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 9999,
        },
    )
    assert duplicate_response.status_code == 409


def test_daemon_replaces_stale_instance_registration() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    live_pids = {1234}
    app = create_app(
        config,
        mps_monitor=_mps_monitor(healthy=True),
        process_alive=lambda pid: pid in live_pids,
    )

    first_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 1234,
        },
    )
    assert first_response.status_code == 200

    conflict_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 9999,
        },
    )
    assert conflict_response.status_code == 409

    live_pids.clear()
    live_pids.add(9999)
    replacement_response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 9999,
        },
    )
    assert replacement_response.status_code == 200


def test_daemon_ready_prunes_stale_registrations() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    live_pids = {1234, 1235, 2345}
    app = create_app(
        config,
        mps_monitor=_mps_monitor(healthy=True),
        process_alive=lambda pid: pid in live_pids,
    )

    for payload in (
        {"device_agent_id": "cuda0", "cuda_device": 0, "nvshmem_rank": 0, "pid": 1234},
        {"device_agent_id": "cuda1", "cuda_device": 1, "nvshmem_rank": 1, "pid": 1235},
    ):
        response = _request(app, "PUT", "/device-agents/register", json=payload)
        assert response.status_code == 200
    response = _request(
        app,
        "PUT",
        "/instances/register",
        json={
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "attention_cuda_devices": [0],
            "pid": 2345,
        },
    )
    assert response.status_code == 200
    assert _request(app, "GET", "/ready").json()["ready"] is True

    live_pids.remove(2345)
    ready = _request(app, "GET", "/ready").json()

    assert ready["ready"] is False
    assert ready["registered_instances"] == []
    assert ready["stale_instances"] == ["deepseek-ai/DeepSeek-V2-Lite-Chat"]
    assert ready["missing_instances"] == ["deepseek-ai/DeepSeek-V2-Lite-Chat"]


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
