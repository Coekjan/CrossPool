from __future__ import annotations

from http import HTTPStatus

from tests.harness.service.daemon import (
    create_app,
    devagent_registration,
    devagent_transport_arenas,
    devagent_transport_arenas_path,
    expire_devagent_registration,
    expire_instance_registration,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig, init_global_config
from xpool.service.daemon.app import create_daemon
from xpool.service.daemon.mps import MpsProbeResult


def test_daemon_readiness_filters_and_combines_requested_scopes() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent = devagent_registration(cuda_device=0)
    registration = instance_registration()

    assert request(app, "POST", "/devagent/register", json=devagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            devagent_transport_arenas_path(0),
            json=devagent_transport_arenas(publisher=devagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    ready = request(app, "GET", "/ready?scope=atn").json()

    assert ready["ready"] is True
    assert ready["mps_status"] == "online"
    assert ready["scopes"] == {"atn": True}
    assert ready["devagents"] == [
        {"pid": devagent["pid"], "cuda_device": 0, "role": "atn", "status": "online"},
    ]
    ffn = request(app, "GET", "/ready?scope=ffn").json()
    assert ffn["ready"] is False
    assert ffn["scopes"] == {"ffn": False}
    assert ffn["instances"] == []
    assert ffn["devagents"] == [
        {"pid": None, "cuda_device": 1, "role": "ffn", "status": "offline"},
    ]

    duplicate = request(app, "GET", "/ready?scope=atn&scope=atn").json()
    assert duplicate["scopes"] == {"atn": True}

    invalid = request(app, "GET", "/ready?scope=invalid")
    assert invalid.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_devagent_list_preserves_stale_registrations() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()

    try:
        assert (
            request(
                app,
                "POST",
                "/devagent/register",
                json=devagent_registration(cuda_device=0, pid=stale_proc_id.pid),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(stale_proc)
    expire_devagent_registration(app, cuda_device=0)

    response = request(app, "GET", "/devagents")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [devagent_registration(cuda_device=0, pid=stale_proc_id.pid)]


def test_instance_list_preserves_stale_registrations() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()

    try:
        assert (
            request(
                app,
                "POST",
                "/instance/register",
                json=instance_registration(pid=stale_proc_id.pid),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(stale_proc)
    expire_instance_registration(app, instance_id="deepseek-ai/DeepSeek-V2-Lite-Chat", rank=0)

    response = request(app, "GET", "/instances")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [instance_registration(pid=stale_proc_id.pid)]


def test_daemon_ready_marks_stale_live_registrations_stale_without_pruning() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()

    devagent0 = devagent_registration(cuda_device=0)
    devagent1 = devagent_registration(cuda_device=1)
    for payload in (devagent0, devagent1):
        assert request(app, "POST", "/devagent/register", json=payload).status_code == HTTPStatus.NO_CONTENT
    try:
        assert (
            request(
                app,
                "POST",
                "/instance/register",
                json=instance_registration(pid=stale_proc_id.pid),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        expire_instance_registration(app, instance_id="deepseek-ai/DeepSeek-V2-Lite-Chat", rank=0)
        ready = request(app, "GET", "/ready").json()
        health = request(app, "GET", "/health")
    finally:
        stop_proc(stale_proc)

    assert ready["ready"] is False
    assert ready["mps_status"] == "online"
    assert health.status_code == HTTPStatus.OK
    assert health.content == b""
    assert ready["devagents"] == [
        {"pid": devagent0["pid"], "cuda_device": 0, "role": "atn", "status": "online"},
        {"pid": devagent1["pid"], "cuda_device": 1, "role": "ffn", "status": "online"},
    ]
    assert ready["instances"] == [
        {
            "pid": stale_proc_id.pid,
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "cuda_device": 0,
            "rank": 0,
            "status": "stale",
        }
    ]


def test_mps_offline_blocks_every_scope_and_recovers_without_daemon_restart() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    results = [
        MpsProbeResult(False, "test controller is offline"),
        MpsProbeResult(True, "test controller recovered"),
    ]
    init_global_config(config=config)
    app = create_daemon(mps_status_provider=lambda: results.pop(0))
    devagent0 = devagent_registration(cuda_device=0)
    devagent1 = devagent_registration(cuda_device=1)
    registration = instance_registration()
    for payload in (devagent0, devagent1):
        assert request(app, "POST", "/devagent/register", json=payload).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            devagent_transport_arenas_path(0),
            json=devagent_transport_arenas(publisher=devagent0),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    offline = request(app, "GET", "/ready").json()

    assert offline["ready"] is False
    assert offline["mps_status"] == "offline"
    assert offline["scopes"] == {"atn": False, "ffn": False}
    assert all(entry["status"] == "online" for entry in offline["devagents"])
    assert all(entry["status"] == "online" for entry in offline["instances"])
    assert request(app, "GET", "/health").status_code == HTTPStatus.OK

    app.state.xpool_daemon_state.mps_cache_at = float("-inf")
    recovered = request(app, "GET", "/ready").json()

    assert recovered["ready"] is True
    assert recovered["mps_status"] == "online"
    assert recovered["scopes"] == {"atn": True, "ffn": True}


def test_daemon_preserves_stale_registration_but_blocks_stale_transport_arenas() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent_proc, devagent_proc_id = start_sleeping_proc()
    devagent = devagent_registration(cuda_device=0, pid=devagent_proc_id.pid)
    try:
        assert (
            request(
                app,
                "POST",
                "/devagent/register",
                json=devagent,
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert (
            request(app, "POST", "/instance/register", json=instance_registration()).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert (
            request(
                app,
                "POST",
                devagent_transport_arenas_path(0),
                json=devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=devagent),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(devagent_proc)
    expire_devagent_registration(app, cuda_device=0)

    ready = request(app, "GET", "/ready").json()
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )

    assert ready["devagents"][0] == {
        "pid": devagent_proc_id.pid,
        "cuda_device": 0,
        "role": "atn",
        "status": "offline",
    }
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention devagent process is not live",
    }
    replacement = devagent_registration(cuda_device=0)
    assert (
        request(
            app,
            "POST",
            "/devagent/register",
            json=replacement,
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            devagent_transport_arenas_path(0),
            json=devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=replacement),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    recovered = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )

    assert recovered.status_code == HTTPStatus.OK


def test_daemon_heartbeat_does_not_report_unimplemented_ffn_devagent_as_stale() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    assert request(app, "POST", "/devagent/register", json=devagent_registration(cuda_device=0)).status_code == (
        HTTPStatus.NO_CONTENT
    )
    registration = instance_registration()
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        "/instance/deepseek-ai/DeepSeek-V2-Lite-Chat/heartbeat?rank=0",
        json={"abi_version": ABI_VERSION, "pid": registration["pid"]},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["warnings"] == []


def test_daemon_reports_not_ready_when_devagent_upserts_before_registration() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json=devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0)),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "devagent must register before upserting transport arenas",
    }
