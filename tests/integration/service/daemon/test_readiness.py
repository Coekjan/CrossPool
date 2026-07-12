from __future__ import annotations

from http import HTTPStatus

from tests.harness.service.daemon import (
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    expire_atnagent_registration,
    expire_instance_registration,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.config import XpoolConfig, init_global_config
from xpool.service.daemon.app import create_daemon
from xpool.service.daemon.mps import MpsProbeResult


def test_daemon_readiness_filters_and_deduplicates_requested_scope() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    registration = instance_registration()

    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    ready = request(app, "GET", "/ready?scope=atn").json()

    assert ready["ready"] is True
    assert ready["mps_status"] == "online"
    assert ready["scopes"] == {"atn": True}
    assert ready["atnagents"] == [{"pid": atnagent["pid"], "cuda_device": 0, "status": "online"}]

    duplicate = request(app, "GET", "/ready?scope=atn&scope=atn").json()
    assert duplicate["scopes"] == {"atn": True}

    invalid = request(app, "GET", "/ready?scope=invalid")
    assert invalid.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_atnagent_list_preserves_stale_registrations() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()

    try:
        assert (
            request(
                app,
                "POST",
                "/atnagent/register",
                json=atnagent_registration(cuda_device=0, pid=stale_proc_id.pid),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(stale_proc)
    expire_atnagent_registration(app, cuda_device=0)

    response = request(app, "GET", "/atnagents")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [atnagent_registration(cuda_device=0, pid=stale_proc_id.pid)]


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

    atnagent0 = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent0).status_code == HTTPStatus.NO_CONTENT
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
    assert ready["atnagents"] == [{"pid": atnagent0["pid"], "cuda_device": 0, "status": "online"}]
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
    atnagent0 = atnagent_registration(cuda_device=0)
    registration = instance_registration()
    assert request(app, "POST", "/atnagent/register", json=atnagent0).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(publisher=atnagent0),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    offline = request(app, "GET", "/ready").json()

    assert offline["ready"] is False
    assert offline["mps_status"] == "offline"
    assert offline["scopes"] == {"atn": False}
    assert all(entry["status"] == "online" for entry in offline["atnagents"])
    assert all(entry["status"] == "online" for entry in offline["instances"])
    assert request(app, "GET", "/health").status_code == HTTPStatus.OK

    app.state.xpool_daemon_state.mps_cache_at = float("-inf")
    recovered = request(app, "GET", "/ready").json()

    assert recovered["ready"] is True
    assert recovered["mps_status"] == "online"
    assert recovered["scopes"] == {"atn": True}


def test_daemon_preserves_stale_registration_but_blocks_stale_transport_arenas() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    atnagent_proc, atnagent_proc_id = start_sleeping_proc()
    atnagent = atnagent_registration(cuda_device=0, pid=atnagent_proc_id.pid)
    try:
        assert (
            request(
                app,
                "POST",
                "/atnagent/register",
                json=atnagent,
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
                atnagent_transport_arenas_path(0),
                json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=atnagent),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(atnagent_proc)
    expire_atnagent_registration(app, cuda_device=0)

    ready = request(app, "GET", "/ready").json()
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )

    assert ready["atnagents"][0] == {
        "pid": atnagent_proc_id.pid,
        "cuda_device": 0,
        "status": "offline",
    }
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention atnagent process is not live",
    }
    replacement = atnagent_registration(cuda_device=0)
    assert (
        request(
            app,
            "POST",
            "/atnagent/register",
            json=replacement,
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=replacement),
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


def test_daemon_reports_not_ready_when_atnagent_upserts_before_registration() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0)),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "atnagent must register before upserting transport arenas",
    }
