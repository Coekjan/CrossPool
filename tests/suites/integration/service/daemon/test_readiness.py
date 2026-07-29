from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.service.daemon.control
from tests.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from tests.harness.support.service.daemon import (
    FakeMonotonicClock,
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    run_watchdog,
    start_sleeping_proc,
    stop_proc,
)
from xpool.config import LoopbackSite
from xpool.service.daemon.mps import MpsProbeResult

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_participant_list_preserves_stale_registrations(participant: str) -> None:
    config = synthetic_config()
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()
    if participant == "atnagent":
        registration = atnagent_registration(cuda_device=0, pid=stale_proc_id.pid)
        register_path = "/atnagent/register"
        list_path = "/atnagents"
    else:
        registration = instance_registration(pid=stale_proc_id.pid)
        register_path = "/instance/register"
        list_path = "/instances"

    try:
        assert request(app, "POST", register_path, json=registration).status_code == HTTPStatus.NO_CONTENT
    finally:
        stop_proc(stale_proc)
    response = request(app, "GET", list_path)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [registration]


def test_daemon_ready_marks_stale_live_registrations_stale_without_pruning(
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config()
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
        deterministic_daemon_dependencies.advance(xpool.service.daemon.control.HEARTBEAT_WARNING_WATERMARK_S + 1.0)
        request(app, "POST", "/atnagent/0/heartbeat", json=process_ref(atnagent0))
        run_watchdog(app)
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
            "instance_id": TEST_MODEL_ID,
            "cuda_device": 0,
            "rank": 0,
            "status": "stale",
        }
    ]


def test_mps_status_recovers_without_daemon_restart_while_fabric_is_initializing(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config()
    results = [
        MpsProbeResult(False, "test controller is offline"),
        MpsProbeResult(True, "test controller recovered"),
    ]
    monkeypatch.setattr(xpool.service.daemon.control, "probe_mps_controller", lambda: results.pop(0))
    app = create_app(config)
    atnagent0 = atnagent_registration(cuda_device=0)
    registration = instance_registration()
    assert request(app, "POST", "/atnagent/register", json=atnagent0).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/ffnagent/register", json=ffnagent_registration()).status_code == HTTPStatus.NO_CONTENT
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
    assert offline["fabric_phase"] == "joining"
    assert offline["transport_ready"] is True
    assert offline["instances_initialized"] is False
    assert all(entry["status"] == "online" for entry in offline["atnagents"])
    assert all(entry["status"] == "online" for entry in offline["instances"])
    assert request(app, "GET", "/health").status_code == HTTPStatus.OK

    deterministic_daemon_dependencies.advance(xpool.service.daemon.control.MPS_READINESS_CACHE_S + 1.0)
    run_watchdog(app)
    recovered = request(app, "GET", "/ready").json()

    assert recovered["ready"] is False
    assert recovered["mps_status"] == "online"
    assert recovered["fabric_phase"] == "joining"


def test_daemon_preserves_stale_registration_but_blocks_stale_transport_arenas() -> None:
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
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
                json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(atnagent_proc)
    run_watchdog(app)
    ready = request(app, "GET", "/ready").json()
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
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
            json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=replacement),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    recovered = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert recovered.status_code == HTTPStatus.OK


def test_daemon_reports_not_ready_when_atnagent_upserts_before_registration() -> None:
    config = synthetic_config()
    app = create_app(config)
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((TEST_MODEL_ID, 0)),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "atnagent must register before upserting transport arenas",
    }
