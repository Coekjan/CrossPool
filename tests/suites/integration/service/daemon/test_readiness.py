from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.service.daemon.control
from xpool.fabric import FabricPlan
from xpool.utils.mps import MpsProbeResult, MpsScope
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import (
    FakeMonotonicClock,
    activate_fabric_world,
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    register,
    request,
    run_watchdog,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_participant_list_preserves_stale_registrations(participant: str) -> None:
    config = synthetic_config()
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()
    if participant == "atnagent":
        registration = atnagent_registration(device=0, pid=stale_proc_id.pid)
        register_path = "/atnagent/register"
        list_path = "/atnagents"
    else:
        registration = instance_registration(pid=stale_proc_id.pid)
        register_path = "/instance/register"
        list_path = "/instances"

    try:
        assert register(app, register_path, registration).status_code == HTTPStatus.NO_CONTENT
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

    atnagent0 = atnagent_registration(device=0)
    assert register(app, "/atnagent/register", atnagent0).status_code == HTTPStatus.NO_CONTENT
    try:
        assert (
            register(
                app,
                "/instance/register",
                instance_registration(pid=stale_proc_id.pid),
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
    assert ready["atnagents"] == [{"pid": atnagent0["pid"], "device": 0, "status": "online"}]
    assert ready["instances"] == [
        {
            "pid": stale_proc_id.pid,
            "model_id": str(TEST_MODEL_ID),
            "device": 0,
            "rank": 0,
            "status": "stale",
        }
    ]


@pytest.mark.parametrize("initial_online", [False, None])
def test_mps_status_recovers_without_daemon_restart_while_fabric_is_initializing(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_daemon_dependencies: FakeMonotonicClock,
    initial_online: bool | None,
) -> None:
    config = synthetic_config()
    results = [
        MpsProbeResult(initial_online, None, "test controller is unavailable"),
        MpsProbeResult(True, 100, "test controller recovered"),
    ]
    monkeypatch.setattr(MpsScope, "probe", lambda self: results.pop(0))
    app = create_app(config)
    atnagent0 = atnagent_registration(device=0)
    registration = instance_registration()
    assert register(app, "/atnagent/register", atnagent0).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent_registration()).status_code == HTTPStatus.NO_CONTENT
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
    assert offline["mps_status"] == ("offline" if initial_online is False else None)
    assert offline["fabric_phase"] == "preparing_join"
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
    assert recovered["fabric_phase"] == "preparing_join"


def test_daemon_preserves_stale_registration_but_blocks_stale_transport_arenas() -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent_proc, atnagent_proc_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_proc_id.pid)
    ffnagent = ffnagent_registration()
    try:
        assert (
            register(
                app,
                "/atnagent/register",
                atnagent,
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
        assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
        plan = FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json())
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))
        assert (
            request(
                app,
                "POST",
                atnagent_transport_arenas_path(0),
                json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0), publisher=atnagent),
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
        instance_transport_arena_acquire_path(str(TEST_MODEL_ID), 0),
        json=process_ref(),
    )

    assert ready["atnagents"][0] == {
        "pid": atnagent_proc_id.pid,
        "device": 0,
        "status": "offline",
    }
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention atnagent process is not live",
    }


def test_daemon_reports_not_ready_when_atnagent_upserts_before_registration() -> None:
    config = synthetic_config()
    app = create_app(config)
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0)),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "atnagent must register before upserting transport arenas",
    }
