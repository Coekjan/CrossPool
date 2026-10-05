from __future__ import annotations

from http import HTTPStatus

import pytest

from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import (
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    register,
    request,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


@pytest.mark.parametrize("physical_order", [(0, 2), (1, 0)], ids=["different-device", "reversed-order"])
def test_config_check_requires_matching_ordered_visibility(physical_order: tuple[int, ...]) -> None:
    config = synthetic_config()
    app = create_app(config)
    response = request(
        app,
        "POST",
        "/config/check",
        json={
            "config": config.model_dump(mode="json"),
            "visible_devices": [f"GPU-00000000-0000-0000-0000-{device:012x}" for device in physical_order],
        },
    )
    assert response.status_code == HTTPStatus.CONFLICT
    assert "visible_devices" in response.json()["detail"]["message"]


def test_daemon_registration_flow() -> None:
    config = synthetic_config()
    app = create_app(config)

    health = request(app, "GET", "/health")
    assert health.status_code == HTTPStatus.OK
    assert health.content == b""

    ready = request(app, "GET", "/ready").json()
    assert ready == {
        "ready": False,
        "generation": None,
        "fabric_phase": None,
        "fabric_invocation_failure": None,
        "fabric_owner_failure": None,
        "fabric_control_failure": None,
        "transport_ready": False,
        "instances_initialized": False,
        "mps_status": "online",
        "devices": [0, 1],
        "atnagents": [{"pid": None, "device": 0, "status": "offline"}],
        "ffnagents": [{"pid": None, "device": 1, "status": "offline"}],
        "instances": [
            {
                "pid": None,
                "model_id": str(TEST_MODEL_ID),
                "device": 0,
                "rank": 0,
                "status": "offline",
            }
        ],
    }

    assert request(app, "GET", "/config").json() == config.model_dump(mode="json")

    atnagent0 = atnagent_registration(device=0)
    response = register(app, "/atnagent/register", atnagent0)
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""

    ffnagent0 = ffnagent_registration()
    response = register(app, "/ffnagent/register", ffnagent0)
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""

    registration = instance_registration()
    response = register(app, "/instance/register", registration)
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(publisher=atnagent0),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    ready = request(app, "GET", "/ready").json()
    assert ready["ready"] is False
    assert ready["generation"] is not None
    assert ready["fabric_phase"] == "preparing_join"
    assert ready["fabric_invocation_failure"] is None
    assert ready["fabric_owner_failure"] is None
    assert ready["fabric_control_failure"] is None
    assert ready["transport_ready"] is True
    assert ready["instances_initialized"] is False
    assert ready["mps_status"] == "online"
    assert ready["devices"] == [0, 1]
    assert ready["atnagents"] == [{"pid": atnagent0["pid"], "device": 0, "status": "online"}]
    assert ready["ffnagents"] == [{"pid": ffnagent0["pid"], "device": 1, "status": "online"}]
    assert ready["instances"] == [
        {
            "pid": registration["pid"],
            "model_id": str(TEST_MODEL_ID),
            "device": 0,
            "rank": 0,
            "status": "online",
        }
    ]
    assert request(app, "GET", "/instances").json() == [registration]
    health = request(app, "GET", "/health")
    assert health.status_code == HTTPStatus.OK
    assert health.content == b""

    duplicate_proc, duplicate_proc_id = start_sleeping_proc()
    try:
        duplicate = {**registration, "pid": duplicate_proc_id.pid}
        assert register(app, "/instance/register", duplicate).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(duplicate_proc)


def test_daemon_accepts_matching_effective_config() -> None:
    config = synthetic_config()
    app = create_app(config)

    response = request(
        app,
        "POST",
        "/config/check",
        json={
            "config": config.model_dump(mode="json"),
            "visible_devices": list(app.state.control_plane.device_uuids),
        },
    )

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""


def test_instance_registration_rejects_negative_runtime_headroom() -> None:
    app = create_app(synthetic_config())
    registration = instance_registration()
    registration["atn_runtime_headroom_bytes"] = -1

    response = register(app, "/instance/register", registration)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_instance_reregistration_rejects_different_runtime_headroom() -> None:
    app = create_app(synthetic_config())
    registration = instance_registration(atn_runtime_headroom_bytes=1024)
    assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
    registration["atn_runtime_headroom_bytes"] = 2048

    response = register(app, "/instance/register", registration)

    assert response.status_code == HTTPStatus.CONFLICT


def test_daemon_reports_all_effective_config_differences() -> None:
    config = synthetic_config()
    app = create_app(config)
    client_config = config.model_dump(mode="json")
    client_config["daemon"]["port"] = 9811
    client_config["atn"]["device_memory_utilization"] = 0.8

    response = request(
        app,
        "POST",
        "/config/check",
        json={
            "config": client_config,
            "visible_devices": list(app.state.control_plane.device_uuids),
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": {
            "kind": "conflict",
            "message": (
                "client xpool config differs from daemon config:\n"
                "- atn.device_memory_utilization: client=0.8, daemon=0.95\n"
                "- daemon.port: client=9811, daemon=9810"
            ),
        }
    }


def test_daemon_rejects_invalid_config_check_body() -> None:
    app = create_app(synthetic_config())

    response = request(app, "POST", "/config/check", json={"atn": {}})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_daemon_replaces_stale_instance_rank_registration() -> None:
    config = synthetic_config()
    app = create_app(config)
    stale_proc, stale_proc_id = start_sleeping_proc()

    try:
        assert (
            register(
                app,
                "/instance/register",
                instance_registration(pid=stale_proc_id.pid),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    finally:
        stop_proc(stale_proc)
    live_registration = instance_registration()
    assert register(app, "/instance/register", live_registration).status_code == HTTPStatus.NO_CONTENT

    duplicate_proc, duplicate_proc_id = start_sleeping_proc()
    try:
        duplicate = {**live_registration, "pid": duplicate_proc_id.pid}
        assert register(app, "/instance/register", duplicate).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(duplicate_proc)
