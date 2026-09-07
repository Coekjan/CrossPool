from __future__ import annotations

from http import HTTPStatus

import pytest

from tests.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from tests.harness.support.service.daemon import (
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    request,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


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
        "cuda_devices": [0, 1],
        "atnagents": [{"pid": None, "cuda_device": 0, "status": "offline"}],
        "ffnagents": [{"pid": None, "cuda_device": 1, "status": "offline"}],
        "instances": [
            {
                "pid": None,
                "instance_id": TEST_MODEL_ID,
                "cuda_device": 0,
                "rank": 0,
                "status": "offline",
            }
        ],
    }

    assert request(app, "GET", "/config").json() == config.model_dump(mode="json")

    atnagent0 = atnagent_registration(cuda_device=0)
    response = request(app, "POST", "/atnagent/register", json=atnagent0)
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""

    ffnagent0 = ffnagent_registration()
    response = request(app, "POST", "/ffnagent/register", json=ffnagent0)
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""

    registration = instance_registration()
    response = request(app, "POST", "/instance/register", json=registration)
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
    assert ready["cuda_devices"] == [0, 1]
    assert ready["atnagents"] == [{"pid": atnagent0["pid"], "cuda_device": 0, "status": "online"}]
    assert ready["ffnagents"] == [{"pid": ffnagent0["pid"], "cuda_device": 1, "status": "online"}]
    assert ready["instances"] == [
        {
            "pid": registration["pid"],
            "instance_id": TEST_MODEL_ID,
            "cuda_device": 0,
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
        assert request(app, "POST", "/instance/register", json=duplicate).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(duplicate_proc)


def test_daemon_accepts_matching_effective_config() -> None:
    config = synthetic_config()
    app = create_app(config)

    response = request(app, "POST", "/config/check", json=config.model_dump(mode="json"))

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""


def test_daemon_reports_all_effective_config_differences() -> None:
    config = synthetic_config()
    app = create_app(config)
    client_config = config.model_dump(mode="json")
    client_config["daemon"]["port"] = 9811
    client_config["ffn"]["devices"] = [2]

    response = request(app, "POST", "/config/check", json=client_config)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": {
            "kind": "conflict",
            "message": (
                "client xpool config differs from daemon config:\n"
                "- daemon.port: client=9811, daemon=9810\n"
                "- ffn.devices[0]: client=2, daemon=1"
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
    live_registration = instance_registration()
    assert request(app, "POST", "/instance/register", json=live_registration).status_code == HTTPStatus.NO_CONTENT

    duplicate_proc, duplicate_proc_id = start_sleeping_proc()
    try:
        duplicate = {**live_registration, "pid": duplicate_proc_id.pid}
        assert request(app, "POST", "/instance/register", json=duplicate).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(duplicate_proc)
