from __future__ import annotations

from http import HTTPStatus
from typing import cast

import pytest

from tests.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config, with_loopback
from tests.harness.support.service.daemon import (
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    instance_transport_arena,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.config import LoopbackSite, XpoolConfig

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
        "fabric_protocol_failure": None,
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
    assert ready["fabric_phase"] == "joining"
    assert ready["fabric_invocation_failure"] is None
    assert ready["fabric_owner_failure"] is None
    assert ready["fabric_protocol_failure"] is None
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
    client_config["debug"]["loopback"] = {"enable": True, "site": "atnagent"}
    client_config["devices"]["ffn_cuda_devices"] = [2]

    response = request(app, "POST", "/config/check", json=client_config)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": {
            "kind": "conflict",
            "message": (
                "client xpool config differs from daemon config:\n"
                "- debug.loopback.enable: client=true, daemon=false\n"
                '- debug.loopback.site: client="atnagent", daemon=null\n'
                "- devices.ffn_cuda_devices[0]: client=2, daemon=1"
            ),
        }
    }


def test_daemon_rejects_invalid_config_check_body() -> None:
    app = create_app(synthetic_config())

    response = request(app, "POST", "/config/check", json={"devices": {}})

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


def test_daemon_incremental_upsert_preserves_publication_when_existing_instance_reregisters() -> None:
    config = with_loopback(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
            }
        ),
        LoopbackSite.ATNAGENT,
    )
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    for instance_id in ("a", "b"):
        assert (
            request(
                app,
                "POST",
                "/instance/register",
                json=instance_registration(instance_id=instance_id, rank=0),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )

    first = atnagent_transport_arenas(("a", 0), publisher=atnagent)
    second = atnagent_transport_arenas(("b", 0), publisher=atnagent)
    second_binding = cast("list[dict[str, object]]", second["bindings"])[0]
    second_binding["handle"] = instance_transport_arena(rank=1)

    assert request(app, "POST", atnagent_transport_arenas_path(0), json=first).status_code == HTTPStatus.NO_CONTENT
    assert request(
        app,
        "POST",
        instance_transport_arena_acquire_path("a", 0),
        json=process_ref(),
    ).json() == instance_transport_arena(rank=0)
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path("b", 0),
            json=process_ref(),
        ).status_code
        == HTTPStatus.SERVICE_UNAVAILABLE
    )
    assert (
        request(
            app,
            "POST",
            "/instance/a/deregister?rank=0",
            json=process_ref(),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    assert request(app, "POST", atnagent_transport_arenas_path(0), json=second).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            "/instance/register",
            json=instance_registration(instance_id="a", rank=0, max_tokens=16),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    stale = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("a", 0),
        json=process_ref(),
    )
    assert stale.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert stale.json()["detail"] == {
        "kind": "not_ready",
        "message": "published transport arena geometry is stale",
    }
    assert request(
        app,
        "POST",
        instance_transport_arena_acquire_path("b", 0),
        json=process_ref(),
    ).json() == instance_transport_arena(rank=1)


def test_daemon_rank_local_fetch_does_not_wait_for_other_ranks() -> None:
    config = with_loopback(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        ),
        LoopbackSite.ATNAGENT,
    )
    app = create_app(config)
    for rank in (0, 1):
        atnagent = atnagent_registration(cuda_device=rank)
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
        request(
            app,
            "POST",
            "/instance/register",
            json=instance_registration(
                instance_id="m",
                rank=0,
                atn_tp_size=2,
            ),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(("m", 0)),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    rank0 = request(app, "POST", instance_transport_arena_acquire_path("m", 0), json=process_ref())
    rank1 = request(app, "POST", instance_transport_arena_acquire_path("m", 1), json=process_ref())

    assert rank0.status_code == HTTPStatus.OK
    assert rank0.json() == instance_transport_arena(rank=0)
    assert rank1.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert rank1.json()["detail"] == {
        "kind": "not_ready",
        "message": "instance rank is not registered",
    }
