from __future__ import annotations

from http import HTTPStatus
from typing import cast

from tests.harness.service.daemon import (
    create_app,
    devagent_registration,
    devagent_transport_arenas,
    devagent_transport_arenas_path,
    expire_instance_registration,
    instance_registration,
    instance_transport_arena,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.config import XpoolConfig


def test_daemon_registration_flow() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)

    health = request(app, "GET", "/health")
    assert health.status_code == HTTPStatus.OK
    assert health.content == b""

    ready = request(app, "GET", "/ready").json()
    assert ready == {
        "ready": False,
        "mps_status": "online",
        "scopes": {"atn": False, "ffn": False},
        "cuda_devices": [0, 1],
        "devagents": [
            {"pid": None, "cuda_device": 0, "role": "atn", "status": "offline"},
            {"pid": None, "cuda_device": 1, "role": "ffn", "status": "offline"},
        ],
        "instances": [
            {
                "pid": None,
                "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
                "cuda_device": 0,
                "rank": 0,
                "status": "offline",
            }
        ],
    }

    assert request(app, "GET", "/config").json() == config.model_dump(mode="json")

    devagent0 = devagent_registration(cuda_device=0)
    devagent1 = devagent_registration(cuda_device=1)
    for payload in (devagent0, devagent1):
        response = request(app, "POST", "/devagent/register", json=payload)
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
            devagent_transport_arenas_path(0),
            json=devagent_transport_arenas(publisher=devagent0),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    ready = request(app, "GET", "/ready").json()
    assert ready == {
        "ready": True,
        "mps_status": "online",
        "scopes": {"atn": True, "ffn": True},
        "cuda_devices": [0, 1],
        "devagents": [
            {"pid": devagent0["pid"], "cuda_device": 0, "role": "atn", "status": "online"},
            {"pid": devagent1["pid"], "cuda_device": 1, "role": "ffn", "status": "online"},
        ],
        "instances": [
            {
                "pid": registration["pid"],
                "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
                "cuda_device": 0,
                "rank": 0,
                "status": "online",
            }
        ],
    }
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
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)

    response = request(app, "POST", "/config/check", json=config.model_dump(mode="json"))

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""


def test_daemon_reports_all_effective_config_differences() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    client_config = config.model_dump(mode="json")
    client_config["debug"]["transport_loopback"]["enable"] = True
    client_config["devices"]["ffn_cuda_devices"] = [2]

    response = request(app, "POST", "/config/check", json=client_config)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": {
            "kind": "conflict",
            "message": (
                "client xpool config differs from daemon config:\n"
                "- debug.transport_loopback.enable: client=true, daemon=false\n"
                "- devices.ffn_cuda_devices[0]: client=2, daemon=1"
            ),
        }
    }


def test_daemon_rejects_invalid_config_check_body() -> None:
    app = create_app(XpoolConfig.from_file("configs/xpool.example.toml"))

    response = request(app, "POST", "/config/check", json={"devices": {}})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_daemon_replaces_stale_instance_rank_registration() -> None:
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

    live_registration = instance_registration()
    assert request(app, "POST", "/instance/register", json=live_registration).status_code == HTTPStatus.NO_CONTENT

    duplicate_proc, duplicate_proc_id = start_sleeping_proc()
    try:
        duplicate = {**live_registration, "pid": duplicate_proc_id.pid}
        assert request(app, "POST", "/instance/register", json=duplicate).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(duplicate_proc)


def test_daemon_incremental_upsert_preserves_publication_when_existing_instance_reregisters() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    app = create_app(config)
    devagent = devagent_registration(cuda_device=0)
    assert request(app, "POST", "/devagent/register", json=devagent).status_code == HTTPStatus.NO_CONTENT
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

    first = devagent_transport_arenas(("a", 0), publisher=devagent)
    second = devagent_transport_arenas(("b", 0), publisher=devagent)
    second_binding = cast("list[dict[str, object]]", second["bindings"])[0]
    second_binding["handle"] = instance_transport_arena(rank=1)

    assert request(app, "POST", devagent_transport_arenas_path(0), json=first).status_code == HTTPStatus.NO_CONTENT
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

    assert request(app, "POST", devagent_transport_arenas_path(0), json=second).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            "/instance/register",
            json=instance_registration(instance_id="a", rank=0, element_size=2),
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
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    for rank in (0, 1):
        devagent = devagent_registration(cuda_device=rank)
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
            devagent_transport_arenas_path(0),
            json=devagent_transport_arenas(("m", 0)),
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


def test_daemon_rejects_client_supplied_create_time() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    registration = instance_registration()

    response = request(app, "POST", "/instance/register", json={**registration, "create_time": 1.0})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
