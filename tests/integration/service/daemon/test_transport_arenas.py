from __future__ import annotations

from http import HTTPStatus
from typing import cast

from tests.harness.service.daemon import (
    create_app,
    devagent_registration,
    devagent_transport_arena_bindings,
    devagent_transport_arenas,
    devagent_transport_arenas_path,
    instance_registration,
    instance_transport_arena,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
)
from xpool.config import XpoolConfig


def test_daemon_brokers_rank_local_ffn_shim_transport_arena() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent = devagent_registration(cuda_device=0)

    assert (
        request(
            app,
            "POST",
            "/devagent/register",
            json=devagent,
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    not_ready = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )
    assert not_ready.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert not_ready.json()["detail"] == {
        "kind": "not_ready",
        "message": "instance rank is not registered",
    }

    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT
    publish_response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json=devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=devagent),
    )
    assert publish_response.status_code == HTTPStatus.NO_CONTENT
    assert publish_response.content == b""

    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == instance_transport_arena(rank=0)


def test_daemon_rejects_stale_arena_geometry_after_instance_reregister() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent = devagent_registration(cuda_device=0)
    initial_registration = instance_registration(element_size=2)

    assert request(app, "POST", "/devagent/register", json=devagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=initial_registration).status_code == HTTPStatus.NO_CONTENT
    initial_arena = devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=devagent)
    assert (
        request(
            app,
            "POST",
            devagent_transport_arenas_path(0),
            json=initial_arena,
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            f"/instance/{initial_registration['instance_id']}/deregister?rank={initial_registration['rank']}",
            json={"pid": initial_registration["pid"], "abi_version": initial_registration["abi_version"]},
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "published transport arena geometry is stale",
    }


def test_daemon_does_not_expose_raw_devagent_transport_arenas() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)

    response = request(app, "GET", "/devagent/0/transport-arenas")

    assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED


def test_daemon_rejects_wrapped_devagent_transport_arenas() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)

    response = request(app, "POST", devagent_transport_arenas_path(0), json={"arenas": []})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_daemon_rejects_transport_tp_geometry_that_disagrees_with_config() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)

    response = request(
        app,
        "POST",
        "/instance/register",
        json=instance_registration(instance_id="m", rank=0, atn_tp_size=1),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"]["message"] == (
        "instance transport TP rank/size does not match configured ATN rank placement"
    )


def test_daemon_rejects_dp_transport_until_reduce_scatter_is_supported() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    registration = instance_registration()
    registration["transport"] = {
        **cast(dict[str, int], registration["transport"]),
        "atn_dp_size": 2,
    }

    response = request(app, "POST", "/instance/register", json=registration)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"]["message"] == "instance transport DP must be rank 0 of size 1"


def test_daemon_rejects_cross_rank_transport_geometry_mismatch() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    rank0 = instance_registration(instance_id="m", rank=0, atn_tp_size=2)
    rank1 = instance_registration(instance_id="m", rank=1, atn_tp_size=2)
    rank1["transport"] = {
        **cast(dict[str, int], rank1["transport"]),
        "hidden_size": 8,
    }

    assert request(app, "POST", "/instance/register", json=rank0).status_code == HTTPStatus.NO_CONTENT
    response = request(app, "POST", "/instance/register", json=rank1)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"]["message"] == "instance transport attributes disagree across ranks"


def test_daemon_returns_rank_local_attention_devagent_transport_arena_for_instance() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    devagents: dict[int, dict[str, int | float]] = {}
    for payload in (
        devagent_registration(cuda_device=0),
        devagent_registration(cuda_device=1),
    ):
        devagents[int(payload["cuda_device"])] = payload
        assert request(app, "POST", "/devagent/register", json=payload).status_code == HTTPStatus.NO_CONTENT
    for rank in (0, 1):
        assert (
            request(
                app,
                "POST",
                "/instance/register",
                json=instance_registration(
                    instance_id="m",
                    rank=rank,
                    atn_tp_size=2,
                ),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
    for rank in (0, 1):
        assert (
            request(
                app,
                "POST",
                devagent_transport_arenas_path(rank),
                json=devagent_transport_arenas(("m", rank)),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )

    rank0 = request(app, "POST", instance_transport_arena_acquire_path("m", 0), json=process_ref())
    rank1 = request(app, "POST", instance_transport_arena_acquire_path("m", 1), json=process_ref())

    assert rank0.status_code == HTTPStatus.OK
    assert rank1.status_code == HTTPStatus.OK
    assert rank0.json() == instance_transport_arena(rank=0)
    assert rank1.json() == instance_transport_arena(rank=1)


def test_daemon_rejects_ffn_shim_transport_arenas_from_ffn_agent() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent = devagent_registration(cuda_device=1)
    assert (
        request(
            app,
            "POST",
            "/devagent/register",
            json=devagent,
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(1),
        json=devagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=devagent),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert "attention devagents" in response.text


def test_daemon_rejects_devagent_transport_arena_with_wrong_rank_device() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    assert (
        request(app, "POST", "/devagent/register", json=devagent_registration(cuda_device=0)).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            "/instance/register",
            json=instance_registration(instance_id="m", rank=1, atn_tp_size=2),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json=devagent_transport_arenas(("m", 1)),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "devagent transport arena handle rank 1 belongs to CUDA device 1, not 0",
    }


def test_daemon_rejects_duplicate_devagent_transport_arena() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    assert (
        request(app, "POST", "/devagent/register", json=devagent_registration(cuda_device=0)).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    arenas = devagent_transport_arena_bindings(
        ("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        ("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
    )
    arenas[1]["handle"] = instance_transport_arena(rank=1)
    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json={"publisher": process_ref(), "bindings": arenas},
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "devagent transport arenas contain duplicate instance-rank handle",
    }


def test_daemon_rejects_duplicate_transport_arena_handle() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    app = create_app(config)
    assert (
        request(app, "POST", "/devagent/register", json=devagent_registration(cuda_device=0)).status_code
        == HTTPStatus.NO_CONTENT
    )
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

    first = devagent_transport_arenas(("a", 0))
    second = devagent_transport_arenas(("b", 0))
    assert request(app, "POST", devagent_transport_arenas_path(0), json=first).status_code == HTTPStatus.NO_CONTENT
    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json=second,
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "devagent transport arenas contain duplicate arena handle",
    }


def test_daemon_rejects_empty_devagent_transport_arena_upsert() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    devagent = devagent_registration(cuda_device=0)
    assert request(app, "POST", "/devagent/register", json=devagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        devagent_transport_arenas_path(0),
        json={"publisher": process_ref(devagent), "bindings": []},
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "devagent transport arena upsert must not be empty",
    }
