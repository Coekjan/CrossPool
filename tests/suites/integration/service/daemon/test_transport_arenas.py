from __future__ import annotations

import threading
from http import HTTPStatus
from typing import cast

import httpx
import pytest

import xpool.service.daemon.registration
from tests.harness.config import TEST_MODEL_ID, synthetic_config, with_loopback
from tests.harness.service.daemon import (
    atnagent_registration,
    atnagent_transport_arena_bindings,
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
)
from xpool.config import LoopbackSite, XpoolConfig
from xpool.service.wire import ProcessRef

pytestmark = pytest.mark.usefixtures(deterministic_daemon_dependencies.__name__)


def test_daemon_requires_executable_fabric_for_transport_arena() -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)

    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    not_ready = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
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
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
    )
    assert publish_response.status_code == HTTPStatus.NO_CONTENT
    assert publish_response.content == b""

    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "Fabric generation is not executable",
    }


def test_daemon_allows_atnagent_loopback_transport_before_fabric() -> None:
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == instance_transport_arena(rank=0)


def test_daemon_rejects_ffnagent_loopback_transport_before_fabric() -> None:
    config = synthetic_config(loopback_site=LoopbackSite.FFNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "Fabric generation is not executable",
    }


def test_daemon_rejects_transport_lease_when_fabric_quiesces_during_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(app, "POST", "/ffnagent/register", json=ffnagent_registration(cuda_device=1)).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(app, "POST", "/instance/register", json=instance_registration(instance_id="m")).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.OK
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(("m", 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    validation_reached = threading.Event()
    resume_acquisition = threading.Event()
    original_validate = xpool.service.daemon.registration.InstanceRegistrationState.validate_process_ref

    def pause_after_validation(
        registration: xpool.service.daemon.registration.InstanceRegistrationState,
        owner: ProcessRef,
        *,
        context: str,
    ) -> None:
        original_validate(registration, owner, context=context)
        validation_reached.set()
        assert resume_acquisition.wait(timeout=2.0)

    monkeypatch.setattr(
        xpool.service.daemon.registration.InstanceRegistrationState,
        "validate_process_ref",
        pause_after_validation,
    )
    responses: list[httpx.Response] = []
    acquisition = threading.Thread(
        target=lambda: responses.append(
            request(
                app,
                "POST",
                instance_transport_arena_acquire_path("m", 0),
                json=process_ref(),
            )
        )
    )
    acquisition.start()
    assert validation_reached.wait(timeout=2.0)
    with app.state.control_plane.lock:
        app.state.control_plane.fabric_controller.quiesce(now=0.0)
    resume_acquisition.set()
    acquisition.join(timeout=2.0)

    assert not acquisition.is_alive()
    assert len(responses) == 1
    response = responses[0]
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "Fabric generation is not accepting Instance arena leases",
    }


def test_daemon_rejects_stale_arena_geometry_after_instance_reregister() -> None:
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    initial_registration = instance_registration(max_tokens=4)

    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=initial_registration).status_code == HTTPStatus.NO_CONTENT
    initial_arena = atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent)
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
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
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "published transport arena geometry is stale",
    }


def test_daemon_rejects_transport_topology_that_does_not_cover_atnagent_world() -> None:
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
        "instance transport TP-by-DP topology does not cover the configured AtnAgent world"
    )


def test_daemon_accepts_dp_transport_with_tp_fastest_rank_order() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)

    for rank in (0, 1):
        response = request(
            app,
            "POST",
            "/instance/register",
            json=instance_registration(
                instance_id="m",
                rank=rank,
                atn_tp_rank=0,
                atn_tp_size=1,
                atn_dp_rank=rank,
                atn_dp_size=2,
            ),
        )
        assert response.status_code == HTTPStatus.NO_CONTENT


def test_daemon_rejects_transport_coordinates_that_do_not_use_tp_fastest_order() -> None:
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
        json=instance_registration(instance_id="m", rank=0, atn_tp_rank=1, atn_tp_size=2),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"]["message"] == ("instance transport coordinates do not use TP-fastest rank order")


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


def test_daemon_returns_rank_local_attention_atnagent_transport_arena_for_instance() -> None:
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
    atnagents: dict[int, dict[str, int | float]] = {}
    for payload in (
        atnagent_registration(cuda_device=0),
        atnagent_registration(cuda_device=1),
    ):
        atnagents[int(payload["cuda_device"])] = payload
        assert request(app, "POST", "/atnagent/register", json=payload).status_code == HTTPStatus.NO_CONTENT
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
                atnagent_transport_arenas_path(rank),
                json=atnagent_transport_arenas(("m", rank)),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )

    rank0 = request(app, "POST", instance_transport_arena_acquire_path("m", 0), json=process_ref())
    rank1 = request(app, "POST", instance_transport_arena_acquire_path("m", 1), json=process_ref())

    assert rank0.status_code == HTTPStatus.OK
    assert rank1.status_code == HTTPStatus.OK
    assert rank0.json() == instance_transport_arena(rank=0)
    assert rank1.json() == instance_transport_arena(rank=1)


def test_daemon_rejects_atnagent_transport_arena_with_wrong_rank_device() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    assert (
        request(app, "POST", "/atnagent/register", json=atnagent_registration(cuda_device=0)).status_code
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
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas(("m", 1)),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "atnagent transport arena handle rank 1 belongs to CUDA device 1, not 0",
    }


def test_daemon_rejects_duplicate_atnagent_transport_arena() -> None:
    config = synthetic_config()
    app = create_app(config)
    assert (
        request(app, "POST", "/atnagent/register", json=atnagent_registration(cuda_device=0)).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    arenas = atnagent_transport_arena_bindings(
        (TEST_MODEL_ID, 0),
        (TEST_MODEL_ID, 0),
    )
    arenas[1]["handle"] = instance_transport_arena(rank=1)
    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json={"publisher": process_ref(), "bindings": arenas},
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "atnagent transport arenas contain duplicate instance-rank handle",
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
        request(app, "POST", "/atnagent/register", json=atnagent_registration(cuda_device=0)).status_code
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

    first = atnagent_transport_arenas(("a", 0))
    second = atnagent_transport_arenas(("b", 0))
    assert request(app, "POST", atnagent_transport_arenas_path(0), json=first).status_code == HTTPStatus.NO_CONTENT
    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=second,
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "atnagent transport arenas contain duplicate arena handle",
    }


def test_daemon_rejects_empty_atnagent_transport_arena_upsert() -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json={"publisher": process_ref(atnagent), "bindings": []},
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "atnagent transport arena upsert must not be empty",
    }
