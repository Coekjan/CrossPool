from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.service.daemon.control
from xpool.config import XpoolConfig
from xpool.fabric import FabricPlan
from xpool.native import ABI_VERSION
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import (
    FakeMonotonicClock,
    activate_fabric_world,
    atnagent_registration,
    atnagent_transport_arena_bindings,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    atnagent_transport_leases_quiesce_path,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    register,
    request,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def test_daemon_quiesce_atnagent_transport_leases_blocks_new_acquires_and_reports_live_owner(
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    registration = instance_registration()
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
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
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path(str(TEST_MODEL_ID), 0),
            json=process_ref(registration),
        ).status_code
        == HTTPStatus.OK
    )
    deterministic_daemon_dependencies.advance(xpool.service.daemon.control.HEARTBEAT_WARNING_WATERMARK_S + 1.0)
    request(app, "POST", "/atnagent/0/heartbeat", json=process_ref(atnagent))

    quiesce = request(
        app,
        "POST",
        atnagent_transport_leases_quiesce_path(0),
        json=process_ref(atnagent),
    )
    republish = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0), publisher=atnagent),
    )
    request(
        app,
        "POST",
        f"/instance/{TEST_MODEL_ID}/heartbeat?rank=0",
        json=process_ref(registration),
    )
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(str(TEST_MODEL_ID), 0),
        json=process_ref(registration),
    )

    assert quiesce.status_code == HTTPStatus.OK
    assert quiesce.json()["in_use"] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "model_id": str(TEST_MODEL_ID),
            "rank": 0,
        }
    ]
    assert app.state.control_plane.registrations.instances.values()[0].proc.is_alive()
    assert republish.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert republish.json()["detail"] == {
        "kind": "not_ready",
        "message": "atnagent transport leases are quiescing",
    }
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention atnagent transport leases are quiescing",
    }


def test_daemon_lease_quiesce_reports_all_live_owners() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": "test/a", "path": "/models/a"},
                {"id": "test/b", "path": "/models/b"},
            ],
        }
    )
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration(model_ids=("test/a", "test/b"))
    owner_processes = [start_sleeping_proc(), start_sleeping_proc()]
    owner_ids = {owner_id.pid for owner, owner_id in owner_processes}
    try:
        assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
        assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
        for model_id, (owner, owner_id) in zip(("test/a", "test/b"), owner_processes, strict=True):
            registration = instance_registration(model_id=model_id, pid=owner_id.pid)
            assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
        plan = FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json())
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))

        bindings = atnagent_transport_arena_bindings(("test/a", 0), ("test/b", 0))
        bindings[1]["handle"] = {"handle": "01" * 64}
        assert (
            request(
                app,
                "POST",
                atnagent_transport_arenas_path(0),
                json={"publisher": process_ref(atnagent), "bindings": bindings},
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        for model_id, (owner, owner_id) in zip(("test/a", "test/b"), owner_processes, strict=True):
            assert (
                request(
                    app,
                    "POST",
                    instance_transport_arena_acquire_path(model_id, 0),
                    json={"pid": owner_id.pid, "abi_version": ABI_VERSION},
                ).status_code
                == HTTPStatus.OK
            )

        response = request(
            app,
            "POST",
            atnagent_transport_leases_quiesce_path(0),
            json=process_ref(atnagent),
        )

        assert response.status_code == HTTPStatus.OK
        assert {entry["pid"] for entry in response.json()["in_use"]} == owner_ids
        assert all(owner.poll() is None for owner, owner_id in owner_processes)
    finally:
        for owner, owner_id in owner_processes:
            stop_proc(owner)


def test_daemon_allows_stale_atnagent_to_quiesce_transport_leases(
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    registration = instance_registration()

    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
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
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path(str(TEST_MODEL_ID), 0),
            json=process_ref(registration),
        ).status_code
        == HTTPStatus.OK
    )
    deterministic_daemon_dependencies.advance(xpool.service.daemon.control.HEARTBEAT_WARNING_WATERMARK_S + 1.0)
    request(
        app,
        "POST",
        f"/instance/{TEST_MODEL_ID}/heartbeat?rank=0",
        json=process_ref(registration),
    )

    response = request(
        app,
        "POST",
        atnagent_transport_leases_quiesce_path(0),
        json=process_ref(atnagent),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["in_use"] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "model_id": str(TEST_MODEL_ID),
            "rank": 0,
        }
    ]
