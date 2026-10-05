from __future__ import annotations

from http import HTTPStatus

import pytest

from xpool.fabric import FabricPlan
from xpool.native import ABI_VERSION
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import (
    ProcUniqId,
    activate_fabric_world,
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
    register,
    request,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def test_daemon_deregisters_instance_rank_owned_by_process() -> None:
    config = synthetic_config()
    app = create_app(config)
    registration = instance_registration()
    model_id = str(registration["model_id"])

    assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "GET", "/instances").json() == [registration]

    response = request(
        app,
        "POST",
        f"/instance/{model_id}/deregister?rank={registration['rank']}",
        json={"pid": registration["pid"], "abi_version": registration["abi_version"]},
    )

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""
    assert request(app, "GET", "/instances").json() == []

    repeated = request(
        app,
        "POST",
        f"/instance/{model_id}/deregister?rank={registration['rank']}",
        json={"pid": registration["pid"], "abi_version": registration["abi_version"]},
    )

    assert repeated.status_code == HTTPStatus.NOT_FOUND
    assert repeated.json()["detail"] == {
        "kind": "not_found",
        "message": "registration is not registered",
    }


def test_daemon_rejects_instance_deregister_from_another_process() -> None:
    config = synthetic_config()
    app = create_app(config)
    registration = instance_registration()
    other_proc, other_proc_id = start_sleeping_proc()
    try:
        assert register(app, "/instance/register", registration).status_code == HTTPStatus.NO_CONTENT

        response = request(
            app,
            "POST",
            f"/instance/{registration['model_id']}/deregister?rank={registration['rank']}",
            json={"pid": other_proc_id.pid, "abi_version": registration["abi_version"]},
        )
    finally:
        stop_proc(other_proc)

    assert response.status_code == HTTPStatus.CONFLICT
    assert request(app, "GET", "/instances").json() == [registration]


def test_daemon_heartbeat_rejects_reused_pid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    config = synthetic_config()
    app = create_app(config)
    registration = atnagent_registration(device=0)
    assert register(app, "/atnagent/register", registration).status_code == HTTPStatus.NO_CONTENT
    proc_id = ProcUniqId.current()

    class ReusedPidProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def create_time(self) -> float:
            return proc_id.create_time + 1

    monkeypatch.setattr("xpool.utils.procs.psutil.Process", ReusedPidProcess)

    response = request(
        app,
        "POST",
        "/atnagent/0/heartbeat",
        json={"abi_version": ABI_VERSION, "pid": registration["pid"]},
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "heartbeat process identity no longer matches registration",
    }


def test_daemon_rejects_transport_arenas_from_non_owner_atnagent() -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    non_owner = {**atnagent, "pid": int(atnagent["pid"]) + 1}
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0), publisher=non_owner),
    )
    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "transport arena publisher pid does not match registration pid",
    }


def test_daemon_preserves_atnagent_transport_arenas_after_same_process_reregister() -> None:
    config = synthetic_config()
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
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

    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(str(TEST_MODEL_ID), 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == instance_transport_arena(rank=0)


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_daemon_rejects_dead_registration_pid(participant: str) -> None:
    config = synthetic_config()
    app = create_app(config)
    if participant == "atnagent":
        path = "/atnagent/register"
        registration = atnagent_registration(device=0, pid=999_999_999)
    else:
        path = "/instance/register"
        registration = instance_registration(pid=999_999_999)

    response = register(app, path, registration)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {"kind": "conflict", "message": "registering pid 999999999 is not live"}


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_daemon_rejects_registration_abi_mismatch(participant: str) -> None:
    config = synthetic_config()
    app = create_app(config)
    if participant == "atnagent":
        path = "/atnagent/register"
        registration = atnagent_registration(device=0, abi_version=ABI_VERSION + 1)
    else:
        path = "/instance/register"
        registration = instance_registration(abi_version=ABI_VERSION + 1)

    response = register(app, path, registration)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": f"{participant} ABI version does not match daemon ABI",
    }
