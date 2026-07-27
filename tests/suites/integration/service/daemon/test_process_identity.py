from __future__ import annotations

from http import HTTPStatus

import pytest

from tests.harness.config import TEST_MODEL_ID, synthetic_config
from tests.harness.service.daemon import (
    ProcUniqId,
    atnagent_registration,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    create_app,
    deterministic_daemon_dependencies,
    instance_registration,
    instance_transport_arena,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.abi import ABI_VERSION
from xpool.config import LoopbackSite

pytestmark = pytest.mark.usefixtures(deterministic_daemon_dependencies.__name__)


def test_daemon_deregisters_instance_rank_owned_by_process() -> None:
    config = synthetic_config()
    app = create_app(config)
    registration = instance_registration()
    instance_id = str(registration["instance_id"])

    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "GET", "/instances").json() == [registration]

    response = request(
        app,
        "POST",
        f"/instance/{instance_id}/deregister?rank={registration['rank']}",
        json={"pid": registration["pid"], "abi_version": registration["abi_version"]},
    )

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert response.content == b""
    assert request(app, "GET", "/instances").json() == []

    repeated = request(
        app,
        "POST",
        f"/instance/{instance_id}/deregister?rank={registration['rank']}",
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
        assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT

        response = request(
            app,
            "POST",
            f"/instance/{registration['instance_id']}/deregister?rank={registration['rank']}",
            json={"pid": other_proc_id.pid, "abi_version": registration["abi_version"]},
        )
    finally:
        stop_proc(other_proc)

    assert response.status_code == HTTPStatus.CONFLICT
    assert request(app, "GET", "/instances").json() == [registration]


def test_daemon_heartbeat_rejects_reused_pid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    config = synthetic_config()
    app = create_app(config)
    registration = atnagent_registration(cuda_device=0)
    assert request(app, "POST", "/atnagent/register", json=registration).status_code == HTTPStatus.NO_CONTENT
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
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    non_owner = {**atnagent, "pid": int(atnagent["pid"]) + 1}
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=instance_registration()).status_code == HTTPStatus.NO_CONTENT

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=non_owner),
    )
    fetch = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(),
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": "transport arena publisher pid does not match registration pid",
    }
    assert fetch.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert fetch.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention atnagent has no transport arena handle for instance rank",
    }


def test_daemon_preserves_atnagent_transport_arenas_after_same_process_reregister() -> None:
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

    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
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
        registration = atnagent_registration(cuda_device=0, pid=999_999_999)
    else:
        path = "/instance/register"
        registration = instance_registration(pid=999_999_999)

    response = request(app, "POST", path, json=registration)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {"kind": "conflict", "message": "registering pid 999999999 is not live"}


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_daemon_rejects_registration_abi_mismatch(participant: str) -> None:
    config = synthetic_config()
    app = create_app(config)
    if participant == "atnagent":
        path = "/atnagent/register"
        registration = atnagent_registration(cuda_device=0, abi_version=ABI_VERSION + 1)
    else:
        path = "/instance/register"
        registration = instance_registration(abi_version=ABI_VERSION + 1)

    response = request(app, "POST", path, json=registration)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["detail"] == {
        "kind": "conflict",
        "message": f"{participant} ABI version does not match daemon ABI",
    }


def test_daemon_health_reports_process_liveness_only() -> None:
    config = synthetic_config()
    app = create_app(config)

    health = request(app, "GET", "/health")
    assert health.status_code == HTTPStatus.OK
    assert health.content == b""

    ready = request(app, "GET", "/ready").json()
    assert ready["ready"] is False
