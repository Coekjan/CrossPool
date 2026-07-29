from __future__ import annotations

import threading
from http import HTTPStatus

import pytest

import xpool.service.daemon.control
from tests.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config, with_loopback
from tests.harness.support.service.daemon import (
    FakeMonotonicClock,
    ProcUniqId,
    atnagent_registration,
    atnagent_transport_arena_bindings,
    atnagent_transport_arenas,
    atnagent_transport_arenas_path,
    atnagent_transport_leases_quiesce_path,
    create_app,
    deterministic_daemon_dependencies,
    instance_registration,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.abi import ABI_VERSION
from xpool.config import LoopbackSite, XpoolConfig

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def test_daemon_replacement_generation_terminates_all_live_lease_owners_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = with_loopback(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [
                    {"id": "a", "path": "/models/a"},
                    {"id": "b", "path": "/models/b"},
                ],
            }
        ),
        LoopbackSite.ATNAGENT,
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    owner_processes = [start_sleeping_proc(), start_sleeping_proc()]
    original_terminate_tree = ProcUniqId.terminate_tree
    owner_barrier = threading.Barrier(len(owner_processes))

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        if self.pid in {owner_id.pid for owner, owner_id in owner_processes}:
            owner_barrier.wait(timeout=2.0)
        return original_terminate_tree(self, term_grace_s=term_grace_s)

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_tree)
    atnagent = atnagent_registration(cuda_device=0, pid=atnagent_id.pid)
    try:
        assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
        for instance_id, (owner, owner_id) in zip(("a", "b"), owner_processes, strict=True):
            registration = instance_registration(instance_id=instance_id, pid=owner_id.pid)
            assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT

        bindings = atnagent_transport_arena_bindings(("a", 0), ("b", 0))
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
        for instance_id, (owner, owner_id) in zip(("a", "b"), owner_processes, strict=True):
            assert (
                request(
                    app,
                    "POST",
                    instance_transport_arena_acquire_path(instance_id, 0),
                    json={"pid": owner_id.pid, "abi_version": ABI_VERSION},
                ).status_code
                == HTTPStatus.OK
            )

        stop_proc(atnagent_process)
        replacement = atnagent_registration(cuda_device=0)
        response = request(app, "POST", "/atnagent/register", json=replacement)

        assert response.status_code == HTTPStatus.NO_CONTENT
        assert request(app, "GET", "/instances").json() == []
    finally:
        if atnagent_process.poll() is None:
            stop_proc(atnagent_process)
        for owner, owner_id in owner_processes:
            if owner.poll() is None:
                stop_proc(owner)


def test_daemon_quiesce_atnagent_transport_leases_blocks_new_acquires_and_terminates_stale_owner(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    registration = instance_registration()
    terminated: list[tuple[int, float]] = []

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        terminated.append((self.pid, term_grace_s))
        return False

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_tree)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
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
        json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
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
        instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
        json=process_ref(registration),
    )

    assert quiesce.status_code == HTTPStatus.OK
    assert quiesce.json()["in_use"] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "instance_id": TEST_MODEL_ID,
            "rank": 0,
        }
    ]
    assert terminated == [(registration["pid"], xpool.service.daemon.control.HEARTBEAT_WARNING_WATERMARK_S)]
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


def test_daemon_lease_quiesce_terminates_all_live_owners_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = with_loopback(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [
                    {"id": "a", "path": "/models/a"},
                    {"id": "b", "path": "/models/b"},
                ],
            }
        ),
        LoopbackSite.ATNAGENT,
    )
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    owner_processes = [start_sleeping_proc(), start_sleeping_proc()]
    owner_ids = {owner_id.pid for owner, owner_id in owner_processes}
    owner_barrier = threading.Barrier(len(owner_processes))
    terminated: list[int] = []

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        assert term_grace_s == xpool.service.daemon.control.HEARTBEAT_WARNING_WATERMARK_S
        if self.pid in owner_ids:
            terminated.append(self.pid)
            owner_barrier.wait(timeout=2.0)
        return False

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_tree)
    try:
        assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
        for instance_id, (owner, owner_id) in zip(("a", "b"), owner_processes, strict=True):
            registration = instance_registration(instance_id=instance_id, pid=owner_id.pid)
            assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT

        bindings = atnagent_transport_arena_bindings(("a", 0), ("b", 0))
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
        for instance_id, (owner, owner_id) in zip(("a", "b"), owner_processes, strict=True):
            assert (
                request(
                    app,
                    "POST",
                    instance_transport_arena_acquire_path(instance_id, 0),
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
        assert set(terminated) == owner_ids
    finally:
        for owner, owner_id in owner_processes:
            stop_proc(owner)


def test_daemon_allows_stale_atnagent_to_quiesce_transport_leases(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = synthetic_config(loopback_site=LoopbackSite.ATNAGENT)
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    registration = instance_registration()

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        return False

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_tree)
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas((TEST_MODEL_ID, 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path(TEST_MODEL_ID, 0),
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
            "instance_id": TEST_MODEL_ID,
            "rank": 0,
        }
    ]
