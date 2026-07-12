from __future__ import annotations

import threading
from http import HTTPStatus

import pytest

from tests.harness.service.daemon import (
    HEARTBEAT_WARNING_WATERMARK_S,
    ProcUniqId,
    atnagent_registration,
    atnagent_transport_arena_bindings,
    atnagent_transport_arenas,
    atnagent_transport_arenas_drain_path,
    atnagent_transport_arenas_path,
    create_app,
    expire_atnagent_registration,
    expire_instance_registration,
    instance_registration,
    instance_transport_arena,
    instance_transport_arena_acquire_path,
    process_ref,
    request,
    start_sleeping_proc,
    stop_proc,
)
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig
from xpool.service.wire import AtnAgentTransportArenaDrainResponse, ProcessRef, TransportArenaHandleRecord


def test_daemon_replacement_generation_terminates_all_live_lease_owners_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [
                {"id": "a", "path": "/models/a"},
                {"id": "b", "path": "/models/b"},
            ],
        }
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


def test_daemon_drain_atnagent_transport_arenas_blocks_new_acquires_and_terminates_stale_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
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
            json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
            json=process_ref(registration),
        ).status_code
        == HTTPStatus.OK
    )
    expire_instance_registration(
        app,
        instance_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        rank=0,
    )

    drain = request(
        app,
        "POST",
        atnagent_transport_arenas_drain_path(0),
        json=process_ref(atnagent),
    )
    republish = request(
        app,
        "POST",
        atnagent_transport_arenas_path(0),
        json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=atnagent),
    )
    heartbeat = request(
        app,
        "POST",
        "/instance/deepseek-ai/DeepSeek-V2-Lite-Chat/heartbeat?rank=0",
        json=process_ref(registration),
    )
    response = request(
        app,
        "POST",
        instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
        json=process_ref(registration),
    )

    assert drain.status_code == HTTPStatus.OK
    assert drain.json()["in_use"] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "rank": 0,
        }
    ]
    assert terminated == [(registration["pid"], HEARTBEAT_WARNING_WATERMARK_S)]
    assert republish.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert republish.json()["detail"] == {
        "kind": "not_ready",
        "message": "atnagent transport arenas are terminating",
    }
    assert heartbeat.status_code == HTTPStatus.OK
    assert heartbeat.json()["warnings"] == [
        {
            "kind": "terminating_atnagent",
            "cuda_device": 0,
            "message": "AtnAgent on CUDA device 0 is terminating",
        },
    ]
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "kind": "not_ready",
        "message": "local attention atnagent transport arenas are terminating",
    }


def test_daemon_drain_terminates_all_live_lease_owners_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [
                {"id": "a", "path": "/models/a"},
                {"id": "b", "path": "/models/b"},
            ],
        }
    )
    app = create_app(config)
    atnagent = atnagent_registration(cuda_device=0)
    owner_processes = [start_sleeping_proc(), start_sleeping_proc()]
    owner_ids = {owner_id.pid for owner, owner_id in owner_processes}
    owner_barrier = threading.Barrier(len(owner_processes))
    terminated: list[int] = []

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        assert term_grace_s == HEARTBEAT_WARNING_WATERMARK_S
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
            atnagent_transport_arenas_drain_path(0),
            json=process_ref(atnagent),
        )

        assert response.status_code == HTTPStatus.OK
        assert {entry["pid"] for entry in response.json()["in_use"]} == owner_ids
        assert set(terminated) == owner_ids
    finally:
        for owner, owner_id in owner_processes:
            stop_proc(owner)


def test_daemon_drain_waits_for_inflight_transport_arena_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    app = create_app(config)
    state = app.state.xpool_daemon_state
    atnagent = atnagent_registration(cuda_device=0)
    registration = instance_registration()
    acquire_blocked = threading.Event()
    release_acquire = threading.Event()
    drain_entered = threading.Event()
    acquire_result: dict[str, object] = {}
    drain_result: dict[str, object] = {}

    def terminate_tree(self: ProcUniqId, *, term_grace_s: float) -> bool:
        return False

    original_acquire_transport_arena_lease = state.instance_registrations.acquire_transport_arena_lease

    def blocking_acquire_transport_arena_lease(*args: object, **kwargs: object) -> None:
        acquire_blocked.set()
        assert release_acquire.wait(timeout=5.0)
        original_acquire_transport_arena_lease(*args, **kwargs)

    def acquire() -> None:
        try:
            acquire_result["arena"] = state.acquire_instance_transport_arena(
                "deepseek-ai/DeepSeek-V2-Lite-Chat",
                rank=0,
                owner=ProcessRef.model_validate(process_ref(registration)),
            )
        except BaseException as exc:
            acquire_result["exc"] = exc

    def drain() -> None:
        drain_entered.set()
        try:
            drain_result["response"] = state.drain_atnagent_transport_arenas(
                0,
                ProcessRef.model_validate(process_ref(atnagent)),
            )
        except BaseException as exc:
            drain_result["exc"] = exc

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_tree)
    monkeypatch.setattr(
        state.instance_registrations,
        "acquire_transport_arena_lease",
        blocking_acquire_transport_arena_lease,
    )
    assert request(app, "POST", "/atnagent/register", json=atnagent).status_code == HTTPStatus.NO_CONTENT
    assert request(app, "POST", "/instance/register", json=registration).status_code == HTTPStatus.NO_CONTENT
    assert (
        request(
            app,
            "POST",
            atnagent_transport_arenas_path(0),
            json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    acquire_thread = threading.Thread(target=acquire)
    drain_thread = threading.Thread(target=drain)
    acquire_thread.start()
    assert acquire_blocked.wait(timeout=5.0)
    drain_thread.start()
    assert drain_entered.wait(timeout=5.0)
    drain_thread.join(timeout=0.2)
    assert drain_thread.is_alive()

    release_acquire.set()
    acquire_thread.join(timeout=5.0)
    drain_thread.join(timeout=5.0)

    assert not acquire_thread.is_alive()
    assert not drain_thread.is_alive()
    assert "exc" not in acquire_result
    assert "exc" not in drain_result
    arena = acquire_result["arena"]
    assert isinstance(arena, TransportArenaHandleRecord)
    assert arena.model_dump(mode="json") == instance_transport_arena(rank=0)
    response = drain_result["response"]
    assert isinstance(response, AtnAgentTransportArenaDrainResponse)
    assert [entry.model_dump(mode="json") for entry in response.in_use] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "rank": 0,
        }
    ]


def test_daemon_allows_stale_atnagent_to_drain_transport_arenas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
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
            json=atnagent_transport_arenas(("deepseek-ai/DeepSeek-V2-Lite-Chat", 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            instance_transport_arena_acquire_path("deepseek-ai/DeepSeek-V2-Lite-Chat", 0),
            json=process_ref(registration),
        ).status_code
        == HTTPStatus.OK
    )
    expire_atnagent_registration(app, cuda_device=0)

    response = request(
        app,
        "POST",
        atnagent_transport_arenas_drain_path(0),
        json=process_ref(atnagent),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["in_use"] == [
        {
            "pid": registration["pid"],
            "abi_version": registration["abi_version"],
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "rank": 0,
        }
    ]
