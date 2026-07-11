from __future__ import annotations

import threading
from http import HTTPStatus

import xpool.utils.procs as procs_module
from tests.harness.runtime.instance import (
    Instance,
    InstanceRegistration,
    ProcessHeartbeat,
    TransportArenaHandle,
    XpoolConfig,
    instance_module,
    patch_native_instance_ops,
    pytest,
    runtime_config,
    runtime_heartbeat,
    runtime_instance,
    transport_arena,
    transport_attributes,
)
from xpool.abi import ABI_VERSION
from xpool.service.wire import ControlPlaneWarning, HeartbeatResponse, ProcessRef
from xpool.utils import background as background_module


def test_instance_deregister_stops_heartbeat_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[object] = []

    class FakeThread(threading.Thread):
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

        def join(self, timeout: float | None = None) -> None:
            events.append(("join", timeout))
            self.alive = False

    monkeypatch.setattr(background_module.threading, "Thread", FakeThread)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRegistration) -> None:
            return None

        def deregister_instance(self, instance_id: str, *, rank: int, owner: object) -> None:
            events.append(("deregister", instance_id, rank, worker.stop_event.is_set(), owner))

    patch_native_instance_ops(monkeypatch, detach=lambda instance_index, rank: events.append(("detach", "m", rank)))
    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)

    instance = runtime_instance(config, monkeypatch)
    instance_module.get_instance().register_runtime(transport_attributes())
    instance_module.get_instance().start_heartbeat_worker()
    heartbeat_worker = instance.heartbeat_worker
    assert heartbeat_worker is not None
    worker = heartbeat_worker.worker
    instance_module.get_instance().deregister_runtime()

    assert events[0] == ("detach", "m", 0)
    assert events[1] == ("join", instance_module.INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S)
    assert isinstance(events[2], tuple)
    assert events[2][:4] == ("deregister", "m", 0, True)


def test_instance_start_registers_heartbeats_and_attaches(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []
    monkeypatch.setattr(Instance, "register_runtime", lambda self, transport: events.append("register"))
    monkeypatch.setattr(Instance, "start_heartbeat_worker", lambda self: events.append("heartbeat"))
    monkeypatch.setattr(Instance, "attach_arena_from_daemon", lambda self: events.append("attach"))
    monkeypatch.setattr(Instance, "start_transport_monitor", lambda self: events.append("monitor"))

    runtime_instance(config, monkeypatch)
    instance_module.get_instance().start_runtime(transport_attributes())

    assert events == ["register", "heartbeat", "attach", "monitor"]


def test_instance_start_deregisters_when_heartbeat_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []

    def fail_heartbeat(self: Instance) -> None:
        events.append("heartbeat")
        raise RuntimeError("heartbeat failed")

    monkeypatch.setattr(Instance, "register_runtime", lambda self, transport: events.append("register"))
    monkeypatch.setattr(Instance, "start_heartbeat_worker", fail_heartbeat)
    monkeypatch.setattr(Instance, "attach_arena_from_daemon", lambda self: events.append("attach"))
    monkeypatch.setattr(Instance, "start_transport_monitor", lambda self: events.append("monitor"))
    monkeypatch.setattr(Instance, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        runtime_instance(config, monkeypatch)
        instance_module.get_instance().start_runtime(transport_attributes())

    assert events == ["register", "heartbeat", "deregister"]


def test_instance_heartbeat_ignores_devagent_warning_for_another_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", instance_id, rank, heartbeat))
            return HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="stale_devagent",
                        cuda_device=0,
                        message="devagent 0 is stale",
                    )
                ]
            )

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("warning handling must not fetch transport arenas")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))

    runtime_heartbeat(config, monkeypatch, rank=1).step()

    assert events == [
        ("heartbeat", "m", 1, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_logs_local_terminating_devagent_without_self_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", rank, heartbeat))
            return HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="terminating_devagent",
                        cuda_device=0,
                        message="devagent is terminating",
                    )
                ]
            )

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("terminating devagent must not fetch transport arenas")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))

    runtime_heartbeat(config, monkeypatch).step()

    assert events == [
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_tolerates_local_stale_devagent_until_recovery_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", instance_id, rank, heartbeat))
            return HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="stale_devagent",
                        cuda_device=0,
                        message="devagent is stale",
                    )
                ]
            )

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("stale devagent warning must not fetch transport arenas")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))

    runtime_heartbeat(config, monkeypatch).step()

    assert events == [
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_fail_closes_after_local_stale_devagent_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", instance_id, rank, heartbeat))
            return HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="stale_devagent",
                        cuda_device=0,
                        message="devagent is stale",
                    )
                ]
            )

    monotonic_values = iter([0.0, instance_module.STALE_DEVAGENT_RECOVERY_GRACE_S + 1.0])

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(instance_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    with pytest.raises(SystemExit) as exc_info:
        heartbeat.step()

    assert exc_info.value.code == 1
    assert events == [
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_retries_daemon_failures_without_terminating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", rank, heartbeat))
            raise instance_module.XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert events == [
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_retries_malformed_status_without_terminating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", rank, heartbeat))
            raise instance_module.XpoolClientError(
                "status",
                "bad gateway",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert events == [
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_keeps_client_on_recoverable_failure_and_closes_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        heartbeat_count = 0

        def __init__(self) -> None:
            events.append("init")

        def close(self) -> None:
            events.append("close")
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", instance_id, rank, heartbeat))
            if FakeXpoolClient.heartbeat_count == 1:
                raise instance_module.XpoolClientError("transport", "daemon unavailable")
            return HeartbeatResponse(warnings=[])

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.stop()

    assert events == [
        "init",
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
        "close",
    ]


def test_instance_heartbeat_fail_closes_after_transport_error_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", rank, heartbeat))
            raise instance_module.XpoolClientError("transport", "daemon unavailable")

    monotonic_values = iter([0.0, instance_module.TRANSPORT_METADATA_RECOVERY_DEADLINE_S + 1.0])
    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(instance_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.code == 1
    assert events == [
        ("heartbeat", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_resets_transport_recovery_deadline_during_stale_devagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        heartbeat_attempts = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            FakeXpoolClient.heartbeat_attempts += 1
            if FakeXpoolClient.heartbeat_attempts in {1, 3}:
                raise instance_module.XpoolClientError("transport", "daemon unavailable")
            events.append(("heartbeat", instance_id, rank, heartbeat))
            return HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="stale_devagent",
                        cuda_device=0,
                        message="devagent is stale",
                    )
                ]
            )

    monotonic_values = iter(
        [
            0.0,
            1.0,
            10.0,
            instance_module.STALE_DEVAGENT_RECOVERY_GRACE_S + 5.0,
            instance_module.STALE_DEVAGENT_RECOVERY_GRACE_S + 5.0,
        ]
    )

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(instance_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert events == [
        ("heartbeat", "m", 0, ProcessHeartbeat(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_reregisters_missing_daemon_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)
    arena = transport_arena()

    class FakeXpoolClient:
        heartbeat_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", instance_id, rank, heartbeat))
            if FakeXpoolClient.heartbeat_count == 1:
                raise instance_module.XpoolDaemonError("not_ready", "registration missing")
            return HeartbeatResponse(warnings=[])

        def register_instance(self, registration: InstanceRegistration) -> None:
            events.append(("register", registration.model_dump(mode="json")))

        def acquire_instance_transport_arena(
            self,
            instance_id: str,
            *,
            rank: int,
            owner: ProcessRef,
        ) -> TransportArenaHandle:
            events.append(("acquire", instance_id, rank, owner))
            return arena

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)

    heartbeat = runtime_heartbeat(config, monkeypatch)
    heartbeat.arena_handle = arena
    heartbeat.step()

    assert isinstance(events[0], tuple)
    assert events[0][0] == "heartbeat"
    assert events[1] == (
        "register",
        {
            "pid": 123,
            "abi_version": ABI_VERSION,
            "instance_id": "m",
            "rank": 0,
            "transport": transport_attributes().model_dump(mode="json"),
        },
    )
    assert events[2] == (
        "acquire",
        "m",
        0,
        ProcessRef(abi_version=ABI_VERSION, pid=123),
    )
    assert isinstance(events[3], tuple)
    assert events[3][0] == "heartbeat"


def test_instance_heartbeat_fail_closes_if_recovered_arena_handle_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(
            self,
            instance_id: str,
            *,
            rank: int,
            heartbeat: object,
        ) -> HeartbeatResponse:
            raise instance_module.XpoolDaemonError("not_ready", "registration missing")

        def register_instance(self, registration: InstanceRegistration) -> None:
            return None

        def acquire_instance_transport_arena(
            self,
            instance_id: str,
            *,
            rank: int,
            owner: ProcessRef,
        ) -> TransportArenaHandle:
            return TransportArenaHandle(handle="11" * 64)

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)
    heartbeat.arena_handle = transport_arena()

    with pytest.raises(SystemExit) as exc_info:
        heartbeat.step()

    assert exc_info.value.code == 1


def test_instance_heartbeat_retries_arena_lease_after_registration_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    arena = transport_arena()
    events: list[str] = []

    class FakeXpoolClient:
        heartbeat_count = 0
        acquire_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(
            self,
            instance_id: str,
            *,
            rank: int,
            heartbeat: object,
        ) -> HeartbeatResponse:
            FakeXpoolClient.heartbeat_count += 1
            events.append("heartbeat")
            if FakeXpoolClient.heartbeat_count == 1:
                raise instance_module.XpoolDaemonError("not_ready", "registration missing")
            return HeartbeatResponse(warnings=[])

        def register_instance(self, registration: InstanceRegistration) -> None:
            events.append("register")

        def acquire_instance_transport_arena(
            self,
            instance_id: str,
            *,
            rank: int,
            owner: ProcessRef,
        ) -> TransportArenaHandle:
            FakeXpoolClient.acquire_count += 1
            events.append("acquire")
            if FakeXpoolClient.acquire_count == 1:
                raise instance_module.XpoolDaemonError("not_ready", "arena not republished")
            return arena

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    heartbeat = runtime_heartbeat(config, monkeypatch)
    heartbeat.arena_handle = arena

    heartbeat.step()
    assert heartbeat.arena_lease_recovery_pending
    heartbeat.step()

    assert not heartbeat.arena_lease_recovery_pending
    assert events == ["heartbeat", "register", "acquire", "heartbeat", "acquire"]


def test_instance_heartbeat_fail_closes_on_daemon_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(self, instance_id: str, *, rank: int, heartbeat: object) -> HeartbeatResponse:
            events.append(("heartbeat", rank, heartbeat))
            raise instance_module.XpoolDaemonError("conflict", "pid mismatch")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.code == 1
    assert len(events) == 1
    assert isinstance(events[0], tuple)
    assert events[0][0] == "heartbeat"


def test_instance_heartbeat_fail_closes_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def heartbeat_instance(
            self,
            instance_id: str,
            *,
            rank: int,
            heartbeat: object,
        ) -> HeartbeatResponse:
            raise ValueError("invalid heartbeat response")

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(procs_module.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.code == 1


def test_instance_start_heartbeat_replaces_dead_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    created: list[FakeThread] = []

    class FakeThread:
        def __init__(
            self,
            *,
            target: object,
            args: tuple[object, ...],
            name: str,
            daemon: bool,
        ) -> None:
            self.alive = False
            created.append(self)

        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

    monkeypatch.setattr(background_module.threading, "Thread", FakeThread)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRegistration) -> None:
            return None

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)

    instance = runtime_instance(config, monkeypatch)
    instance_module.get_instance().register_runtime(transport_attributes())
    instance_module.get_instance().start_heartbeat_worker()
    heartbeat_worker = instance.heartbeat_worker
    assert heartbeat_worker is not None
    worker = heartbeat_worker.worker
    instance_module.get_instance().start_heartbeat_worker()
    created[0].alive = False
    instance_module.get_instance().start_heartbeat_worker()

    assert instance.heartbeat_worker is heartbeat_worker
    assert heartbeat_worker.worker is worker
    assert len(created) == 2
