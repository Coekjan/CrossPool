from __future__ import annotations

import threading
from http import HTTPStatus

import pytest

import xpool.runtime.instance
import xpool.utils.background
import xpool.utils.procs
from tests.harness.runtime.instance import (
    install_offline_instance_client,
    install_scripted_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_heartbeat,
    runtime_instance,
    transport_arena,
    transport_attributes,
    workload,
)
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig
from xpool.runtime.instance import Instance
from xpool.service.wire import (
    ControlPlaneWarning,
    ControlPlaneWarningKind,
    HeartbeatResponse,
    InstanceRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle

pytestmark = pytest.mark.usefixtures(install_offline_instance_client.__name__)


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

    monkeypatch.setattr(xpool.utils.background.threading, "Thread", FakeThread)

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRegistration) -> None:
            return None

        def deregister_instance(self, instance_id: str, *, rank: int, owner: object) -> None:
            events.append(("deregister", instance_id, rank, worker.stop_event.is_set(), owner))

    patch_native_instance_ops(monkeypatch, detach=lambda: events.append(("detach", "m", 0)))
    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)

    instance = runtime_instance(config, monkeypatch)
    instance.register_runtime(transport_attributes(), workload())
    instance.arena_handle = transport_arena()
    instance.start_heartbeat_worker()
    heartbeat_worker = instance.heartbeat_worker
    assert heartbeat_worker is not None
    worker = heartbeat_worker.worker
    instance.deregister_runtime()

    assert events[0] == ("detach", "m", 0)
    assert events[1] == ("join", xpool.runtime.instance.INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S)
    assert isinstance(events[2], tuple)
    assert events[2][:4] == ("deregister", "m", 0, True)


def test_instance_start_registers_heartbeat_without_attaching_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []
    monkeypatch.setattr(Instance, "register_runtime", lambda self, transport, resolved: events.append("register"))
    monkeypatch.setattr(Instance, "start_heartbeat_worker", lambda self: events.append("heartbeat"))
    monkeypatch.setattr(
        Instance,
        "attach_arena_from_daemon",
        lambda self: pytest.fail("Transport attachment must wait for executable Fabric"),
    )

    instance = runtime_instance(config, monkeypatch)
    instance.start_runtime(transport_attributes(), workload())

    assert events == ["register", "heartbeat"]


def test_instance_start_deregisters_when_heartbeat_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []

    def fail_heartbeat(self: Instance) -> None:
        events.append("heartbeat")
        raise RuntimeError("heartbeat failed")

    monkeypatch.setattr(Instance, "register_runtime", lambda self, transport, resolved: events.append("register"))
    monkeypatch.setattr(Instance, "start_heartbeat_worker", fail_heartbeat)
    monkeypatch.setattr(Instance, "attach_arena_from_daemon", lambda self: events.append("attach"))
    monkeypatch.setattr(Instance, "start_failure_monitor", lambda self: events.append("monitor"))
    monkeypatch.setattr(Instance, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        instance = runtime_instance(config, monkeypatch)
        instance.start_runtime(transport_attributes(), workload())

    assert events == ["register", "heartbeat", "deregister"]


@pytest.mark.parametrize(
    ("warning_kind", "warning_cuda_device", "rank"),
    [
        ("stale_atnagent", 0, 1),
        ("quiescing_atnagent", 0, 0),
        ("stale_atnagent", 0, 0),
    ],
)
def test_instance_heartbeat_routes_nonfatal_atnagent_warning(
    monkeypatch: pytest.MonkeyPatch,
    warning_kind: ControlPlaneWarningKind,
    warning_cuda_device: int,
    rank: int,
) -> None:
    config = (
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
        if rank == 1
        else runtime_config(enabled=True)
    )
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind=warning_kind,
                        cuda_device=warning_cuda_device,
                        message=f"atnagent {warning_kind}",
                    )
                ]
            )
        ],
    )
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))

    runtime_heartbeat(config, monkeypatch, rank=rank).step()

    assert client.calls == [
        ("heartbeat", "m", rank, ProcessRef(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_fail_closes_after_local_stale_atnagent_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    stale_response = HeartbeatResponse(
        warnings=[
            ControlPlaneWarning(
                kind="stale_atnagent",
                cuda_device=0,
                message="atnagent is stale",
            )
        ]
    )
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[stale_response, stale_response],
    )

    monotonic_values = iter([0.0, xpool.runtime.instance.STALE_ATNAGENT_RECOVERY_GRACE_S + 1.0])

    monkeypatch.setattr(xpool.runtime.instance.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    with pytest.raises(SystemExit) as exc_info:
        heartbeat.step()

    assert exc_info.value.code == 1
    assert client.calls == [
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
    ]


@pytest.mark.parametrize("failure_kind", ["status", "transport"])
def test_instance_heartbeat_retries_recoverable_response_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    config = runtime_config(enabled=True)
    failure = (
        xpool.runtime.instance.XpoolClientError(
            "status",
            "bad gateway",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
        if failure_kind == "status"
        else xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable")
    )
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[failure, failure, failure],
    )
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert len(client.calls) == 3
    assert all(call[:3] == ("heartbeat", "m", 0) for call in client.calls)


def test_instance_heartbeat_keeps_client_on_recoverable_failure_and_closes_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable"),
            HeartbeatResponse(warnings=[]),
        ],
    )
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.stop()

    assert client.calls == [
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("close",),
    ]
    assert client.close_count == 1


def test_instance_heartbeat_fail_closes_after_transport_error_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable")],
    )

    monotonic_values = iter([0.0, xpool.runtime.instance.TRANSPORT_METADATA_RECOVERY_DEADLINE_S + 1.0])
    monkeypatch.setattr(xpool.runtime.instance.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.code == 1
    assert client.calls == [
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_resets_transport_recovery_deadline_during_stale_atnagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable"),
            HeartbeatResponse(
                warnings=[
                    ControlPlaneWarning(
                        kind="stale_atnagent",
                        cuda_device=0,
                        message="atnagent is stale",
                    )
                ]
            ),
            xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable"),
        ],
    )

    monotonic_values = iter(
        [
            0.0,
            1.0,
            10.0,
            xpool.runtime.instance.STALE_ATNAGENT_RECOVERY_GRACE_S + 5.0,
            xpool.runtime.instance.STALE_ATNAGENT_RECOVERY_GRACE_S + 5.0,
        ]
    )

    monkeypatch.setattr(xpool.runtime.instance.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: pytest.fail(f"unexpected os._exit({code})"))
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert client.calls == [
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
    ]


def test_instance_heartbeat_reregisters_missing_daemon_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    arena = transport_arena()
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            xpool.runtime.instance.XpoolDaemonError("not_ready", "registration missing"),
            HeartbeatResponse(warnings=[]),
        ],
        arena_results=[arena],
    )

    heartbeat = runtime_heartbeat(config, monkeypatch)
    heartbeat.arena_handle = arena
    heartbeat.step()

    assert client.calls[0] == ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123))
    assert client.calls[1] == ("register", heartbeat.registration)
    assert client.calls[2] == (
        "acquire",
        "m",
        0,
        ProcessRef(abi_version=ABI_VERSION, pid=123),
    )
    assert client.calls[3] == ("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123))


def test_instance_heartbeat_fail_closes_if_recovered_arena_handle_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[xpool.runtime.instance.XpoolDaemonError("not_ready", "registration missing")],
        arena_results=[TransportArenaHandle(handle="11" * 64)],
    )
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))
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
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            xpool.runtime.instance.XpoolDaemonError("not_ready", "registration missing"),
            HeartbeatResponse(warnings=[]),
        ],
        arena_results=[
            xpool.runtime.instance.XpoolDaemonError("not_ready", "arena not republished"),
            arena,
        ],
    )
    heartbeat = runtime_heartbeat(config, monkeypatch)
    heartbeat.arena_handle = arena

    heartbeat.step()
    assert heartbeat.arena_lease_recovery_pending
    heartbeat.step()

    assert not heartbeat.arena_lease_recovery_pending
    assert [call[0] for call in client.calls] == ["heartbeat", "register", "acquire", "heartbeat", "acquire"]


@pytest.mark.parametrize(
    "failure",
    [
        xpool.runtime.instance.XpoolDaemonError("conflict", "pid mismatch"),
        ValueError("invalid heartbeat response"),
    ],
)
def test_instance_heartbeat_fail_closes_on_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    config = runtime_config(enabled=True)
    client = install_scripted_instance_client(monkeypatch, heartbeat_results=[failure])
    monkeypatch.setattr(xpool.utils.procs.os, "_exit", lambda code: (item for item in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.code == 1
    assert client.calls == [("heartbeat", "m", 0, ProcessRef(abi_version=ABI_VERSION, pid=123))]
