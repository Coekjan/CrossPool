from __future__ import annotations

import logging
import threading
from http import HTTPStatus
from types import SimpleNamespace

import pytest

import xpool.runtime.instance
import xpool.utils.background
from xpool.fabric import FabricGenerationPhase
from xpool.model import ModelId
from xpool.native import ABI_VERSION
from xpool.runtime.instance import InstanceRankError, InstanceRankRuntime
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceRankRegistration,
    ProcessRef,
)
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config
from xtest.harness.support.kv import kv_capacity_profile
from xtest.harness.support.runtime.instance import (
    ffn_profile,
    install_offline_instance_client,
    install_scripted_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_heartbeat,
    runtime_instance,
    transport_arena,
    transport_attributes,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_offline_instance_client.__name__)


def test_instance_deregister_stops_heartbeat_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config()
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

        def register_instance(self, registration: InstanceRankRegistration) -> None:
            return None

        def deregister_instance(self, model_id: ModelId, *, rank: int, owner: object) -> None:
            events.append(("deregister", model_id, rank, worker.stop_event.is_set(), owner))

    patch_native_instance_ops(monkeypatch, detach=lambda: events.append(("detach", TEST_MODEL_ID, 0)))
    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)

    instance = runtime_instance(config, monkeypatch)
    instance.register_runtime(transport_attributes(), ffn_profile(), kv_capacity_profile(), 0)
    instance.arena_handle = transport_arena()
    instance.start_heartbeat_worker()
    heartbeat_worker = instance.heartbeat_worker
    assert heartbeat_worker is not None
    worker = heartbeat_worker.worker
    instance.deregister_runtime()

    assert events[0] == ("detach", TEST_MODEL_ID, 0)
    assert events[1] == ("join", xpool.runtime.instance.INSTANCE_HEARTBEAT_STOP_JOIN_TIMEOUT_S)
    assert isinstance(events[2], tuple)
    assert events[2][:4] == ("deregister", TEST_MODEL_ID, 0, True)


def test_instance_start_registers_heartbeat_without_attaching_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config()
    events: list[str] = []
    monkeypatch.setattr(InstanceRankRuntime, "start_heartbeat_worker", lambda self: events.append("heartbeat"))
    monkeypatch.setattr(
        InstanceRankRuntime,
        "attach_arena_from_daemon",
        lambda self: pytest.fail("Transport attachment must wait for executable Fabric"),
    )

    instance = runtime_instance(config, monkeypatch)
    monkeypatch.setattr(instance.client, "register_instance", lambda registration: events.append("register"))
    instance.start_runtime(transport_attributes(), ffn_profile(), kv_capacity_profile(), 0)
    instance.start_runtime(transport_attributes(), ffn_profile(), kv_capacity_profile(), 0)

    assert events == ["register", "heartbeat"]


def test_instance_start_deregisters_when_heartbeat_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config()
    events: list[str] = []

    def fail_heartbeat(self: InstanceRankRuntime) -> None:
        events.append("heartbeat")
        raise RuntimeError("heartbeat failed")

    monkeypatch.setattr(
        InstanceRankRuntime,
        "register_runtime",
        lambda self, transport, resolved, kv_capacity, headroom: events.append("register"),
    )
    monkeypatch.setattr(InstanceRankRuntime, "start_heartbeat_worker", fail_heartbeat)
    monkeypatch.setattr(InstanceRankRuntime, "attach_arena_from_daemon", lambda self: events.append("attach"))
    monkeypatch.setattr(InstanceRankRuntime, "start_failure_monitor", lambda self: events.append("monitor"))
    monkeypatch.setattr(InstanceRankRuntime, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        instance = runtime_instance(config, monkeypatch)
        instance.start_runtime(transport_attributes(), ffn_profile(), kv_capacity_profile(), 0)

    assert events == ["register", "heartbeat", "deregister"]


@pytest.mark.parametrize("failure_kind", ["status", "transport"])
def test_instance_heartbeat_retries_recoverable_response_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    config = runtime_config()
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
    heartbeat = runtime_heartbeat(config, monkeypatch)

    heartbeat.step()
    heartbeat.step()
    heartbeat.step()

    assert len(client.calls) == 3
    assert all(call[:3] == ("heartbeat", TEST_MODEL_ID, 0) for call in client.calls)


def test_instance_heartbeat_keeps_client_on_recoverable_failure_and_closes_on_stop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = runtime_config()
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[
            xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable"),
            HeartbeatResponse(warnings=[]),
        ],
    )
    heartbeat = runtime_heartbeat(config, monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="xpool.runtime.instance"):
        heartbeat.step()
        heartbeat.step()
    heartbeat.stop()

    assert client.calls == [
        ("heartbeat", TEST_MODEL_ID, 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("heartbeat", TEST_MODEL_ID, 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
        ("close",),
    ]
    assert client.close_count == 1
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.INFO]


def test_instance_heartbeat_reports_transport_error_after_recovery_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config()
    failure = xpool.runtime.instance.XpoolClientError("transport", "daemon unavailable")
    client = install_scripted_instance_client(
        monkeypatch,
        heartbeat_results=[failure],
    )

    monotonic_values = iter([0.0, xpool.runtime.instance.TRANSPORT_METADATA_RECOVERY_DEADLINE_S + 1.0])
    monkeypatch.setattr(xpool.runtime.instance, "time", SimpleNamespace(monotonic=lambda: next(monotonic_values)))

    with pytest.raises(InstanceRankError, match="heartbeat transport did not restore") as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value.__cause__ is failure
    assert client.calls == [
        ("heartbeat", TEST_MODEL_ID, 0, ProcessRef(abi_version=ABI_VERSION, pid=123)),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        xpool.runtime.instance.XpoolDaemonError("conflict", "pid mismatch"),
        xpool.runtime.instance.XpoolDaemonError("not_ready", "registration missing"),
        ValueError("invalid heartbeat response"),
    ],
)
def test_instance_heartbeat_preserves_unrecoverable_error(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    config = runtime_config()
    client = install_scripted_instance_client(monkeypatch, heartbeat_results=[failure])

    with pytest.raises(type(failure)) as exc_info:
        runtime_heartbeat(config, monkeypatch).step()

    assert exc_info.value is failure
    assert client.calls == [("heartbeat", TEST_MODEL_ID, 0, ProcessRef(abi_version=ABI_VERSION, pid=123))]


@pytest.mark.parametrize("phase", [FabricGenerationPhase.ABORTING, FabricGenerationPhase.STOPPED])
def test_terminal_heartbeat_notifies_lifecycle_without_main_thread_polling(
    monkeypatch: pytest.MonkeyPatch,
    phase: FabricGenerationPhase,
) -> None:
    install_scripted_instance_client(
        monkeypatch, heartbeat_results=[HeartbeatResponse(warnings=[], fabric_phase=phase)]
    )
    notified = threading.Event()
    failures: list[BaseException] = []

    def report(error: BaseException) -> None:
        failures.append(error)
        notified.set()

    heartbeat = runtime_heartbeat(runtime_config(), monkeypatch, on_failure=report)
    heartbeat.start()
    try:
        assert notified.wait(1.0)
    finally:
        heartbeat.worker.stop()
    with pytest.raises(InstanceRankError, match=phase.value) as error:
        heartbeat.worker.raise_if_failed()
    assert failures == [error.value]
