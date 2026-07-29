from __future__ import annotations

import pytest

import xpool.runtime.instance
from tests.harness.support.config import reset_global_config
from tests.harness.support.runtime.instance import (
    install_offline_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_instance,
    transport_arena,
    transport_attributes,
    workload,
)
from xpool.abi import ABI_VERSION, FfnResultCode
from xpool.runtime.instance import InstanceError, InstanceFailureMonitor
from xpool.service.wire import InstanceRegistration, ProcessRef
from xpool.transport import TransportArenaHandle

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_offline_instance_client.__name__)


def test_failure_monitor_keeps_polling_healthy_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        xpool.runtime.instance.xpool.native.transport,
        "read_generation_failure",
        lambda: calls.append(True) or int(FfnResultCode.OK),
    )

    monitor = InstanceFailureMonitor(instance_id="m", instance_index=3, rank=2)

    assert monitor.step()
    assert calls == [True]


def test_failure_monitor_fail_closes_on_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProcessTerminated(Exception):
        pass

    def terminate_process(logger: object, message: str, *args: object) -> None:
        raise ProcessTerminated(message % args)

    monkeypatch.setattr(
        xpool.runtime.instance.xpool.native.transport,
        "read_generation_failure",
        lambda: int(FfnResultCode.NOT_IMPLEMENTED),
    )
    monkeypatch.setattr(
        xpool.runtime.instance,
        "bail",
        terminate_process,
    )

    monitor = InstanceFailureMonitor(instance_id="m", instance_index=3, rank=2)

    with pytest.raises(ProcessTerminated, match="NOT_IMPLEMENTED"):
        monitor.step()


def test_detach_transport_arena_clears_process_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    calls: list[tuple[object, ...]] = []
    patch_native_instance_ops(monkeypatch, detach=lambda *args: calls.append(args))

    instance = runtime_instance(config, monkeypatch)
    instance.attach_arena(transport_arena())
    instance.detach_arena()

    assert calls == [()]


def test_instance_attach_from_daemon_uses_rank_local_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    calls: list[tuple[str, int, int]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def acquire_instance_transport_arena(
            self,
            instance_id: str,
            *,
            rank: int,
            owner: ProcessRef,
        ) -> TransportArenaHandle:
            calls.append((instance_id, rank, owner.pid))
            return transport_arena()

    def fake_install(instance_index: int, rank: int, arena: TransportArenaHandle) -> None:
        calls.append((str(instance_index), rank, xpool.runtime.instance.os.getpid()))

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)
    patch_native_instance_ops(monkeypatch, attach=fake_install)

    instance = runtime_instance(config, monkeypatch)
    instance.attach_arena_from_daemon()

    assert calls == [("m", 0, xpool.runtime.instance.os.getpid()), ("0", 0, xpool.runtime.instance.os.getpid())]


def test_started_instance_rejects_different_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    changed_transport = transport_attributes().model_copy(update={"hidden_size": 8})

    instance = runtime_instance(config, monkeypatch)
    instance.registration = InstanceRegistration(
        instance_id=instance.instance_id,
        rank=instance.rank,
        abi_version=ABI_VERSION,
        pid=instance.process_ref.pid,
        transport=transport_attributes(),
        workload=workload(),
    )
    with pytest.raises(InstanceError, match="different transport"):
        instance.start_runtime(changed_transport, workload())
