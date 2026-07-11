from __future__ import annotations

from tests.harness.runtime.instance import (
    Instance,
    InstanceRegistration,
    InstanceTransportAttributes,
    TransportArenaHandle,
    arena_for_rank,
    instance_module,
    native_attach_args,
    patch_native_instance_ops,
    pytest,
    runtime_config,
    runtime_instance,
    transport_arena,
    transport_attributes,
)
from xpool.abi import ABI_VERSION, FfnResultErrorCode
from xpool.runtime.instance import InstanceError, InstanceTransportMonitor
from xpool.service.wire import ProcessRef


def test_attach_transport_arena_registers_native_arenas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    config = runtime_config(enabled=True)
    patch_native_instance_ops(monkeypatch, attach=lambda *args: calls.append(args))
    handle = transport_arena()

    runtime_instance(config, monkeypatch)
    instance_module.get_instance().attach_arena(handle)

    assert calls == [native_attach_args(0, 0, handle)]


def test_transport_monitor_keeps_polling_healthy_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        instance_module.xpool.ops.instance,
        "transport_error_snapshot",
        lambda instance_index, rank: calls.append((instance_index, rank)) or FfnResultErrorCode.OK,
    )

    monitor = InstanceTransportMonitor(instance_id="m", instance_index=3, rank=2)

    assert monitor.step()
    assert calls == [(3, 2)]


def test_transport_monitor_fail_closes_on_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProcessTerminated(Exception):
        pass

    def terminate_process(logger: object, message: str, *args: object) -> None:
        raise ProcessTerminated(message % args)

    monkeypatch.setattr(
        instance_module.xpool.ops.instance,
        "transport_error_snapshot",
        lambda instance_index, rank: FfnResultErrorCode.NOT_IMPLEMENTED,
    )
    monkeypatch.setattr(
        instance_module,
        "bail",
        terminate_process,
    )

    monitor = InstanceTransportMonitor(instance_id="m", instance_index=3, rank=2)

    with pytest.raises(ProcessTerminated, match="NOT_IMPLEMENTED"):
        monitor.step()


def test_attach_transport_arena_does_not_require_transport_loopback_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    config = runtime_config(enabled=False)
    patch_native_instance_ops(monkeypatch, attach=lambda *args: calls.append(args))
    handle = transport_arena()

    runtime_instance(config, monkeypatch)
    instance_module.get_instance().attach_arena(handle)

    assert calls == [native_attach_args(0, 0, handle)]


def test_attach_transport_arena_rejects_unknown_instance_id() -> None:
    runtime_config(enabled=True)

    with pytest.raises(InstanceError, match="unknown instance id"):
        instance_module.init_instance(instance_id="missing", rank=0, transport=transport_attributes())


def test_detach_transport_arena_uses_config_instance_index(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    calls: list[tuple[object, ...]] = []
    patch_native_instance_ops(monkeypatch, detach=lambda *args: calls.append(args))

    runtime_instance(config, monkeypatch)
    instance_module.get_instance().detach_arena()

    assert calls == [(0, 0)]


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
            return arena_for_rank(rank=rank)

    def fake_install(instance_index: int, rank: int, arena: TransportArenaHandle) -> None:
        calls.append((str(instance_index), rank, instance_module.os.getpid()))

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    patch_native_instance_ops(monkeypatch, attach=fake_install)

    runtime_instance(config, monkeypatch)
    instance_module.get_instance().attach_arena_from_daemon()

    assert calls == [("m", 0, instance_module.os.getpid()), ("0", 0, instance_module.os.getpid())]


def test_instance_start_deregisters_when_transport_attach_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []

    def fail_attach(self: Instance) -> None:
        events.append("attach")
        raise RuntimeError("attach failed")

    monkeypatch.setattr(Instance, "register_runtime", lambda self, transport: events.append("register"))
    monkeypatch.setattr(Instance, "start_heartbeat_worker", lambda self: events.append("heartbeat"))
    monkeypatch.setattr(Instance, "attach_arena_from_daemon", fail_attach)
    monkeypatch.setattr(Instance, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="attach failed"):
        runtime_instance(config, monkeypatch)
        instance_module.get_instance().start_runtime(transport_attributes())

    assert events == ["register", "heartbeat", "attach", "deregister"]


def test_instance_init_rejects_different_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)
    changed_transport = transport_attributes().model_copy(update={"hidden_size": 8})

    def fake_start(self: Instance, transport: InstanceTransportAttributes) -> None:
        self.registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
        )

    monkeypatch.setattr(Instance, "start_runtime", fake_start)

    instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())
    with pytest.raises(InstanceError, match="different transport"):
        instance_module.init_instance(instance_id="m", rank=0, transport=changed_transport)
