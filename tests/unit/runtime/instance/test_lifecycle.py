from __future__ import annotations

from tests.harness.runtime.instance import (
    Instance,
    InstanceRegistration,
    InstanceTransportAttributes,
    XpoolConfig,
    instance_module,
    patch_native_instance_ops,
    pytest,
    runtime_config,
    runtime_instance,
    transport_attributes,
)
from xpool.abi import ABI_VERSION
from xpool.config import init_global_config
from xpool.runtime.instance import InstanceError


def test_instance_constructs_unstarted_runtime_from_global_config() -> None:
    runtime_config(enabled=True)

    instance = Instance(instance_id="m", rank=0)

    assert instance.instance_id == "m"
    assert instance.rank == 0
    assert instance.registration is None


def test_instance_register_rejects_unknown_instance() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)

    with pytest.raises(InstanceError, match="unknown instance"):
        instance_module.init_instance(instance_id="missing", rank=0, transport=transport_attributes())


def test_instance_register_publishes_proc_uniq_id(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    registrations: list[InstanceRegistration] = []

    class FakeXpoolClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRegistration) -> None:
            registrations.append(registration)

    monkeypatch.setattr(instance_module, "XpoolClient", FakeXpoolClient)
    runtime_instance(config, monkeypatch)
    instance_module.get_instance().register_runtime(transport_attributes())

    assert registrations
    payload = registrations[0].model_dump(mode="json")
    assert payload["abi_version"] == ABI_VERSION
    assert isinstance(payload["pid"], int)
    assert payload["transport"] == transport_attributes().model_dump(mode="json")
    assert "create_time" not in payload


def test_instance_deregister_keeps_registration_when_runtime_clear_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)

    def fail_clear(config: XpoolConfig, instance_id: str, *, rank: int) -> None:
        raise RuntimeError("runtime is still busy")

    patch_native_instance_ops(
        monkeypatch,
        detach=lambda instance_index, rank: fail_clear(config, "m", rank=rank),
    )
    runtime_instance(config, monkeypatch)
    with pytest.raises(RuntimeError, match="runtime is still busy"):
        instance_module.get_instance().deregister_runtime()


def test_instance_start_does_not_cleanup_when_registration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    events: list[str] = []

    def fail_register(self: Instance, transport: InstanceTransportAttributes) -> None:
        events.append("register")
        raise RuntimeError("register failed")

    monkeypatch.setattr(Instance, "register_runtime", fail_register)
    monkeypatch.setattr(Instance, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="register failed"):
        runtime_instance(config, monkeypatch)
        instance_module.get_instance().start_runtime(transport_attributes())

    assert events == ["register"]


def test_instance_init_starts_and_retains_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)
    events: list[tuple[str, str, int, int]] = []

    def fake_start(self: Instance, transport: InstanceTransportAttributes) -> None:
        events.append(("start", self.instance_id, self.rank, transport.element_size))
        self.registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
        )

    monkeypatch.setattr(Instance, "start_runtime", fake_start)

    instance = instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())

    assert instance is instance_module.instance_runtime
    assert events == [("start", "m", 0, 4)]


def test_instance_init_reuses_matching_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)
    starts: list[object] = []

    def fake_start(self: Instance, transport: InstanceTransportAttributes) -> None:
        starts.append(transport)
        self.registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
        )

    monkeypatch.setattr(Instance, "start_runtime", fake_start)

    first = instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())
    second = instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())

    assert second is first
    assert starts == [transport_attributes()]


def test_instance_init_restarts_matching_inactive_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)
    starts: list[object] = []

    def fake_start(self: Instance, transport: InstanceTransportAttributes) -> None:
        starts.append(transport)
        self.registration = InstanceRegistration(
            instance_id=self.instance_id,
            rank=self.rank,
            abi_version=ABI_VERSION,
            pid=self.process_ref.pid,
            transport=transport,
        )

    monkeypatch.setattr(Instance, "start_runtime", fake_start)

    first = instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())
    first.registration = None
    second = instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())

    assert second is first
    assert starts == [transport_attributes(), transport_attributes()]


def test_instance_init_rejects_different_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)

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
    with pytest.raises(InstanceError, match="already owns m rank 0"):
        instance_module.init_instance(instance_id="m", rank=1, transport=transport_attributes())


def test_instance_init_does_not_install_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)

    def fail_start(self: Instance, transport: InstanceTransportAttributes) -> None:
        raise RuntimeError("start failed")

    monkeypatch.setattr(Instance, "start_runtime", fail_start)

    with pytest.raises(RuntimeError, match="start failed"):
        instance_module.init_instance(instance_id="m", rank=0, transport=transport_attributes())

    assert instance_module.instance_runtime is None


def test_del_instance_clears_singleton_after_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    instance = runtime_instance(config, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(instance, "deregister_runtime", lambda: events.append("deregister"))
    monkeypatch.setattr(instance, "close", lambda: events.append("close"))

    instance_module.del_instance()

    assert events == ["deregister", "close"]
    assert instance_module.instance_runtime is None


def test_del_instance_retains_singleton_when_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config(enabled=True)
    instance = runtime_instance(config, monkeypatch)

    def fail_cleanup() -> None:
        raise RuntimeError("native arena remains attached")

    monkeypatch.setattr(instance, "deregister_runtime", fail_cleanup)

    with pytest.raises(RuntimeError, match="native arena remains attached"):
        instance_module.del_instance()

    assert instance_module.instance_runtime is instance
