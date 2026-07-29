from __future__ import annotations

import pytest

import xpool.runtime.instance
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.runtime.instance import (
    install_offline_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_instance,
    transport_arena,
    transport_attributes,
    workload,
)
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig
from xpool.runtime.instance import Instance, InstanceError
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.wire import InstanceRegistration

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_offline_instance_client.__name__)


def test_instance_register_rejects_unknown_instance() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    install_test_config(config=config)

    with pytest.raises(InstanceError, match="unknown instance"):
        Instance.start(instance_id="missing", rank=0, transport=transport_attributes(), workload=workload())


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

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)
    instance = runtime_instance(config, monkeypatch)
    instance.register_runtime(transport_attributes(), workload())

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

    def fail_clear() -> None:
        raise RuntimeError("runtime is still busy")

    patch_native_instance_ops(
        monkeypatch,
        detach=fail_clear,
    )
    instance = runtime_instance(config, monkeypatch)
    instance.arena_handle = transport_arena()
    with pytest.raises(RuntimeError, match="runtime is still busy"):
        instance.deregister_runtime()


def test_instance_deregister_detaches_before_publishing_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(enabled=True)
    instance = runtime_instance(config, monkeypatch)
    instance.registration = InstanceRegistration(
        instance_id="m",
        rank=0,
        abi_version=ABI_VERSION,
        pid=instance.process_ref.pid,
        transport=transport_attributes(),
        workload=workload(),
    )
    events: list[str] = []
    monkeypatch.setattr(instance, "stop_failure_monitor", lambda: events.append("stop_monitor"))
    monkeypatch.setattr(instance, "detach_arena", lambda: events.append("detach"))
    monkeypatch.setattr(instance, "stop_heartbeat_worker", lambda: events.append("stop_heartbeat"))
    monkeypatch.setattr(
        instance.client,
        "deregister_instance",
        lambda instance_id, rank, owner: events.append("deregister"),
    )

    instance.deregister_runtime()

    assert events == ["stop_monitor", "detach", "stop_heartbeat", "deregister"]


def test_instance_start_does_not_cleanup_when_registration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config(enabled=True)
    events: list[str] = []

    def fail_register(self: Instance, transport: InstanceTransportAttributes, resolved: object) -> None:
        events.append("register")
        raise RuntimeError("register failed")

    monkeypatch.setattr(Instance, "register_runtime", fail_register)
    monkeypatch.setattr(Instance, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="register failed"):
        Instance.start(instance_id="m", rank=0, transport=transport_attributes(), workload=workload())

    assert events == ["register", "deregister"]


def test_instance_close_releases_runtime_before_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = runtime_instance(runtime_config(enabled=True), monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(instance, "deregister_runtime", lambda: events.append("deregister"))
    monkeypatch.setattr(instance.client, "close", lambda: events.append("close"))

    instance.close()

    assert events == ["deregister", "close"]
