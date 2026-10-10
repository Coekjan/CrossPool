from __future__ import annotations

from types import SimpleNamespace

import pytest

import xpool.runtime.instance
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId, FabricGenerationPhase
from xpool.model import ModelId
from xpool.native import ABI_VERSION
from xpool.runtime.instance import InstanceRankError, InstanceRankRuntime
from xpool.runtime.transport import InstanceRankTransportProfile
from xpool.service.wire import InstanceRankRegistration, ReadinessSnapshot, ReadinessStatus
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config
from xtest.harness.support.kv import kv_capacity_profile
from xtest.harness.support.runtime.instance import (
    ffn_profile,
    install_offline_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_instance,
    transport_arena,
    transport_attributes,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_offline_instance_client.__name__)


def test_instance_register_rejects_unknown_instance() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    install_test_config(config=config)

    with pytest.raises(InstanceRankError, match="unknown Model ID"):
        InstanceRankRuntime.start(
            model_id=ModelId("test/missing"),
            rank=0,
            transport=transport_attributes(),
            ffn_profile=ffn_profile(),
            kv_capacity=kv_capacity_profile(),
            atn_runtime_headroom_bytes=0,
            on_failure=lambda error: None,
        )


def test_instance_register_publishes_runtime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    config = runtime_config()
    registrations: list[InstanceRankRegistration] = []

    class FakeXpoolClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRankRegistration) -> None:
            registrations.append(registration)

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)
    instance = runtime_instance(config, monkeypatch)
    instance.register_runtime(transport_attributes(), ffn_profile(), kv_capacity_profile(), 1024)

    assert registrations
    payload = registrations[0].model_dump(mode="json")
    assert payload["abi_version"] == ABI_VERSION
    assert isinstance(payload["pid"], int)
    assert payload["transport"] == transport_attributes().model_dump(mode="json")
    assert payload["atn_runtime_headroom_bytes"] == 1024


def test_instance_deregister_keeps_registration_when_runtime_clear_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config()

    def fail_clear() -> None:
        raise RuntimeError("runtime is still busy")

    patch_native_instance_ops(
        monkeypatch,
        detach=fail_clear,
    )
    instance = runtime_instance(config, monkeypatch)
    registration = InstanceRankRegistration(
        model_id=TEST_MODEL_ID,
        rank=0,
        abi_version=ABI_VERSION,
        pid=instance.process_ref.pid,
        transport=transport_attributes(),
        ffn_profile=ffn_profile(),
        kv_capacity=kv_capacity_profile(),
        atn_runtime_headroom_bytes=0,
    )
    arena_handle = transport_arena()
    instance.registration = registration
    instance.arena_handle = arena_handle
    with pytest.raises(RuntimeError, match="runtime is still busy"):
        instance.deregister_runtime()
    assert instance.registration is registration
    assert instance.arena_handle is arena_handle


def test_instance_deregister_detaches_before_publishing_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config()
    instance = runtime_instance(config, monkeypatch)
    instance.registration = InstanceRankRegistration(
        model_id=TEST_MODEL_ID,
        rank=0,
        abi_version=ABI_VERSION,
        pid=instance.process_ref.pid,
        transport=transport_attributes(),
        ffn_profile=ffn_profile(),
        kv_capacity=kv_capacity_profile(),
        atn_runtime_headroom_bytes=0,
    )
    events: list[str] = []
    monkeypatch.setattr(instance, "stop_failure_monitor", lambda: events.append("stop_monitor"))
    monkeypatch.setattr(instance, "detach_arena", lambda: events.append("detach"))
    monkeypatch.setattr(instance, "stop_heartbeat_worker", lambda: events.append("stop_heartbeat"))
    monkeypatch.setattr(
        instance.client,
        "deregister_instance",
        lambda model_id, rank, owner: events.append("deregister"),
    )

    instance.deregister_runtime()

    assert events == ["stop_monitor", "detach", "stop_heartbeat", "deregister"]


def test_instance_start_closes_local_client_without_remote_deregistration_on_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config()
    events: list[str] = []
    failure = RuntimeError("register failed")
    reported: list[BaseException] = []

    class FakeXpoolClient:
        def close(self) -> None:
            events.append("close")

        def deregister_instance(self, *args: object, **kwargs: object) -> None:
            events.append("remote_deregister")

    def fail_register(
        self: InstanceRankRuntime,
        transport: InstanceRankTransportProfile,
        resolved: object,
        kv_capacity: object,
        atn_runtime_headroom_bytes: int,
    ) -> None:
        events.append("register")
        raise failure

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(InstanceRankRuntime, "register_runtime", fail_register)

    with pytest.raises(RuntimeError, match="register failed") as error:
        InstanceRankRuntime.start(
            model_id=TEST_MODEL_ID,
            rank=0,
            transport=transport_attributes(),
            ffn_profile=ffn_profile(),
            kv_capacity=kv_capacity_profile(),
            atn_runtime_headroom_bytes=0,
            on_failure=reported.append,
        )

    assert error.value is failure
    assert reported == [failure]
    assert events == ["register", "close"]


def test_instance_close_releases_runtime_before_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = runtime_instance(runtime_config(), monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(instance, "deregister_runtime", lambda: events.append("deregister"))
    monkeypatch.setattr(instance.client, "close", lambda: events.append("close"))

    instance.close()

    assert events == ["deregister", "close"]


def test_instance_waits_through_every_fabric_startup_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = runtime_instance(runtime_config(), monkeypatch)
    generation = FabricGenerationId.create()
    phases = iter(
        (
            FabricGenerationPhase.PREPARING_JOIN,
            FabricGenerationPhase.JOINING,
            FabricGenerationPhase.PREPARING_EXECUTION,
            FabricGenerationPhase.ACTIVATING,
            FabricGenerationPhase.EXECUTABLE,
        )
    )

    def readiness() -> ReadinessSnapshot:
        return ReadinessSnapshot(
            ready=False,
            generation=generation,
            fabric_phase=next(phases),
            fabric_invocation_failure=None,
            fabric_owner_failure=None,
            fabric_control_failure=None,
            transport_ready=False,
            instances_initialized=False,
            mps_status=ReadinessStatus.ONLINE,
            devices=(0, 1),
            atnagents=[],
            ffnagents=[],
            instances=[],
        )

    plan = SimpleNamespace(generation=generation)
    monkeypatch.setattr(instance.client, "readiness", readiness, raising=False)
    monkeypatch.setattr(instance.client, "fabric_plan", lambda: plan, raising=False)
    monkeypatch.setattr(xpool.runtime.instance.time, "sleep", lambda _: None)

    assert instance.wait_for_fabric_executable() is plan
