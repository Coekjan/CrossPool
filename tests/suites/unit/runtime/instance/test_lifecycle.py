from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import xpool.runtime.instance
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.runtime.instance import (
    ffn_profile,
    install_offline_instance_client,
    patch_native_instance_ops,
    runtime_config,
    runtime_instance,
    transport_arena,
    transport_attributes,
)
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId, FabricGenerationPhase, FabricPlan
from xpool.native import ABI_VERSION
from xpool.runtime.instance import InstanceRankError, InstanceRankRuntime
from xpool.runtime.transport import InstanceRankTransportProfile
from xpool.service.wire import InstanceRankRegistration, ReadinessSnapshot, ReadinessStatus

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, install_offline_instance_client.__name__)


def test_instance_register_rejects_unknown_instance() -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    install_test_config(config=config)

    with pytest.raises(InstanceRankError, match="unknown instance"):
        InstanceRankRuntime.start(
            instance_id="missing",
            rank=0,
            transport=transport_attributes(),
            ffn_profile=ffn_profile(),
        )


def test_instance_register_publishes_proc_uniq_id(monkeypatch: pytest.MonkeyPatch) -> None:
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
    instance.register_runtime(transport_attributes(), ffn_profile())

    assert registrations
    payload = registrations[0].model_dump(mode="json")
    assert payload["abi_version"] == ABI_VERSION
    assert isinstance(payload["pid"], int)
    assert payload["transport"] == transport_attributes().model_dump(mode="json")
    assert "create_time" not in payload


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
    instance.arena_handle = transport_arena()
    with pytest.raises(RuntimeError, match="runtime is still busy"):
        instance.deregister_runtime()


def test_instance_deregister_detaches_before_publishing_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config()
    instance = runtime_instance(config, monkeypatch)
    instance.registration = InstanceRankRegistration(
        instance_id="m",
        rank=0,
        abi_version=ABI_VERSION,
        pid=instance.process_ref.pid,
        transport=transport_attributes(),
        ffn_profile=ffn_profile(),
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
    runtime_config()
    events: list[str] = []

    def fail_register(self: InstanceRankRuntime, transport: InstanceRankTransportProfile, resolved: object) -> None:
        events.append("register")
        raise RuntimeError("register failed")

    monkeypatch.setattr(InstanceRankRuntime, "register_runtime", fail_register)
    monkeypatch.setattr(InstanceRankRuntime, "deregister_runtime", lambda self: events.append("deregister"))

    with pytest.raises(RuntimeError, match="register failed"):
        InstanceRankRuntime.start(
            instance_id="m",
            rank=0,
            transport=transport_attributes(),
            ffn_profile=ffn_profile(),
        )

    assert events == ["register", "deregister"]


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
            cuda_devices=(0, 1),
            atnagents=[],
            ffnagents=[],
            instances=[],
        )

    plan = cast(FabricPlan, SimpleNamespace(generation=generation))
    monkeypatch.setattr(instance.client, "readiness", readiness, raising=False)
    monkeypatch.setattr(instance.client, "fabric_plan", lambda: plan, raising=False)
    monkeypatch.setattr(xpool.runtime.instance.time, "sleep", lambda _: None)

    assert instance.wait_for_fabric_executable() is plan
