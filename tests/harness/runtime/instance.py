"""Provide explicitly imported fixtures for Instance runtime lifecycle tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest

import xpool.runtime.instance
from tests.harness.config import install_test_config
from xpool.abi import ABI_VERSION, TensorDType
from xpool.config import XpoolConfig
from xpool.fabric import FfnLayerKind, FfnLayerSpec, FfnWorkload
from xpool.runtime.instance import (
    Instance,
    InstanceHeartbeat,
)
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle


@dataclass(slots=True)
class ScriptedInstanceClient:
    """Response scripts and call observations for one Instance heartbeat test."""

    heartbeat_results: list[HeartbeatResponse | BaseException] = field(default_factory=list)
    arena_results: list[TransportArenaHandle | BaseException] = field(default_factory=list)
    calls: list[tuple[object, ...]] = field(default_factory=list)
    close_count: int = 0

    def close(self) -> None:
        self.close_count += 1
        self.calls.append(("close",))

    def heartbeat_instance(
        self,
        instance_id: str,
        *,
        rank: int,
        heartbeat: ProcessRef,
    ) -> HeartbeatResponse:
        self.calls.append(("heartbeat", instance_id, rank, heartbeat))
        return self.consume(self.heartbeat_results, "heartbeat")

    def register_instance(self, registration: InstanceRegistration) -> None:
        self.calls.append(("register", registration))

    def deregister_instance(self, instance_id: str, *, rank: int, owner: ProcessRef) -> None:
        self.calls.append(("deregister", instance_id, rank, owner))

    def acquire_instance_transport_arena(
        self,
        instance_id: str,
        *,
        rank: int,
        owner: ProcessRef,
    ) -> TransportArenaHandle:
        self.calls.append(("acquire", instance_id, rank, owner))
        return self.consume(self.arena_results, "arena acquisition")

    @staticmethod
    def consume[T](script: list[T | BaseException], operation: str) -> T:
        if not script:
            raise AssertionError(f"unexpected scripted Instance client {operation}")
        result = script.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def install_scripted_instance_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    heartbeat_results: list[HeartbeatResponse | BaseException],
    arena_results: list[TransportArenaHandle | BaseException] | None = None,
) -> ScriptedInstanceClient:
    """Install one isolated scripted client for a production Instance heartbeat."""

    client = ScriptedInstanceClient(
        heartbeat_results=list(heartbeat_results),
        arena_results=list(arena_results or ()),
    )
    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", lambda: client)
    return client


@pytest.fixture
def install_offline_instance_client(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    class OfflineXpoolClient:
        def close(self) -> None:
            pass

        def register_instance(self, registration: InstanceRegistration) -> None:
            pytest.fail("test must install a client fake before daemon registration")

        def deregister_instance(self, *args: object, **kwargs: object) -> None:
            pytest.fail("test must install a client fake before daemon deregistration")

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("test must install a client fake before arena acquisition")

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", OfflineXpoolClient)
    yield


def runtime_config(*, enabled: bool) -> XpoolConfig:
    env = {"XPOOL_DEBUG_LOOPBACK_ENABLE": "1", "XPOOL_DEBUG_LOOPBACK_SITE": "atnagent"} if enabled else {}
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env=env,
    )
    install_test_config(config)
    return config


def runtime_instance(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> Instance:
    monkeypatch.setattr(xpool.runtime.instance.os, "getpid", lambda: pid)
    install_test_config(config)
    return Instance(instance_id="m", rank=rank)


def runtime_heartbeat(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> InstanceHeartbeat:
    monkeypatch.setattr(xpool.runtime.instance.os, "getpid", lambda: pid)
    install_test_config(config)
    instance_id = "m"
    local_cuda_device = config.devices.atn_cuda_devices[rank]
    heartbeat = ProcessRef(abi_version=ABI_VERSION, pid=pid)
    return InstanceHeartbeat(
        instance_id=instance_id,
        rank=rank,
        local_cuda_device=local_cuda_device,
        registration=InstanceRegistration(
            instance_id=instance_id,
            rank=rank,
            abi_version=ABI_VERSION,
            pid=pid,
            transport=transport_attributes(),
            workload=workload(),
        ),
        heartbeat=heartbeat,
    )


def transport_arena() -> TransportArenaHandle:
    return TransportArenaHandle(handle="00" * 64)


def patch_native_instance_ops(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attach: Callable[..., object] | None = None,
    detach: Callable[..., object] | None = None,
    error_snapshot: Callable[..., object] | None = None,
) -> None:
    def attach_arena(instance_index: int, rank: int, handle: str) -> object:
        callback = attach or (lambda *args: None)
        return callback(instance_index, rank, TransportArenaHandle(handle=handle))

    monkeypatch.setattr(
        xpool.runtime.instance.xpool.native.transport,
        "attach_arena",
        attach_arena,
    )
    monkeypatch.setattr(
        xpool.runtime.instance.xpool.native.transport,
        "detach_arena",
        detach or (lambda *args: None),
    )
    monkeypatch.setattr(
        xpool.runtime.instance.xpool.native.transport,
        "read_generation_failure",
        error_snapshot or (lambda *args: int(xpool.runtime.instance.FfnResultCode.OK)),
    )


def transport_attributes() -> InstanceTransportAttributes:
    return InstanceTransportAttributes(
        hidden_size=4,
        max_tokens=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def workload() -> FfnWorkload:
    """Return the minimal valid workload used by instance runtime tests."""

    return FfnWorkload(
        model_config_digest="a" * 64,
        dtype=TensorDType.BF16,
        hidden_size=4,
        layers=(FfnLayerSpec(layer_id=0, kind=FfnLayerKind.DENSE),),
        max_decode_rows=1,
        max_prefill_rows=1,
    )
