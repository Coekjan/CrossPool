from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from xpool.abi import ABI_VERSION, TransportArenaHandle
from xpool.config import XpoolConfig, init_global_config
from xpool.runtime import instance as instance_module
from xpool.runtime.instance import (
    Instance,
    InstanceHeartbeat,
)
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.wire import (
    InstanceRegistration,
    ProcessHeartbeat,
)


@pytest.fixture(autouse=True)
def reset_instance_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    class OfflineXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_instance(self, registration: InstanceRegistration) -> None:
            pytest.fail("test must install a client fake before daemon registration")

        def deregister_instance(self, *args: object, **kwargs: object) -> None:
            pytest.fail("test must install a client fake before daemon deregistration")

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("test must install a client fake before arena acquisition")

    monkeypatch.setattr(instance_module, "instance_runtime", None)
    monkeypatch.setattr(instance_module, "XpoolClient", OfflineXpoolClient)
    yield
    monkeypatch.setattr(instance_module, "instance_runtime", None)


def runtime_config(*, enabled: bool) -> XpoolConfig:
    env = {"XPOOL_DEBUG_TRANSPORT_LOOPBACK_ENABLE": "1"} if enabled else {}
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env=env,
    )
    init_global_config(config=config)
    return config


def runtime_instance(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> Instance:
    monkeypatch.setattr(instance_module.os, "getpid", lambda: pid)
    init_global_config(config=config)
    instance = Instance(instance_id="m", rank=rank)
    monkeypatch.setattr(instance_module, "instance_runtime", instance)
    return instance


def runtime_heartbeat(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> InstanceHeartbeat:
    monkeypatch.setattr(instance_module.os, "getpid", lambda: pid)
    init_global_config(config=config)
    instance_id = "m"
    local_cuda_device = config.devices.atn_cuda_devices[rank]
    heartbeat = ProcessHeartbeat(abi_version=ABI_VERSION, pid=pid)
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
        ),
        heartbeat=heartbeat,
    )


def arena_for_rank(*, rank: int) -> TransportArenaHandle:
    return transport_arena()


def transport_arena() -> TransportArenaHandle:
    return TransportArenaHandle(handle="00" * 64)


def native_attach_args(instance_index: int, rank: int, handle: TransportArenaHandle) -> tuple[object, ...]:
    return (instance_index, rank, handle)


def patch_native_instance_ops(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attach: Callable[..., object] | None = None,
    detach: Callable[..., object] | None = None,
    error_snapshot: Callable[..., object] | None = None,
) -> None:
    monkeypatch.setattr(instance_module.xpool.ops.instance, "attach_transport_arena", attach or (lambda *args: None))
    monkeypatch.setattr(instance_module.xpool.ops.instance, "detach_transport_arena", detach or (lambda *args: None))
    monkeypatch.setattr(
        instance_module.xpool.ops.instance,
        "transport_error_snapshot",
        error_snapshot or (lambda *args: instance_module.FfnResultErrorCode.OK),
    )


def transport_attributes() -> InstanceTransportAttributes:
    return InstanceTransportAttributes(
        element_size=4,
        hidden_size=4,
        max_tokens=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )
