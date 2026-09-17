"""Provide explicitly imported fixtures for Instance-rank runtime lifecycle tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest
import torch

import xpool.runtime.instance
from tests.harness.support.config import install_test_config
from xpool.config import XpoolConfig
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.native import ABI_VERSION
from xpool.native.ffn import LayerKind
from xpool.runtime.instance import (
    InstanceRankHeartbeat,
    InstanceRankRuntime,
)
from xpool.runtime.transport import InstanceRankTransportProfile
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceRankRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle


@dataclass(slots=True)
class ScriptedInstanceClient:
    """Response scripts and call observations for one Instance-rank heartbeat test."""

    heartbeat_results: list[HeartbeatResponse | BaseException] = field(default_factory=list)
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

    @staticmethod
    def consume[T](script: list[T | BaseException], operation: str) -> T:
        if not script:
            raise AssertionError(f"unexpected scripted Instance-rank client {operation}")
        result = script.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def install_scripted_instance_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    heartbeat_results: list[HeartbeatResponse | BaseException],
) -> ScriptedInstanceClient:
    """Install one isolated scripted client for an Instance-rank heartbeat."""

    client = ScriptedInstanceClient(heartbeat_results=list(heartbeat_results))
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

        def register_instance(self, registration: InstanceRankRegistration) -> None:
            pytest.fail("test must install a client fake before daemon registration")

        def deregister_instance(self, *args: object, **kwargs: object) -> None:
            pytest.fail("test must install a client fake before daemon deregistration")

        def acquire_instance_transport_arena(self, *args: object, **kwargs: object) -> TransportArenaHandle:
            pytest.fail("test must install a client fake before arena acquisition")

    monkeypatch.setattr(xpool.runtime.instance, "XpoolClient", OfflineXpoolClient)
    yield


def runtime_config() -> XpoolConfig:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
    )
    install_test_config(config)
    return config


def runtime_instance(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> InstanceRankRuntime:
    monkeypatch.setattr(xpool.runtime.instance.os, "getpid", lambda: pid)
    install_test_config(config)
    return InstanceRankRuntime(instance_id="m", rank=rank)


def runtime_heartbeat(
    config: XpoolConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    pid: int = 123,
) -> InstanceRankHeartbeat:
    monkeypatch.setattr(xpool.runtime.instance.os, "getpid", lambda: pid)
    install_test_config(config)
    instance_id = "m"
    heartbeat = ProcessRef(abi_version=ABI_VERSION, pid=pid)
    return InstanceRankHeartbeat(
        instance_id=instance_id,
        rank=rank,
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
        error_snapshot or (lambda *args: int(xpool.runtime.instance.ResultCode.OK)),
    )


def transport_attributes() -> InstanceRankTransportProfile:
    return InstanceRankTransportProfile(
        hidden_size=4,
        payload_row_capacity=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def ffn_profile() -> InstanceFfnProfile:
    """Return the minimal valid FFN Profile used by instance runtime tests."""

    return InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.bfloat16,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=1,
        group_sum_complete_admitted=False,
    )
