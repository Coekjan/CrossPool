from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from http import HTTPStatus

import pytest

import xpool.runtime.atnagent as atn_module
from xpool.abi import ABI_VERSION, TransportArenaHandle
from xpool.config import XpoolConfig, init_global_config
from xpool.runtime.agent import (
    AGENT_HEARTBEAT_INTERVAL_S,
    Agent,
    AgentHeartbeat,
)
from xpool.runtime.atnagent import (
    AtnAgent,
    AtnArenaResource,
)
from xpool.service.client import XpoolClientError
from xpool.service.wire import (
    HeartbeatResponse,
    InstanceRegistration,
    ProcessHeartbeat,
)


class SynchronousAgentHeartbeat:
    """Run production heartbeat classification synchronously in lifecycle tests."""

    def __init__(
        self,
        *,
        cuda_device: int,
        heartbeat: ProcessHeartbeat,
        sender: Callable[[int, ProcessHeartbeat], HeartbeatResponse],
        interval_s: float = AGENT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self.worker = AgentHeartbeat(
            cuda_device=cuda_device,
            heartbeat=heartbeat,
            sender=sender,
            interval_s=interval_s,
        )
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()

    def consume_registration_missing(self) -> bool:
        return self.worker.consume_registration_missing()

    def raise_if_failed(self) -> None:
        if not self.started:
            return
        self.worker.heartbeat_once()


@pytest.fixture
def reset_atnagent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_agent_runtime: None,
) -> Iterator[None]:
    """Install AtnAgent-specific native and heartbeat test doubles."""

    monkeypatch.setattr(atn_module, "AgentHeartbeat", SynchronousAgentHeartbeat)
    yield


def create_atnagent(config: XpoolConfig, *, cuda_device: int) -> Agent:
    """Install config and construct one production AtnAgent for tests."""

    init_global_config(config=config)
    return AtnAgent(cuda_device=cuda_device)


def health_client_class(healthy: bool) -> type:
    class FakeXpoolClient:
        def __init__(self) -> None:
            if not healthy:
                raise XpoolClientError("transport", "daemon health check failed")

        def close(self) -> None:
            return None

        def health(self) -> HTTPStatus:
            return HTTPStatus.OK

    return FakeXpoolClient


def wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def wait_until_raise(callback: Callable[[], None], *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        callback()
        time.sleep(0.01)
    callback()


def instance_registration_view(*, instance_id: str, rank: int) -> dict[str, object]:
    return {
        "pid": 1,
        "instance_id": instance_id,
        "rank": rank,
        "abi_version": ABI_VERSION,
        "transport": {
            "element_size": 4,
            "hidden_size": 4,
            "max_tokens": 8,
            "atn_tp_rank": 0,
            "atn_tp_size": 1,
            "atn_dp_rank": 0,
            "atn_dp_size": 1,
        },
    }


def transport_arena(rank: int) -> dict[str, object]:
    return {
        "handle": f"{rank:02x}" * 64,
    }


def transport_arena_resource(*, instance_id: str, rank: int, handle_rank: int) -> AtnArenaResource:
    registration = InstanceRegistration.model_validate(instance_registration_view(instance_id=instance_id, rank=rank))
    handle = TransportArenaHandle(handle=f"{handle_rank:02x}" * 64)
    return AtnArenaResource(instance_id=instance_id, registration=registration, handle=handle)


def patch_native_atnagent_ops(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[tuple[object, ...]] | None = None,
    create: Callable[..., object] | None = None,
    launch: Callable[..., object] | None = None,
    destroy: Callable[[TransportArenaHandle], object] | None = None,
) -> None:
    def fake_create(
        cuda_device: int,
        max_tokens: int,
        hidden_size: int,
        element_size: int,
        atn_dp_size: int,
    ) -> TransportArenaHandle:
        return TransportArenaHandle(handle=f"{cuda_device:02x}" * 64)

    def fake_start(handle: TransportArenaHandle) -> None:
        return None

    def fake_destroy(handle: TransportArenaHandle) -> None:
        if events is not None:
            events.append(("destroy", int(handle.handle[:2], 16)))

    monkeypatch.setattr(atn_module.xpool.ops.atnagent, "create_transport_arena", create or fake_create)
    monkeypatch.setattr(atn_module.xpool.ops.atnagent, "launch_transport_kernel", launch or fake_start)
    monkeypatch.setattr(atn_module.xpool.ops.atnagent, "destroy_transport_arena", destroy or fake_destroy)
