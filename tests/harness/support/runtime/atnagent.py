"""Provide explicitly imported fixtures for AtnAgent runtime lifecycle tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

import xpool.runtime.agent
import xpool.runtime.atnagent
from tests.harness.support.config import install_test_config
from tests.harness.support.kv import kv_capacity_profile
from xpool.config import XpoolConfig
from xpool.native import ABI_VERSION
from xpool.runtime.agent import (
    Agent,
    AgentHeartbeat,
)
from xpool.runtime.atnagent import (
    AtnAgent,
    AtnAgentTransportArenaState,
)
from xpool.service.wire import HeartbeatResponse, InstanceRankRegistration
from xpool.transport import TransportArenaHandle


class SynchronousAgentHeartbeat:
    """Run production heartbeat classification synchronously in lifecycle tests."""

    def __init__(
        self,
        *,
        agent: Agent,
        interval_s: float = 5.0,
    ) -> None:
        self.worker = AgentHeartbeat(
            agent=agent,
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

    def consume_response(self) -> HeartbeatResponse | None:
        return self.worker.consume_response()

    def raise_if_failed(self) -> None:
        if not self.started:
            return
        self.worker.heartbeat_once()


@pytest.fixture
def reset_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    """Replace process-global Agent bootstrap dependencies for one test."""

    class HealthyClient:
        def close(self) -> None:
            pass

    monkeypatch.setattr(xpool.runtime.agent.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(xpool.runtime.agent.devkit, "install", lambda: None)
    monkeypatch.setattr(xpool.runtime.agent, "XpoolClient", HealthyClient)
    yield


@pytest.fixture
def reset_atnagent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_agent_runtime: None,
) -> Iterator[None]:
    """Install AtnAgent-specific native and heartbeat test doubles."""

    monkeypatch.setattr(xpool.runtime.atnagent, "AgentHeartbeat", SynchronousAgentHeartbeat)
    yield


def create_atnagent(config: XpoolConfig, *, cuda_device: int) -> AtnAgent:
    """Install config and construct one production AtnAgent for tests."""

    install_test_config(config)
    return AtnAgent(cuda_device=cuda_device)


def forming_heartbeat_response() -> HeartbeatResponse:
    """Return the daemon response before a Fabric generation is available."""

    return HeartbeatResponse(
        warnings=[],
        generation=None,
        fabric_phase=None,
    )


def instance_registration_view(*, instance_id: str, rank: int) -> dict[str, object]:
    return {
        "pid": 1,
        "instance_id": instance_id,
        "rank": rank,
        "abi_version": ABI_VERSION,
        "transport": {
            "hidden_size": 4,
            "payload_row_capacity": 8,
            "atn_tp_rank": 0,
            "atn_tp_size": 1,
            "atn_dp_rank": 0,
            "atn_dp_size": 1,
        },
        "ffn_profile": {
            "payload_dtype": "bfloat16",
            "hidden_size": 4,
            "layers": [{"layer_id": 0, "kind": 1}],
            "decode_payload_row_capacity": 1,
            "prefill_payload_row_capacity": 1,
            "group_sum_complete_admitted": False,
        },
        "kv_capacity": kv_capacity_profile().model_dump(mode="json"),
    }


def transport_arena(rank: int) -> dict[str, object]:
    return {
        "handle": f"{rank:02x}" * 64,
    }


def transport_entry(*, instance_id: str, rank: int, handle_rank: int) -> AtnAgentTransportArenaState:
    """Return one rank-local transport arena state."""

    registration = InstanceRankRegistration.model_validate(
        instance_registration_view(instance_id=instance_id, rank=rank)
    )
    handle = TransportArenaHandle(handle=f"{handle_rank:02x}" * 64)
    return AtnAgentTransportArenaState(instance_id=instance_id, registration=registration, handle=handle)


def patch_native_atnagent_ops(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[tuple[object, ...]] | None = None,
    create: Callable[..., object] | None = None,
    activate: Callable[[], object] | None = None,
    check_health: Callable[[], object] | None = None,
    drain_async: Callable[[], object] | None = None,
    drain_pending: Callable[[], bool] | None = None,
    destroy: Callable[[TransportArenaHandle], object] | None = None,
) -> None:
    def fake_create(
        instance_index: int,
        instance_rank: int,
        payload_row_capacity: int,
        hidden_size: int,
        dtype: int,
        atn_tp_rank: int,
        atn_tp_size: int,
        atn_dp_rank: int,
        atn_dp_size: int,
    ) -> str:
        return "00" * 64

    def fake_destroy(handle: TransportArenaHandle) -> None:
        if events is not None:
            events.append(("destroy", int(handle.handle[:2], 16)))

    def create_arena(*args: object) -> str:
        create_function: Callable[..., object] = create if create is not None else fake_create
        result = create_function(*args)
        if isinstance(result, TransportArenaHandle):
            return result.handle
        return str(result)

    def destroy_arenas(handles: list[str]) -> None:
        destroy_function = destroy or fake_destroy
        for handle in handles:
            destroy_function(TransportArenaHandle(handle=handle))

    monkeypatch.setattr(xpool.runtime.atnagent.xpool.native.transport, "create_arena", create_arena)
    monkeypatch.setattr(
        xpool.runtime.atnagent.xpool.native.transport,
        "activate",
        activate or (lambda: None),
    )
    monkeypatch.setattr(
        xpool.runtime.atnagent.xpool.native.transport,
        "check_health",
        check_health or (lambda: None),
    )
    monkeypatch.setattr(
        xpool.runtime.atnagent.xpool.native.transport,
        "drain_async",
        drain_async or (lambda: None),
    )
    monkeypatch.setattr(
        xpool.runtime.atnagent.xpool.native.transport,
        "drain_pending",
        drain_pending or (lambda: False),
    )
    monkeypatch.setattr(
        xpool.runtime.atnagent.xpool.native.transport,
        "destroy_arenas",
        destroy_arenas,
    )
