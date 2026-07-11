from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from http import HTTPStatus

import pytest

import xpool.runtime.devagent as devagent_module
import xpool.runtime.devagent.atn as atn_module
import xpool.runtime.devagent.common as common_module
from xpool.abi import ABI_VERSION, TransportArenaHandle
from xpool.config import XpoolConfig, init_global_config
from xpool.runtime.devagent import DevagentError
from xpool.runtime.devagent.atn import (
    AtnArenaResource,
)
from xpool.runtime.devagent.common import (
    DEVAGENT_HEARTBEAT_INTERVAL_S,
    Devagent,
)
from xpool.service.client import XpoolClientError, XpoolDaemonError
from xpool.service.wire import (
    InstanceRegistration,
    ProcessHeartbeat,
)


class FakeDevagentHeartbeat:
    def __init__(
        self,
        *,
        cuda_device: int,
        heartbeat: ProcessHeartbeat,
        interval_s: float = DEVAGENT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self.cuda_device = cuda_device
        self.heartbeat = heartbeat
        self.started = False
        self.registration_missing = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()

    def consume_registration_missing(self) -> bool:
        registration_missing = self.registration_missing
        self.registration_missing = False
        return registration_missing

    def raise_if_failed(self) -> None:
        if not self.started:
            return
        client = common_module.XpoolClient()
        try:
            client.heartbeat_devagent(self.cuda_device, self.heartbeat)
        except XpoolDaemonError as exc:
            if exc.kind == "not_ready":
                self.registration_missing = True
                return
            raise DevagentError(f"devagent heartbeat received unrecoverable daemon error: {exc}") from exc
        except XpoolClientError as exc:
            atn_module.logger.warning("devagent heartbeat failed: %s", exc)
        finally:
            client.close()


@pytest.fixture(autouse=True)
def reset_devagent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    class FakeHealthyXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(common_module.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(common_module.devkit, "install", lambda: None)
    monkeypatch.setattr(common_module, "XpoolClient", FakeHealthyXpoolClient)
    monkeypatch.setattr(atn_module, "DevagentHeartbeat", FakeDevagentHeartbeat)
    yield


def create_devagent(config: XpoolConfig, *, cuda_device: int) -> Devagent:
    """Install config and construct one production devagent for tests."""

    init_global_config(config=config)
    return devagent_module.create_devagent(cuda_device)


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


def patch_native_devagent_ops(
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

    monkeypatch.setattr(atn_module.xpool.ops.devagent, "create_transport_arena", create or fake_create)
    monkeypatch.setattr(atn_module.xpool.ops.devagent, "launch_transport_kernel", launch or fake_start)
    monkeypatch.setattr(atn_module.xpool.ops.devagent, "destroy_transport_arena", destroy or fake_destroy)
