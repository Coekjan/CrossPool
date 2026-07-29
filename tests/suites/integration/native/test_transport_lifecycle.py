"""Native Transport attachment, readiness, and owner lifecycle contracts."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from tests.harness.native.transport.owner import (
    controlled_atnagent_arena_process,
    transport_arena_handles,
)
from tests.harness.support.config import reset_global_config
from tests.harness.support.native.transport import initialize_instance_transport, instance_transport_runtime
from xpool.abi import FfnResultCode, TensorDType
from xpool.config import LoopbackSite
from xpool.runtime import RuntimeRole

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
    pytest.mark.usefixtures(reset_global_config.__name__),
]


@pytest.mark.usefixtures(instance_transport_runtime.__name__)
def test_transport_attachment_lifecycle(tmp_path: Path) -> None:
    with transport_arena_handles(workdir=tmp_path / "owners") as create_arena:
        first = create_arena(instance_index=1, instance_rank=0)
        xpool.native.transport.attach_arena(1, 0, first)
        assert FfnResultCode(xpool.native.transport.read_generation_failure()) is FfnResultCode.OK

        xpool.native.transport.attach_arena(1, 0, first)
        xpool.native.transport.detach_arena()
        xpool.native.transport.detach_arena()

        second = create_arena(instance_index=2, instance_rank=0)
        xpool.native.transport.attach_arena(2, 0, second)
        assert FfnResultCode(xpool.native.transport.read_generation_failure()) is FfnResultCode.OK


def isolated_transport_production_activation_requires_joined_fabric() -> None:
    xpool.native.initialize(
        RuntimeRole.ATNAGENT,
        cuda_device=torch.cuda.current_device(),
        debug_options=native_debug_options(),
    )
    arena = xpool.native.transport.create_arena(0, 0, 8, 4, int(TensorDType.FP32), 0, 1, 0, 1)
    try:
        with pytest.raises(RuntimeError, match="unavailable outside a joined generation"):
            xpool.native.transport.activate()
    finally:
        xpool.native.transport.destroy_arenas([arena])


def isolated_transport_attach_waits_for_resident_readiness(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
    with controlled_atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=torch.cuda.current_device(),
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=1,
        activate_resident=False,
    ) as controller:
        with ThreadPoolExecutor(max_workers=1) as executor:
            attachment = executor.submit(xpool.native.transport.attach_arena, 0, 0, controller.handle)
            time.sleep(0.1)
            assert not attachment.done()
            controller.activate()
            attachment.result(timeout=10)
        xpool.native.transport.detach_arena()


def isolated_transport_attach_rejects_closed_endpoint(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
    with controlled_atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=torch.cuda.current_device(),
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=1,
        activate_resident=True,
    ) as controller:
        controller.drain()
        with pytest.raises(RuntimeError, match="invalid state before first use"):
            xpool.native.transport.attach_arena(0, 0, controller.handle)


def isolated_transport_owner_lifecycle() -> None:
    xpool.native.initialize(
        RuntimeRole.ATNAGENT,
        torch.cuda.current_device(),
        native_debug_options(loopback_site=LoopbackSite.ATNAGENT),
    )
    handle = xpool.native.transport.create_arena(0, 0, 8, 4, int(TensorDType.FP32), 0, 1, 0, 1)
    with pytest.raises(RuntimeError, match="has not been activated"):
        xpool.native.transport.drain_async()
    with pytest.raises(RuntimeError, match="drain has not been started"):
        xpool.native.transport.drain_pending()
    xpool.native.transport.activate()
    drained = False
    destroyed = False
    try:
        with pytest.raises(RuntimeError, match="preactivation rollback or completed Resident drain"):
            xpool.native.transport.destroy_arenas([handle])
        with pytest.raises(RuntimeError, match="drain has not been started"):
            xpool.native.transport.drain_pending()
        xpool.native.transport.drain_async()
        xpool.native.transport.drain_async()
        wait_for_transport_drain()
        drained = True
        xpool.native.transport.drain_async()
        assert not xpool.native.transport.drain_pending()
        assert xpool.native.transport.read_trace(handle) is None
        xpool.native.transport.destroy_arenas([handle])
        destroyed = True
        with pytest.raises(RuntimeError, match="after arena destruction"):
            xpool.native.transport.drain_async()
        with pytest.raises(RuntimeError, match="after arena destruction"):
            xpool.native.transport.drain_pending()
        with pytest.raises(RuntimeError, match="unknown or already destroyed"):
            xpool.native.transport.read_trace(handle)
    finally:
        if not drained:
            xpool.native.transport.drain_async()
            wait_for_transport_drain()
        if not destroyed:
            xpool.native.transport.destroy_arenas([handle])


def wait_for_transport_drain() -> None:
    """Wait for the bounded native Transport drain used by lifecycle tests."""

    deadline = time.monotonic() + 5.0
    while xpool.native.transport.drain_pending():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out draining Transport Resident")
        time.sleep(0.001)


def test_transport_production_activation_requires_joined_fabric(tmp_path: Path) -> None:
    run_native_case(isolated_transport_production_activation_requires_joined_fabric, workdir=tmp_path / "case")


@pytest.mark.requires_mps
def test_transport_attach_waits_for_resident_readiness(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_attach_waits_for_resident_readiness,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )


@pytest.mark.requires_mps
def test_transport_attach_rejects_closed_endpoint(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_attach_rejects_closed_endpoint,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )


@pytest.mark.requires_mps
def test_transport_owner_drains_asynchronously_before_destroy(tmp_path: Path) -> None:
    run_native_case(isolated_transport_owner_lifecycle, workdir=tmp_path / "case")
