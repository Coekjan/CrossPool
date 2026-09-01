from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from tests.harness.native.fabric.bootstrap import create_fabric_uid
from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.native import RuntimeRole


def test_fabric_uid_is_opaque_unique_hex(tmp_path: Path) -> None:
    first = create_fabric_uid(workdir=tmp_path / "first")
    second = create_fabric_uid(workdir=tmp_path / "second")
    assert len(first.value) == FABRIC_UID_HEX_LENGTH
    assert set(first.value) <= set("0123456789abcdef")
    assert first != second


def isolated_fabric_binding_validation() -> None:
    xpool.native.initialize(RuntimeRole.ATNAGENT, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="requires a joined process runtime"):
        xpool.native.fabric.drain_async()
    with pytest.raises(RuntimeError, match="drain has not been started"):
        xpool.native.fabric.drain_pending()
    uid = "00" * 128
    scheduler = xpool.native.fabric.SchedulerPolicy.fifo()
    with pytest.raises(RuntimeError, match="requires at least one Instance"):
        xpool.native.fabric.ArenaProjection(1, 1, uid, 1, 1, 1, scheduler, ())
    with pytest.raises(RuntimeError, match="requires at least one FFN layer"):
        xpool.native.fabric.ArenaProjection(
            1,
            1,
            uid,
            1,
            1,
            1,
            scheduler,
            (xpool.native.fabric.InstanceProjection(torch.bfloat16, 8, 4, 4, True, 1, 1, (0,), ()),),
        )
    with pytest.raises(TypeError):
        xpool.native.fabric.InstanceProjection(99, 8, 4, 4, True, 1, 1, (0,), ())
    with pytest.raises(TypeError):
        xpool.native.fabric.InstanceLayerProjection(0, 99, 0, (0,))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        xpool.native.fabric.InstanceLayerProjection(-1, xpool.native.ffn.LayerKind.DENSE, 0, (0,))
    with pytest.raises(TypeError):
        xpool.native.fabric.InstanceProjection(torch.bfloat16, 8, -1, 4, True, 1, 1, (0,), ())
    with pytest.raises(TypeError):
        xpool.native.fabric.ArenaProjection(1, 1, uid, 1, 1, -1, scheduler, ())


def isolated_fabric_role_guard() -> None:
    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="requires runtime role ffnagent"):
        xpool.native.ffnagent.activate()


def isolated_native_allocation_sizing() -> None:
    xpool.native.initialize(RuntimeRole.DAEMON, None, native_debug_options())
    arena_bytes = xpool.native.fabric.arena_allocation_bytes(2, 3, 2, 2, 3, 3, 6, 300, 32)
    assert arena_bytes > 0
    assert xpool.native.fabric.arena_allocation_bytes(2, 3, 2, 2, 3, 3, 6, 300, 32) == arena_bytes
    assert xpool.native.fabric.ffnagent_control_allocation_bytes(is_coordinator=False, instance_count=2) == 4
    assert xpool.native.fabric.ffnagent_control_allocation_bytes(is_coordinator=True, instance_count=2) > 4
    assert xpool.native.ffnagent.execution_state_allocation_bytes(1, 4, 4, 2) > 0
    assert xpool.native.devkit.fabric_observer.allocation_bytes(2, 2) == 0
    assert xpool.native.devkit.ffn_routing_observer.allocation_bytes(8) == 0


def isolated_enabled_observer_sizing() -> None:
    xpool.native.initialize(
        RuntimeRole.DAEMON,
        None,
        native_debug_options(
            fabric_observer=True,
            ffn_routing_observer=True,
            record_capacity=2,
            routing_record_capacity=2,
        ),
    )
    fabric_bytes = xpool.native.devkit.fabric_observer.allocation_bytes(2, 2)
    assert fabric_bytes > 0
    assert xpool.native.devkit.fabric_observer.allocation_bytes(2, 2) == fabric_bytes
    with pytest.raises(RuntimeError, match="positive Instance and executor-Lane counts"):
        xpool.native.devkit.fabric_observer.allocation_bytes(0, 2)
    assert xpool.native.devkit.ffn_routing_observer.allocation_bytes(0) == 0
    assert xpool.native.devkit.ffn_routing_observer.allocation_bytes(8) > 0


@pytest.mark.requires_cuda()
def test_fabric_join_validates_metadata_before_collective_initialization(tmp_path: Path) -> None:
    run_native_case(isolated_fabric_binding_validation, workdir=tmp_path / "case")


@pytest.mark.requires_cuda()
def test_fabric_binding_rejects_wrong_runtime_role(tmp_path: Path) -> None:
    run_native_case(isolated_fabric_role_guard, workdir=tmp_path / "case")


def test_native_allocation_sizing_is_host_only_and_deterministic(tmp_path: Path) -> None:
    run_native_case(isolated_native_allocation_sizing, workdir=tmp_path / "case")


def test_observer_sizing_reads_host_debug_options(tmp_path: Path) -> None:
    run_native_case(isolated_enabled_observer_sizing, workdir=tmp_path / "case")
