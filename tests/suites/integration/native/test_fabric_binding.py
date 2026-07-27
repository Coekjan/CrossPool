"""Native Fabric bootstrap and metadata binding contracts."""

from __future__ import annotations

import pytest
import torch

import xpool.native
from tests.harness.native.fabric import create_fabric_uid, fabric_bootstrap, run_fabric_topology
from tests.harness.native.process import run_native_case
from xpool.abi import TensorDType, XPoolForwardMode
from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.runtime import RuntimeRole


def test_fabric_uid_is_opaque_unique_hex() -> None:
    first = create_fabric_uid()
    second = create_fabric_uid()
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
    scheduler = xpool.native.FfnSchedulerPolicy.fifo()
    with pytest.raises(RuntimeError, match="requires at least one model"):
        xpool.native.fabric.join(xpool.native.FabricJoinMetadata(uid, 0, 1, 1, 1, scheduler, []))
    with pytest.raises(RuntimeError, match="requires at least one FFN layer"):
        xpool.native.fabric.join(
            xpool.native.FabricJoinMetadata(
                uid,
                0,
                1,
                1,
                1,
                scheduler,
                [xpool.native.FabricModelMetadata(4, 4, int(TensorDType.FP32), 8, 1, 1, [])],
            )
        )
    with pytest.raises(RuntimeError, match="FabricModelMetadata received an invalid dtype"):
        xpool.native.FabricModelMetadata(4, 4, 99, 8, 1, 1, [])
    with pytest.raises(RuntimeError, match="FabricLayerMetadata received an invalid kind"):
        xpool.native.FabricLayerMetadata(0, 99)
    with pytest.raises(TypeError):
        xpool.native.FabricLayerMetadata(-1, 1)
    with pytest.raises(TypeError):
        xpool.native.FabricModelMetadata(-1, 4, int(TensorDType.FP32), 8, 1, 1, [])
    with pytest.raises(TypeError):
        xpool.native.FabricJoinMetadata(uid, 0, 1, 1, -1, scheduler, [])


def isolated_fabric_role_guard() -> None:
    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="requires runtime role ffnagent"):
        xpool.native.fabric.activate()


@pytest.mark.requires_cuda()
def test_fabric_join_validates_metadata_before_collective_initialization() -> None:
    run_native_case(isolated_fabric_binding_validation)


@pytest.mark.requires_cuda()
def test_fabric_binding_rejects_wrong_runtime_role() -> None:
    run_native_case(isolated_fabric_role_guard)


@pytest.mark.requires_cuda(min_devices=2)
@pytest.mark.requires_mps
@pytest.mark.timeout(180)
def test_fabric_trace_binding_rejects_mismatched_event_families() -> None:
    """Read real records without allowing Python misuse to trigger fail-stop."""

    with fabric_bootstrap() as uid:
        report = run_fabric_topology(
            uid,
            atnagent_count=1,
            ffnagent_count=1,
            executor_count=1,
            forward_modes=(XPoolForwardMode.DECODE,),
            dtype=TensorDType.FP32,
            repetition_count=1,
        )
    assert report.participants
