"""Native Transport binding validation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.transport.owner import (
    atnagent_arena_process,
    transport_arena_handles,
)
from tests.harness.support.config import reset_global_config
from tests.harness.support.native.transport import instance_transport_runtime
from xpool.abi import FfnResultCode, TensorDType
from xpool.runtime import RuntimeRole

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.usefixtures(reset_global_config.__name__),
    pytest.mark.usefixtures(instance_transport_runtime.__name__),
]


def test_transport_attachment_rejects_identity_or_handle_mutation(tmp_path: Path) -> None:
    with transport_arena_handles(workdir=tmp_path / "owners") as create_arena:
        handle = create_arena(instance_index=1, instance_rank=0)
        xpool.native.transport.attach_arena(1, 0, handle)
        different = create_arena(instance_index=1, instance_rank=0)

        with pytest.raises(RuntimeError, match="different identity or handle"):
            xpool.native.transport.attach_arena(1, 0, different)
        with pytest.raises(RuntimeError, match="different identity or handle"):
            xpool.native.transport.attach_arena(2, 0, handle)

        assert FfnResultCode(xpool.native.transport.read_generation_failure()) is FfnResultCode.OK


def test_transport_initial_attachment_rejects_malformed_identity_or_handle(tmp_path: Path) -> None:
    with transport_arena_handles(workdir=tmp_path / "owners") as create_arena:
        handle = create_arena(instance_index=1, instance_rank=0)
        with pytest.raises(RuntimeError, match="identity does not match"):
            xpool.native.transport.attach_arena(2, 0, handle)
    with pytest.raises(RuntimeError, match="unexpected byte length"):
        xpool.native.transport.attach_arena(1, 0, "00" * 63)


def test_transport_unattached_operations_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="requires an attached arena"):
        xpool.native.transport.read_generation_failure()
    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="no attached instance transport arena"):
        torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 0)


def test_transport_arena_geometry_overflow_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="overflows result type"):
        with atnagent_arena_process(
            workdir=tmp_path / "owner",
            cuda_device=torch.cuda.current_device(),
            max_tokens=2**32 - 1,
            hidden_size=2**32 - 2,
            dtype=TensorDType.FP32,
            atn_dp_size=1,
            activate_resident=False,
        ):
            pass


def isolated_transport_binding_guards() -> None:
    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="requires runtime role atnagent"):
        xpool.native.transport.create_arena(0, 0, 8, 4, int(TensorDType.FP32), 0, 1, 0, 1)
    arguments = [0, 0, 8, 4, int(TensorDType.FP32), 0, 1, 0, 1]
    for argument_index in (0, 1, 2, 3, 5, 6, 7, 8):
        invalid = arguments.copy()
        invalid[argument_index] = -1
        with pytest.raises(TypeError):
            xpool.native.transport.create_arena(*invalid)
    for instance_index, rank in ((-1, 0), (0, -1)):
        with pytest.raises(TypeError):
            xpool.native.transport.attach_arena(instance_index, rank, "00" * 64)


def test_transport_binding_rejects_role_and_unsigned_conversion_mismatches(tmp_path: Path) -> None:
    run_native_case(isolated_transport_binding_guards, workdir=tmp_path / "case")
