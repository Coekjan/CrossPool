from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.support.config import reset_global_config
from tests.harness.support.native.transport import instance_transport_runtime
from xpool.native import RuntimeRole

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.usefixtures(reset_global_config.__name__),
    pytest.mark.usefixtures(instance_transport_runtime.__name__),
]


def test_transport_initial_attachment_rejects_malformed_handle() -> None:
    with pytest.raises(RuntimeError, match="unexpected byte length"):
        xpool.native.transport.attach_arena(1, 0, "00" * 63)


def test_transport_unattached_operations_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="requires an attached arena"):
        xpool.native.transport.read_generation_failure()
    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    with pytest.raises(RuntimeError, match="no attached instance transport arena"):
        torch.ops.xpool.ffn_shim(hidden_states, None, output, 0, 2, 1, 0)


def isolated_transport_arena_geometry_overflow() -> None:
    xpool.native.initialize(RuntimeRole.ATNAGENT, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="overflows result type"):
        xpool.native.transport.create_arena(
            0,
            0,
            2**32 - 1,
            2**32 - 2,
            torch.bfloat16,
            0,
            1,
            0,
            1,
        )


def test_transport_arena_geometry_overflow_fails_closed(tmp_path: Path) -> None:
    run_native_case(isolated_transport_arena_geometry_overflow, workdir=tmp_path / "case")


def isolated_transport_binding_guards() -> None:
    xpool.native.initialize(RuntimeRole.INSTANCE, torch.cuda.current_device(), None)
    with pytest.raises(RuntimeError, match="requires runtime role atnagent"):
        xpool.native.transport.create_arena(0, 0, 8, 4, torch.bfloat16, 0, 1, 0, 1)
    arguments = [0, 0, 8, 4, torch.bfloat16, 0, 1, 0, 1]
    for argument_index in (0, 1, 2, 3, 5, 6, 7, 8):
        invalid = arguments.copy()
        invalid[argument_index] = -1
        with pytest.raises(TypeError):
            xpool.native.transport.create_arena(*invalid)  # ty: ignore[invalid-argument-type]
    for instance_index, rank in ((-1, 0), (0, -1)):
        with pytest.raises(TypeError):
            xpool.native.transport.attach_arena(instance_index, rank, "00" * 64)


def test_transport_binding_rejects_role_and_unsigned_conversion_mismatches(tmp_path: Path) -> None:
    run_native_case(isolated_transport_binding_guards, workdir=tmp_path / "case")
