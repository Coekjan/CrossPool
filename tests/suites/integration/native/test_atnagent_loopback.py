"""AtnAgent-local hidden-pair loopback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.debug import native_debug_options
from tests.harness.native.transport.owner import atnagent_arena_process
from tests.harness.support.native.loopback import expected_loopback_rotation
from xpool.abi import TensorDType
from xpool.config import LoopbackSite
from xpool.runtime import RuntimeRole

pytestmark = [pytest.mark.requires_cuda(), pytest.mark.requires_mps, pytest.mark.timeout(180)]


def isolated_atnagent_loopback_rotates_hidden_pairs(dtype_name: str, workdir: str) -> None:
    xpool.native.initialize(
        RuntimeRole.INSTANCE,
        cuda_device=torch.cuda.current_device(),
        debug_options=native_debug_options(loopback_site=LoopbackSite.ATNAGENT),
    )
    torch_dtype = getattr(torch, dtype_name)
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=torch_dtype,
    )
    with atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype={
            torch.bfloat16: TensorDType.BF16,
            torch.float16: TensorDType.FP16,
            torch.float32: TensorDType.FP32,
        }[torch_dtype],
        atn_dp_size=1,
        activate_resident=True,
    ) as arena:
        xpool.native.transport.attach_arena(0, 0, arena)
        try:
            output = torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 0)
        finally:
            xpool.native.transport.detach_arena()

    torch.testing.assert_close(output, expected_loopback_rotation(hidden_states), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_atnagent_loopback_rotates_hidden_pairs(dtype_name: str, tmp_path: Path) -> None:
    run_native_case(
        isolated_atnagent_loopback_rotates_hidden_pairs,
        dtype_name,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )
