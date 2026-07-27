"""Torch dispatcher boundary for the sole Tensor data-path operation."""

from __future__ import annotations

import pytest
import torch


def test_dispatcher_registers_only_ffn_shim() -> None:
    names = {name for name in torch._C._dispatch_get_all_op_names() if name.startswith("xpool")}
    assert names == {"xpool::ffn_shim"}


def test_dispatcher_meta_fake_preserves_shape_dtype_and_device() -> None:
    hidden_states = torch.empty((2, 4), device="meta", dtype=torch.float32)
    output = torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 0)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device.type == "meta"


def test_dispatcher_has_no_cpu_kernel() -> None:
    hidden_states = torch.empty((2, 4), device="cpu", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="'CPU' backend"):
        torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 0)
