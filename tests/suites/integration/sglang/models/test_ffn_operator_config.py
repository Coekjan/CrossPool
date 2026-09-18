"""SGLang-owned FFN operator configuration behavior."""

from __future__ import annotations

import pytest
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

from xpool.runtime.ffnagent import operators


def test_sglang_moe_config_selection_restores_exact_function_after_failure() -> None:
    previous = fused_moe_triton_config.get_exec
    with pytest.raises(RuntimeError, match="selection failure"):
        with operators.sglang_moe_config_selection():
            assert fused_moe_triton_config.get_exec is not previous
            assert not fused_moe_triton_config.get_exec().deterministic.enable_deterministic_inference
            raise RuntimeError("selection failure")
    assert fused_moe_triton_config.get_exec is previous
