from __future__ import annotations

import pytest
import torch

from xpool.ffn import RouterPrecision
from xpool.runtime.ffnagent import execution


@pytest.mark.parametrize(
    "precision, expected_dtype, expected_bytes",
    (
        (RouterPrecision.PAYLOAD, torch.bfloat16, 16),
        (RouterPrecision.FP32_LOGITS, torch.bfloat16, 32),
        (RouterPrecision.FP32, torch.float32, 96),
    ),
)
def test_router_resources_follow_execution_semantics(
    precision: RouterPrecision, expected_dtype: torch.dtype, expected_bytes: int
) -> None:
    assert precision.weight_dtype(payload_dtype=torch.bfloat16) is expected_dtype
    assert (
        execution.router_workspace_bytes(
            precision,
            payload_dtype=torch.bfloat16,
            payload_row_capacity=2,
            hidden_size=8,
            routed_expert_count=4,
            routed_topk=2,
        )
        == expected_bytes
    )
