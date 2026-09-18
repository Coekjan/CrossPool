"""DeepSeek V2 FFN router behavior."""

from __future__ import annotations

import pytest
import torch

from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.models.deepseek_v2 import DeepseekV2Adapter

pytestmark = pytest.mark.requires_cuda


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
def test_deepseek_router_matches_admitted_softmax_formula(payload_dtype: torch.dtype) -> None:
    hidden_states = torch.tensor(
        ((3.0, 1.0, -1.0, -3.0), (-2.0, 0.0, 2.0, 4.0)),
        device="cuda",
        dtype=payload_dtype,
    )
    router = weights.MoeRouterWeights(
        weight=torch.eye(4, device="cuda", dtype=payload_dtype),
        correction_bias=None,
    )
    workspace = torch.empty(
        DeepseekV2Adapter.router_workspace_bytes(
            payload_dtype=payload_dtype,
            payload_row_capacity=2,
            hidden_size=4,
            routed_expert_count=4,
            routed_topk=2,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    routed_ids = torch.empty((2, 2), device="cuda", dtype=torch.int32)
    routed_weights = torch.empty((2, 2), device="cuda", dtype=torch.float32)

    DeepseekV2Adapter.compute_routed_topk(
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=False,
    )

    scores = torch.softmax(hidden_states.float(), dim=-1)
    expected_weights, expected_ids = torch.topk(scores, 2, dim=-1, sorted=False)

    torch.testing.assert_close(routed_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(routed_weights, expected_weights.float(), rtol=1e-5, atol=1e-6)
