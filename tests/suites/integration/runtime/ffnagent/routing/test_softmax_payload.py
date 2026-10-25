"""Payload-precision Softmax Router behavior."""

from __future__ import annotations

import pytest
import torch
import triton_kernels.topk

import xtest
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, TopKWeightRounding
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear, routing


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
@pytest.mark.parametrize("weight_rounding", tuple(TopKWeightRounding))
@xtest.requirements(device_count=1)
def test_softmax_payload_router_matches_softmax_formula(
    payload_dtype: torch.dtype, weight_rounding: TopKWeightRounding
) -> None:
    hidden_states = torch.tensor(
        ((3.0, 1.0, -1.0, -3.0), (-2.0, 0.0, 2.0, 4.0)),
        device="cuda",
        dtype=payload_dtype,
    )
    router = weights.MoeRouterWeights(
        projection=weights.ProjectionWeights(
            values=torch.eye(4, device="cuda", dtype=payload_dtype), scales=None, bias=None
        ),
        correction_bias=None,
    )
    workspace = torch.empty(
        execution.router_workspace_bytes(
            RouterProjectionPrecision.PAYLOAD,
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

    routing.compute_routed_topk(
        RouterAlgorithm.SOFTMAX,
        RouterProjectionPrecision.PAYLOAD,
        projection_operation=linear.LinearOperationKind.TORCH,
        weight_rounding=weight_rounding,
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=True,
    )

    scores = torch.softmax(hidden_states.float(), dim=-1)
    expected_weights, expected_ids = torch.topk(scores, 2, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    if weight_rounding is TopKWeightRounding.LOGITS_DTYPE:
        reference = triton_kernels.topk.topk(hidden_states, 2)
        expected_weights = reference.vals.float()
        expected_ids = reference.indx
    torch.testing.assert_close(routed_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(routed_weights, expected_weights.float(), rtol=1e-5, atol=1e-6)
