"""FP32-logit Softmax Router behavior."""

from __future__ import annotations

import pytest
import torch

import xtest
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, TopKWeightRounding
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear, routing


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
@pytest.mark.parametrize("correction_present", (False, True), ids=("uncorrected", "corrected"))
@xtest.requirements(device_count=1)
def test_softmax_fp32_router_matches_selection_only_correction(
    payload_dtype: torch.dtype, correction_present: bool
) -> None:
    hidden_states = torch.tensor(
        ((3.0, 1.0, -1.0, -3.0), (-2.0, 0.0, 2.0, 4.0)),
        device="cuda",
        dtype=payload_dtype,
    )
    correction_bias = (
        torch.tensor((-0.75, -0.25, 0.5, 0.8), device="cuda", dtype=torch.float32) if correction_present else None
    )
    router = weights.MoeRouterWeights(
        projection=weights.ProjectionWeights(
            values=torch.eye(4, device="cuda", dtype=payload_dtype), scales=None, bias=None
        ),
        correction_bias=correction_bias,
    )
    workspace = torch.empty(
        execution.router_workspace_bytes(
            RouterProjectionPrecision.FP32_LOGITS,
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
        RouterProjectionPrecision.FP32_LOGITS,
        projection_operation=linear.LinearOperationKind.TORCH,
        weight_rounding=TopKWeightRounding.NONE,
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=False,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routing.compute_routed_topk(
            RouterAlgorithm.SOFTMAX,
            RouterProjectionPrecision.FP32_LOGITS,
            projection_operation=linear.LinearOperationKind.TORCH,
            weight_rounding=TopKWeightRounding.NONE,
            hidden_states=hidden_states,
            router_weights=router,
            workspace=workspace,
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            renormalize=False,
        )
    hidden_states.mul_(0.5)
    if correction_bias is not None:
        correction_bias.neg_()
    graph.replay()
    torch.cuda.synchronize()

    scores = torch.softmax(hidden_states.float(), dim=-1)
    selection = scores if correction_bias is None else scores + correction_bias
    expected_ids = torch.topk(selection, 2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    if correction_bias is not None:
        assert not torch.equal(expected_ids, torch.topk(scores, 2, dim=-1).indices)

    torch.testing.assert_close(routed_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(routed_weights, expected_weights.float(), rtol=1e-5, atol=1e-6)
