"""FP32-logit Softmax Router behavior."""

from __future__ import annotations

import pytest
import torch
from tests.suites.integration.runtime.ffnagent.support import router_plan

import xtest
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, TopKWeightRounding
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.operators import routing


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
    plan = router_plan(
        hidden_states,
        router,
        algorithm=RouterAlgorithm.SOFTMAX,
        precision=RouterProjectionPrecision.FP32_LOGITS,
        routed_topk=2,
        renormalize=False,
        weight_rounding=TopKWeightRounding.NONE,
    )
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = routing.Router(plan, workspace)
    routed_ids, routed_weights = operation.ids, operation.weights

    operation.run_into(hidden_states=hidden_states, router_weights=router)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.run_into(hidden_states=hidden_states, router_weights=router)
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
