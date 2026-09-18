"""GLM MoE Lite FFN router behavior."""

from __future__ import annotations

import pytest
import torch

from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.models.glm4_moe_lite import Glm4MoeLiteAdapter

pytestmark = pytest.mark.requires_cuda


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
@pytest.mark.parametrize("expert_count, topk", ((64, 4), (48, 3), (17, 1), (17, 17)))
def test_glm_router_matches_biased_sigmoid_formula(payload_dtype: torch.dtype, expert_count: int, topk: int) -> None:
    hidden_states = torch.arange(2 * expert_count, device="cuda", dtype=torch.float32).view(2, expert_count).to(
        payload_dtype
    ) / (2 * expert_count)
    correction_bias = torch.linspace(-0.25, 0.25, expert_count, device="cuda", dtype=torch.float32)
    router = weights.MoeRouterWeights(
        weight=torch.diag(torch.linspace(0.101, 1.909, expert_count, device="cuda", dtype=torch.float32)),
        correction_bias=correction_bias,
    )
    workspace = torch.empty(
        Glm4MoeLiteAdapter.router_workspace_bytes(
            payload_dtype=payload_dtype,
            payload_row_capacity=2,
            hidden_size=expert_count,
            routed_expert_count=expert_count,
            routed_topk=topk,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    routed_ids = torch.empty((2, topk), device="cuda", dtype=torch.int32)
    routed_weights = torch.empty((2, topk), device="cuda", dtype=torch.float32)

    Glm4MoeLiteAdapter.compute_routed_topk(
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=True,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        Glm4MoeLiteAdapter.compute_routed_topk(
            hidden_states=hidden_states,
            router_weights=router,
            workspace=workspace,
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            renormalize=True,
        )
    hidden_states.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()

    scores = torch.sigmoid(torch.mm(hidden_states.float(), router.weight.t()))
    converted_input = workspace.view(torch.float32)[: hidden_states.numel()].view_as(hidden_states)
    torch.testing.assert_close(converted_input, hidden_states.float(), rtol=0, atol=0)
    _, expected_ids = torch.topk(scores + correction_bias, topk, dim=-1)
    expected_weights = torch.gather(scores, 1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    order = routed_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    torch.testing.assert_close(
        routed_ids.gather(1, order), expected_ids.gather(1, expected_order).to(torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        routed_weights.gather(1, order), expected_weights.gather(1, expected_order), rtol=1e-5, atol=1e-6
    )


def test_glm_router_prefers_higher_expert_ids_on_ties_and_preserves_zero_weights() -> None:
    hidden_states = torch.tensor(((0.0,) * 17, (-1000.0,) * 17), device="cuda", dtype=torch.bfloat16)
    correction_bias = torch.zeros(17, device="cuda", dtype=torch.float32)
    router = weights.MoeRouterWeights(
        weight=torch.eye(17, device="cuda", dtype=torch.float32), correction_bias=correction_bias
    )
    workspace = torch.empty(
        Glm4MoeLiteAdapter.router_workspace_bytes(
            payload_dtype=hidden_states.dtype,
            payload_row_capacity=2,
            hidden_size=17,
            routed_expert_count=17,
            routed_topk=3,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    routed_ids = torch.empty((2, 3), device="cuda", dtype=torch.int32)
    routed_weights = torch.empty((2, 3), device="cuda", dtype=torch.float32)
    Glm4MoeLiteAdapter.compute_routed_topk(
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=True,
    )
    torch.testing.assert_close(
        routed_ids, torch.tensor(((16, 15, 14), (16, 15, 14)), device="cuda", dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        routed_weights,
        torch.tensor(((1 / 3, 1 / 3, 1 / 3), (0.0, 0.0, 0.0)), device="cuda", dtype=torch.float32),
        rtol=1e-5,
        atol=1e-6,
    )
