"""FP32 Sigmoid Router behavior."""

from __future__ import annotations

import pytest
import torch

import xtest
from xpool.ffn import RouterAlgorithm, RouterPrecision
from xpool.runtime.ffnagent import execution, routing, weights


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
@pytest.mark.parametrize(
    "expert_count, topk, correction_scale", ((64, 4, 4096.0), (48, 3, None), (17, 1, 1.0), (17, 17, 1.0))
)
@xtest.requirements(device_count=1)
def test_sigmoid_router_matches_selection_only_correction(
    payload_dtype: torch.dtype, expert_count: int, topk: int, correction_scale: float | None
) -> None:
    hidden_states = torch.arange(2 * expert_count, device="cuda", dtype=torch.float32).view(2, expert_count).to(
        payload_dtype
    ) / (2 * expert_count)
    # Larger correction exposes FP32 information loss from subtract-bias recovery.
    correction_bias = (
        torch.linspace(-0.25, 0.25, expert_count, device="cuda", dtype=torch.float32) * correction_scale
        if correction_scale is not None
        else None
    )
    router = weights.MoeRouterWeights(
        weight=torch.diag(torch.linspace(0.101, 1.909, expert_count, device="cuda", dtype=torch.float32)),
        correction_bias=correction_bias,
    )
    workspace = torch.empty(
        execution.router_workspace_bytes(
            RouterPrecision.FP32,
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

    routing.compute_routed_topk(
        RouterAlgorithm.SIGMOID,
        RouterPrecision.FP32,
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=True,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routing.compute_routed_topk(
            RouterAlgorithm.SIGMOID,
            RouterPrecision.FP32,
            hidden_states=hidden_states,
            router_weights=router,
            workspace=workspace,
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            renormalize=True,
        )
    hidden_states.mul_(0.5)
    if correction_bias is not None:
        correction_bias.neg_()
    graph.replay()
    torch.cuda.synchronize()

    scores = torch.sigmoid(torch.mm(hidden_states.float(), router.weight.t()))
    converted_input = workspace.view(torch.float32)[: hidden_states.numel()].view_as(hidden_states)
    torch.testing.assert_close(converted_input, hidden_states.float(), rtol=0, atol=0)
    selection = scores if correction_bias is None else scores + correction_bias
    _, expected_ids = torch.topk(selection, topk, dim=-1)
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


@pytest.mark.parametrize("algorithm", (RouterAlgorithm.SOFTMAX, RouterAlgorithm.SIGMOID))
@xtest.requirements(device_count=1)
def test_router_tie_policy_excludes_padding_and_preserves_zero_weights(algorithm: RouterAlgorithm) -> None:
    hidden_states = torch.tensor(((0.0,) * 17, (-1000.0,) * 17), device="cuda", dtype=torch.bfloat16)
    precision = RouterPrecision.FP32 if algorithm is RouterAlgorithm.SIGMOID else RouterPrecision.PAYLOAD
    router_dtype = precision.weight_dtype(payload_dtype=hidden_states.dtype)
    router = weights.MoeRouterWeights(weight=torch.eye(17, device="cuda", dtype=router_dtype), correction_bias=None)
    workspace = torch.empty(
        execution.router_workspace_bytes(
            precision,
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
    routing.compute_routed_topk(
        algorithm,
        precision,
        hidden_states=hidden_states,
        router_weights=router,
        workspace=workspace,
        routed_ids=routed_ids,
        routed_weights=routed_weights,
        renormalize=True,
    )
    ids = (16, 15, 14) if algorithm is RouterAlgorithm.SIGMOID else (0, 1, 2)
    second_weights = (0.0, 0.0, 0.0) if algorithm is RouterAlgorithm.SIGMOID else (1 / 3, 1 / 3, 1 / 3)
    torch.testing.assert_close(routed_ids, torch.tensor((ids, ids), device="cuda", dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        routed_weights,
        torch.tensor(((1 / 3, 1 / 3, 1 / 3), second_weights), device="cuda", dtype=torch.float32),
        rtol=1e-5,
        atol=1e-6,
    )
