"""CUDA behavior tests for fixed-Capacity Dense FFN components."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional

from xpool.ffn import ActivationKind
from xpool.runtime.ffnagent import execution, operators, weights

pytestmark = pytest.mark.requires_cuda

ROW_CAPACITY = 4
HIDDEN_SIZE = 16
LOCAL_INTERMEDIATE_SIZE = 8


def dense_values(
    payload_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, weights.DenseFfnWeights, torch.Tensor, torch.Tensor]:
    """Allocate one small valid Dense component input and caller-owned outputs."""

    hidden_states = torch.randn((ROW_CAPACITY, HIDDEN_SIZE), device="cuda", dtype=payload_dtype)
    layer_weights = weights.DenseFfnWeights(
        gate_up_weight=torch.randn(
            (2 * LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            device="cuda",
            dtype=payload_dtype,
        ),
        down_weight=torch.randn(
            (HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE),
            device="cuda",
            dtype=payload_dtype,
        ),
    )
    workspace = torch.empty(
        execution.dense_workspace_bytes(
            payload_dtype=payload_dtype,
            row_capacity=ROW_CAPACITY,
            local_intermediate_size=LOCAL_INTERMEDIATE_SIZE,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    output = torch.empty_like(hidden_states)
    return hidden_states, layer_weights, workspace, output


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
def test_compute_dense_partial_matches_reference_without_steady_allocation(payload_dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    hidden_states, layer_weights, workspace, output = dense_values(payload_dtype)

    returned = operators.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=ActivationKind.SILU,
    )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    operators.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=ActivationKind.SILU,
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operators.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=workspace,
            output=output,
            activation=ActivationKind.SILU,
        )
    graph.replay()
    torch.cuda.synchronize()

    gate_up = torch.mm(hidden_states, layer_weights.gate_up_weight.t())
    gate, up = gate_up.chunk(2, dim=-1)
    activated_up = (torch.nn.functional.silu(gate.float()) * up.float()).to(payload_dtype)
    expected = torch.mm(activated_up, layer_weights.down_weight.t())
    assert returned is output
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)


def test_compute_dense_partial_keeps_live_prefix_independent_of_capacity_tail() -> None:
    torch.manual_seed(23)
    hidden_states, layer_weights, workspace, output = dense_values()
    operators.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=ActivationKind.SILU,
    )
    first_prefix = output[:2].clone()
    hidden_states[2:] = torch.randn_like(hidden_states[2:])
    operators.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=ActivationKind.SILU,
    )

    torch.testing.assert_close(output[:2], first_prefix, rtol=0, atol=0)


def test_compute_dense_partial_rejects_misaligned_workspace() -> None:
    hidden_states, layer_weights, workspace, output = dense_values()
    storage = torch.empty(workspace.numel() + 1, device="cuda", dtype=torch.uint8)
    misaligned_workspace = storage[1:]

    with pytest.raises(ValueError, match="16-byte aligned"):
        operators.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=misaligned_workspace,
            output=output,
            activation=ActivationKind.SILU,
        )


def test_finalize_moe_routing_replays_one_graph_for_dynamic_live_rows() -> None:
    routed_ids = torch.tensor(
        ((0, 2), (1, 3), (2, 0), (3, 1)),
        device="cuda",
        dtype=torch.int32,
    )
    routed_weights = torch.tensor(
        ((0.75, 0.25), (0.6, 0.4), (0.9, 0.1), (0.55, 0.45)),
        device="cuda",
        dtype=torch.float32,
    )
    final_ids = torch.empty((4, 3), device="cuda", dtype=torch.int32)
    final_weights = torch.empty((4, 3), device="cuda", dtype=torch.float32)
    payload_rows = torch.tensor((2,), device="cuda", dtype=torch.int64)

    def finalize() -> None:
        operators.finalize_moe_routing(
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            final_ids=final_ids,
            final_weights=final_weights,
            payload_rows=payload_rows,
            routed_expert_count=4,
            shared_expert_count=1,
            routed_scaling_factor=2.0,
        )

    finalize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        finalize()

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        final_ids,
        torch.tensor(((0, 2, 4), (1, 3, 4), (-1, -1, -1), (-1, -1, -1)), device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        final_weights,
        torch.tensor(
            ((0.75, 0.25, 0.5), (0.6, 0.4, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            device="cuda",
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )

    payload_rows.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(final_ids[2], torch.tensor((2, 0, 4), device="cuda", dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        final_weights[2],
        torch.tensor((0.9, 0.1, 0.5), device="cuda", dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(final_ids[3], torch.full((3,), -1, device="cuda", dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[3], torch.zeros(3, device="cuda"), rtol=0, atol=0)

    routed_weights.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(final_weights[:3, :2], torch.zeros((3, 2), device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[:3, 2], torch.full((3,), 0.5, device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[3], torch.zeros(3, device="cuda"), rtol=0, atol=0)


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
def test_compute_moe_partial_matches_reference_with_overlapped_workspace_and_graph(
    payload_dtype: torch.dtype,
) -> None:
    torch.manual_seed(31)
    row_capacity = 4
    hidden_size = 16
    local_intermediate_size = 8
    expert_count = 4
    effective_topk = 2
    route_count = row_capacity * effective_topk
    hidden_states = torch.randn((row_capacity, hidden_size), device="cuda", dtype=payload_dtype) / 4
    layer_weights = weights.MoeFfnWeights(
        expert_gate_up_weight=torch.randn(
            (expert_count, 2 * local_intermediate_size, hidden_size),
            device="cuda",
            dtype=payload_dtype,
        )
        / 4,
        expert_down_weight=torch.randn(
            (expert_count, hidden_size, local_intermediate_size),
            device="cuda",
            dtype=payload_dtype,
        )
        / 4,
        router=None,
    )
    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (1, 3), (-1, -1)),
        device="cuda",
        dtype=torch.int32,
    )
    routed_scaling_factor = 1.5
    topk_weights = torch.tensor(
        ((0.75, 0.25), (0.6, 0.4), (0.9, 0.1), (0.0, 0.0)),
        device="cuda",
        dtype=torch.float32,
    )
    w13_config, w2_config = operators.select_moe_kernel_configs(
        layer_weights=layer_weights,
        row_capacity=row_capacity,
        effective_topk=effective_topk,
    )
    maximum_padded, expert_block_count, cumsum_count = execution.moe_alignment_workspace_shapes(
        row_capacity=row_capacity,
        effective_topk=effective_topk,
        expert_count=expert_count,
        block_size_m=w13_config["BLOCK_SIZE_M"],
    )
    sorted_token_ids = torch.empty(maximum_padded, device="cuda", dtype=torch.int32)
    expert_ids = torch.empty(expert_block_count, device="cuda", dtype=torch.int32)
    num_tokens_post_padded = torch.empty(1, device="cuda", dtype=torch.int32)
    cumsum_buffer = torch.empty(cumsum_count, device="cuda", dtype=torch.int32)
    overlap = torch.empty(
        max(route_count * 2 * local_intermediate_size, route_count * hidden_size),
        device="cuda",
        dtype=payload_dtype,
    )
    gate_up = overlap[: route_count * 2 * local_intermediate_size].view(
        route_count,
        2 * local_intermediate_size,
    )
    route_outputs = overlap[: route_count * hidden_size].view(
        row_capacity,
        effective_topk,
        hidden_size,
    )
    activated = torch.empty(
        (route_count, local_intermediate_size),
        device="cuda",
        dtype=payload_dtype,
    )
    output = torch.empty_like(hidden_states)

    def compute() -> torch.Tensor:
        return operators.compute_moe_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            cumsum_buffer=cumsum_buffer,
            gate_up=gate_up,
            activated=activated,
            route_outputs=route_outputs,
            output=output,
            w13_config=w13_config,
            w2_config=w2_config,
            activation=ActivationKind.SILU,
            routed_scaling_factor=routed_scaling_factor,
        )

    returned = compute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute()
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros_like(hidden_states, dtype=torch.float32)
    for row in range(3):
        for slot in range(effective_topk):
            expert = int(topk_ids[row, slot])
            projected = torch.mv(layer_weights.expert_gate_up_weight[expert], hidden_states[row])
            gate, up = projected.chunk(2)
            activated_reference = (torch.nn.functional.silu(gate.float()) * up.float()).to(payload_dtype)
            route = torch.mv(layer_weights.expert_down_weight[expert], activated_reference)
            expected[row] += route.float() * topk_weights[row, slot]
        expected[row] *= routed_scaling_factor
    expected = expected.to(payload_dtype)

    assert returned is output
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)
