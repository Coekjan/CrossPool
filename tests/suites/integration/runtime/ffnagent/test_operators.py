"""CUDA behavior tests for fixed-Capacity FFN components."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional

import xtest
from xpool import ffn
from xpool.ffn import SiluActivation
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import dense, sglang_moe

ROW_CAPACITY = 4
HIDDEN_SIZE = 16
LOCAL_INTERMEDIATE_SIZE = 8


def dense_values(
    payload_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, weights.DenseFfnWeights, torch.Tensor, torch.Tensor]:
    """Allocate one small valid Dense component input and caller-owned outputs."""

    hidden_states = torch.randn((ROW_CAPACITY, HIDDEN_SIZE), device="cuda", dtype=payload_dtype)
    layer_weights = weights.DenseFfnWeights(
        gate_up=weights.ProjectionWeights(
            values=torch.randn(
                (2 * LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE),
                device="cuda",
                dtype=payload_dtype,
            ),
            scales=None,
            bias=None,
        ),
        down=weights.ProjectionWeights(
            values=torch.randn(
                (HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE),
                device="cuda",
                dtype=payload_dtype,
            ),
            scales=None,
            bias=None,
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
@xtest.requirements(device_count=1)
def test_compute_dense_partial_matches_reference_without_steady_allocation(payload_dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    hidden_states, layer_weights, workspace, output = dense_values(payload_dtype)

    returned = dense.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=SiluActivation(),
    )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    dense.compute_dense_partial(
        hidden_states=hidden_states,
        layer_weights=layer_weights,
        workspace=workspace,
        output=output,
        activation=SiluActivation(),
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=workspace,
            output=output,
            activation=SiluActivation(),
        )
    hidden_states.fill_(3)
    graph.replay()
    torch.cuda.synchronize()

    gate_up = torch.mm(hidden_states, layer_weights.gate_up.values.t())
    gate, up = gate_up.chunk(2, dim=-1)
    activated_up = (torch.nn.functional.silu(gate.float()) * up.float()).to(payload_dtype)
    expected = torch.mm(activated_up, layer_weights.down.values.t())
    assert returned is output
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)


@xtest.requirements(device_count=1)
def test_compute_dense_partial_rejects_misaligned_workspace() -> None:
    hidden_states, layer_weights, workspace, output = dense_values()
    storage = torch.empty(workspace.numel() + 1, device="cuda", dtype=torch.uint8)
    misaligned_workspace = storage[1:]

    with pytest.raises(ValueError, match="16-byte aligned"):
        dense.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=misaligned_workspace,
            output=output,
            activation=SiluActivation(),
        )


@pytest.mark.parametrize(
    "payload_dtype,with_bias,activation",
    (
        (torch.bfloat16, False, SiluActivation()),
        (torch.float16, False, SiluActivation()),
        (torch.bfloat16, True, SiluActivation()),
        (torch.bfloat16, True, SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)),
    ),
)
@xtest.requirements(device_count=1)
def test_floating_experts_match_reference_with_overlapped_workspace_and_graph(
    payload_dtype: torch.dtype,
    with_bias: bool,
    activation: ffn.ActivationSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    row_capacity = 4
    hidden_size = 16
    local_intermediate_size = 8
    expert_count = 4
    effective_topk = 2
    hidden_states = torch.randn((row_capacity, hidden_size), device="cuda", dtype=payload_dtype) / 4
    layer_weights = weights.MoeFfnWeights(
        gate_up=weights.ProjectionWeights(
            values=torch.randn(
                (expert_count, 2 * local_intermediate_size, hidden_size),
                device="cuda",
                dtype=payload_dtype,
            )
            / 4,
            scales=None,
            bias=None,
        ),
        down=weights.ProjectionWeights(
            values=torch.randn(
                (expert_count, hidden_size, local_intermediate_size),
                device="cuda",
                dtype=payload_dtype,
            )
            / 4,
            scales=None,
            bias=None,
        ),
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
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=payload_dtype,
        payload_row_capacity=row_capacity,
        hidden_size=hidden_size,
        local_intermediate_size=local_intermediate_size,
        expert_count=expert_count,
        effective_topk=effective_topk,
        activation=activation,
        routed_scaling_factor=routed_scaling_factor,
        router=None,
        operation=execution.ExpertOperationKind.SGLANG_TRITON,
        gate_up=weights.ProjectionLayout(
            values=TensorLayout((expert_count, 2 * local_intermediate_size, hidden_size), payload_dtype, 0),
            scales=None,
            bias=TensorLayout(
                (
                    expert_count,
                    2 * local_intermediate_size,
                ),
                torch.float32,
                0,
            )
            if with_bias
            else None,
        ),
        down=weights.ProjectionLayout(
            values=TensorLayout((expert_count, hidden_size, local_intermediate_size), payload_dtype, 0),
            scales=None,
            bias=TensorLayout(
                (
                    expert_count,
                    hidden_size,
                ),
                torch.float32,
                0,
            )
            if with_bias
            else None,
        ),
    )
    workspace = torch.empty(
        sglang_moe.SglangTritonExperts.workspace_layout(signature)[2], device="cuda", dtype=torch.uint8
    )
    operation = sglang_moe.SglangTritonExperts(signature, workspace)
    gate_up = weights.ProjectionWeights(
        values=layer_weights.gate_up.values,
        scales=None,
        bias=torch.randn((expert_count, 2 * local_intermediate_size), dtype=torch.float32, device="cuda")
        if with_bias
        else None,
    )
    down = weights.ProjectionWeights(
        values=layer_weights.down.values,
        scales=None,
        bias=torch.randn((expert_count, hidden_size), dtype=torch.float32, device="cuda") if with_bias else None,
    )
    output = torch.empty_like(hidden_states)

    def compute() -> None:
        operation.run(
            hidden_states=hidden_states,
            gate_up=gate_up,
            down=down,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
        )

    compute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        compute()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert before == after
    hidden_states.fill_(3)
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros_like(hidden_states, dtype=torch.float32)
    for row in range(3):
        for slot in range(effective_topk):
            expert = int(topk_ids[row, slot])
            projected = torch.mv(gate_up.values[expert].float(), hidden_states[row].float())
            if gate_up.bias is not None:
                projected += gate_up.bias[expert]
            projected = projected.to(payload_dtype)
            gate, up = projected.chunk(2)
            gate, up = gate.float(), up.float()
            if activation.limit is not None:
                gate = gate.clamp(max=activation.limit)
                up = up.clamp(min=-activation.limit, max=activation.limit)
            activated_reference = (gate * torch.sigmoid(activation.alpha * gate) * (up + activation.up_offset)).to(
                payload_dtype
            )
            route = torch.mv(down.values[expert].float(), activated_reference.float())
            if down.bias is not None:
                route += down.bias[expert]
            expected[row] += (route * topk_weights[row, slot]).to(payload_dtype).float()
        expected[row] *= routed_scaling_factor
    expected = expected.to(payload_dtype)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)
