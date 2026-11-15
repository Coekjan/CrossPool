"""CUDA behavior tests for fixed-Capacity FFN components."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional
from tests.suites.integration.runtime.ffnagent.support import dense_plan

import xtest
from xpool import ffn, quantization
from xpool.ffn import SiluActivation
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import dense, moe
from xpool.runtime.ffnagent.operators.activation import Activation

ROW_CAPACITY = 4
HIDDEN_SIZE = 16
LOCAL_INTERMEDIATE_SIZE = 8


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16), ids=("bfloat16", "float16"))
@xtest.requirements(device_count=1)
def test_dense_matches_reference_without_steady_allocation(payload_dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn((ROW_CAPACITY, HIDDEN_SIZE), device="cuda", dtype=payload_dtype)
    layer_weights = weights.DenseFfnWeights(
        i_proj=weights.ProjectionWeights(
            values=torch.randn(
                (2 * LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE),
                device="cuda",
                dtype=payload_dtype,
            ),
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionWeights(
            values=torch.randn(
                (HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE),
                device="cuda",
                dtype=payload_dtype,
            ),
            scales=None,
            bias=None,
        ),
    )
    plan = dense_plan(hidden_states, layer_weights, Activation(function=SiluActivation(), gate=ffn.GateTransform()))
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = dense.Dense(plan, workspace)
    output = torch.empty_like(hidden_states)

    operation.run_into(hidden_states=hidden_states, layer_weights=layer_weights, output=output)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    operation.run_into(hidden_states=hidden_states, layer_weights=layer_weights, output=output)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.run_into(hidden_states=hidden_states, layer_weights=layer_weights, output=output)
    hidden_states.fill_(3)
    graph.replay()
    torch.cuda.synchronize()

    gate_up = torch.mm(hidden_states, layer_weights.i_proj.values.t())
    gate, up = gate_up.chunk(2, dim=-1)
    activated_up = (torch.nn.functional.silu(gate.float()) * up.float()).to(payload_dtype)
    expected = torch.mm(activated_up, layer_weights.o_proj.values.t())
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16))
@xtest.requirements(device_count=1)
def test_ungated_dense_bias_and_graph_replay(payload_dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    hidden = torch.randn((ROW_CAPACITY, HIDDEN_SIZE), device="cuda", dtype=payload_dtype)
    projection = weights.DenseFfnWeights(
        i_proj=weights.ProjectionWeights(
            values=torch.randn((LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE), device="cuda", dtype=payload_dtype),
            scales=None,
            bias=torch.randn((LOCAL_INTERMEDIATE_SIZE,), device="cuda", dtype=payload_dtype),
        ),
        o_proj=weights.ProjectionWeights(
            values=torch.randn((HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE), device="cuda", dtype=payload_dtype),
            scales=None,
            bias=torch.randn((HIDDEN_SIZE,), device="cuda", dtype=payload_dtype),
        ),
    )
    activation = Activation(function=ffn.ReluActivation(), gate=None)
    plan = dense_plan(hidden, projection, activation)
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = dense.Dense(plan, workspace)
    output = torch.empty_like(hidden)
    operation.run_into(hidden_states=hidden, layer_weights=projection, output=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.run_into(hidden_states=hidden, layer_weights=projection, output=output)
    hidden.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    first = projection.i_proj
    second = projection.o_proj
    assert first.bias is not None and second.bias is not None
    expected = torch.addmm(second.bias, torch.addmm(first.bias, hidden, first.values.t()).relu(), second.values.t())
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)


@pytest.mark.parametrize(
    "payload_dtype,with_bias,function,gate_transform",
    (
        (torch.bfloat16, False, SiluActivation(), ffn.GateTransform()),
        (torch.float16, False, SiluActivation(), ffn.GateTransform()),
        (torch.bfloat16, True, SiluActivation(), ffn.GateTransform()),
        (torch.bfloat16, True, SiluActivation(alpha=1.702), ffn.GateTransform(limit=7.0, up_offset=1.0)),
        (torch.bfloat16, True, ffn.ReluActivation(), ffn.GateTransform(limit=4.0, up_offset=-0.5)),
        (torch.bfloat16, True, ffn.GeluActivation(approximate="none"), ffn.GateTransform()),
        (torch.bfloat16, True, ffn.GeluActivation(approximate="tanh"), ffn.GateTransform()),
    ),
)
@xtest.requirements(device_count=1)
def test_floating_experts_match_reference_with_overlapped_workspace_and_graph(
    payload_dtype: torch.dtype,
    with_bias: bool,
    function: ffn.ActivationSpec,
    gate_transform: ffn.GateTransform,
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
        i_proj=weights.ProjectionWeights(
            values=torch.randn(
                (expert_count, 2 * local_intermediate_size, hidden_size),
                device="cuda",
                dtype=payload_dtype,
            )
            / 4,
            scales=None,
            bias=None,
        ),
        o_proj=weights.ProjectionWeights(
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
        activation=Activation(function=function, gate=gate_transform),
        routed_scaling_factor=routed_scaling_factor,
        router=None,
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.INTERLEAVED,
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
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
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
    plan = moe.ExpertExecutionPlan.prepare(signature)
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = moe.Experts(plan, workspace)
    i_proj = weights.ProjectionWeights(
        values=layer_weights.i_proj.values,
        scales=None,
        bias=torch.randn((expert_count, 2 * local_intermediate_size), dtype=torch.float32, device="cuda")
        if with_bias
        else None,
    )
    o_proj = weights.ProjectionWeights(
        values=layer_weights.o_proj.values,
        scales=None,
        bias=torch.randn((expert_count, hidden_size), dtype=torch.float32, device="cuda") if with_bias else None,
    )
    output = torch.empty_like(hidden_states)

    def compute() -> None:
        operation.run_into(
            hidden_states=hidden_states,
            i_proj=i_proj,
            o_proj=o_proj,
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
            projected = torch.mv(i_proj.values[expert].float(), hidden_states[row].float())
            if i_proj.bias is not None:
                projected += i_proj.bias[expert]
            projected = projected.to(payload_dtype)
            gate, up = projected[::2], projected[1::2]
            gate, up = gate.float(), up.float()
            if gate_transform.limit is not None:
                gate = gate.clamp(max=gate_transform.limit)
                up = up.clamp(min=-gate_transform.limit, max=gate_transform.limit)
            match function:
                case ffn.ReluActivation():
                    activated = gate.relu()
                case ffn.SiluActivation(alpha=alpha):
                    activated = gate * torch.sigmoid(alpha * gate)
                case ffn.GeluActivation(approximate=approximate):
                    activated = torch.nn.functional.gelu(gate, approximate=approximate)
            activated_reference = (activated * (up + gate_transform.up_offset)).to(payload_dtype)
            route = torch.mv(o_proj.values[expert].float(), activated_reference.float())
            if o_proj.bias is not None:
                route += o_proj.bias[expert]
            expected[row] += (route * topk_weights[row, slot]).to(payload_dtype).float()
        expected[row] *= routed_scaling_factor
    expected = expected.to(payload_dtype)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)
