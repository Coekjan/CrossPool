"""Packed math and data-dependent routing through the selected upstream kernels."""

import pytest
import torch

import xtest
from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, Unquantized
from xpool.runtime.ffnagent import execution, preparation
from xpool.runtime.ffnagent.operators import moe
from xpool.runtime.ffnagent.operators.activation import Activation
from xtest.harness.native.ffn.qualification import packed_moe_reference
from xtest.harness.support.config import TEST_MODEL_ID


def packed_layer(experts: int, intermediate_size: int, topk: int) -> ffn.MoeFfnSpec:
    return ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
        router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
        expert_intermediate_size=intermediate_size,
        routed_expert_count=experts,
        shared_expert_count=0,
        routed_topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key="router.weight"),
        routed_experts=ffn.StackedExpertSource(
            gate_up=ffn.ProjectionSpec(
                quantization=Mxfp4(),
                values_key="gate_up.blocks",
                scales_key="gate_up.scales",
                values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                bias_key="gate_up.bias",
            ),
            down=ffn.ProjectionSpec(
                quantization=Mxfp4(),
                values_key="down.blocks",
                scales_key="down.scales",
                values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                bias_key="down.bias",
            ),
            gate_up_order=ffn.GateUpOrder.INTERLEAVED,
        ),
        shared_expert=None,
    )


@pytest.mark.parametrize(
    "capacity,hidden_size,intermediate_size,experts,function,gate_transform,topk",
    (
        (65, 64, 64, 8, ffn.SiluActivation(alpha=1.702), ffn.GateTransform(limit=7.0, up_offset=1.0), 5),
        (16385, 64, 64, 8, ffn.SiluActivation(alpha=1.702), ffn.GateTransform(limit=7.0, up_offset=1.0), 4),
        (65, 2880, 736, 32, ffn.SiluActivation(alpha=1.702), ffn.GateTransform(limit=7.0, up_offset=1.0), 4),
        (65, 2880, 384, 32, ffn.SiluActivation(alpha=1.702), ffn.GateTransform(limit=7.0, up_offset=1.0), 4),
        (65, 64, 64, 8, ffn.SiluActivation(), ffn.GateTransform(), 4),
        (65, 64, 64, 8, ffn.SiluActivation(alpha=1.3), ffn.GateTransform(limit=4.0, up_offset=-0.5), 4),
    ),
)
@xtest.requirements(device_count=1)
def test_packed_experts_keep_dynamic_routes_and_capacity_tails_without_capture_allocation(
    capacity: int,
    hidden_size: int,
    intermediate_size: int,
    experts: int,
    function: ffn.SiluActivation,
    gate_transform: ffn.GateTransform,
    topk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(73)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    layer = packed_layer(experts, intermediate_size, topk)
    layout = preparation.prepare_ffn_layer(
        ffn.FfnModelSpec(
            model_id=TEST_MODEL_ID,
            hidden_size=hidden_size,
            activation=function,
            gate_transform=gate_transform,
            layers=(layer,),
        ),
        layer_ordinal=0,
        payload_dtype=torch.bfloat16,
        tp_size=1,
        tp_rank=0,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    i_proj, o_proj = layout.i_proj.zero_probe(device), layout.o_proj.zero_probe(device)
    for projection in (i_proj, o_proj):
        assert projection.scales is not None and projection.bias is not None
        projection.values.copy_(
            torch.arange(projection.values.numel(), device=device).to(torch.uint8).view_as(projection.values)
        )
        projection.scales.copy_(torch.randint(123, 126, projection.scales.shape, device=device, dtype=torch.uint8))
        projection.bias.copy_(
            torch.linspace(-9, 9, projection.bias.numel(), device=device).view_as(projection.bias).bfloat16().float()
        )
    signature = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=capacity,
        hidden_size=hidden_size,
        local_intermediate_size=intermediate_size,
        expert_count=experts,
        effective_topk=topk,
        activation=Activation(function=function, gate=gate_transform),
        routed_scaling_factor=1.3,
        router=None,
        i_proj=layout.i_proj,
        o_proj=layout.o_proj,
    )
    plan = moe.ExpertExecutionPlan.prepare(signature)
    workspace = torch.empty(plan.workspace.nbytes, dtype=torch.uint8, device=device)
    operation = moe.Experts(plan, workspace)
    hidden = torch.randn((capacity, hidden_size), device=device, dtype=torch.bfloat16)
    rows = torch.arange(capacity, device=device)[:, None]
    slots = torch.arange(topk, device=device)[None, :]
    ids = ((rows * 3 + slots) % experts).to(torch.int32)
    probabilities = torch.arange(1, topk + 1, dtype=torch.float32, device=device)
    probabilities = (probabilities / probabilities.sum()).expand(capacity, topk).contiguous()
    output = torch.empty_like(hidden)

    def run() -> None:
        operation.run_into(
            hidden_states=hidden, i_proj=i_proj, o_proj=o_proj, topk_ids=ids, topk_weights=probabilities, output=output
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert after == before
    for live, shift in ((capacity, 0), (17, 2), (1, 0), (capacity, 1)):
        ids.copy_(((rows * 3 + slots + shift) % experts).to(torch.int32))
        ids[live:] = -1
        probabilities.fill_(1 / topk)
        probabilities[live:] = 0
        expected = packed_moe_reference(
            hidden,
            i_proj=i_proj,
            o_proj=o_proj,
            ids=ids,
            probabilities=probabilities,
            activation=function,
            gate_transform=gate_transform,
            routed_scaling_factor=signature.routed_scaling_factor,
        )
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        counts = torch.bincount(ids[ids >= 0].long(), minlength=experts)
        torch.testing.assert_close(operation.metadata.slice_sizes, counts.to(torch.int32), rtol=0, atol=0)
        scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
        error = output.float() - expected.float()
        assert error.square().mean().sqrt() / scale <= 0.01
        assert (error.abs() / torch.maximum(expected.float().abs(), scale)).max() <= 0.10
        assert torch.count_nonzero(output[live:]) == 0
