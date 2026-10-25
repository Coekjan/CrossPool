"""Packed math and data-dependent routing through the selected upstream kernels."""

import pytest
import torch
from triton_kernels import matmul
from triton_kernels.matmul_details import opt_flags

import xtest
from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, Unquantized
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators.triton_moe import TritonKernelsExperts
from xtest.harness.native.ffn.qualification import packed_moe_reference


def packed_layer(experts: int, intermediate_size: int) -> ffn.MoeFfnSpec:
    return ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
        router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
        expert_intermediate_size=intermediate_size,
        routed_expert_count=experts,
        shared_expert_count=0,
        routed_topk=4,
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
    "capacity,hidden_size,intermediate_size,experts,activation",
    (
        (65, 64, 64, 8, ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)),
        (16385, 64, 64, 8, ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)),
        (65, 2880, 736, 32, ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)),
        (65, 2880, 384, 32, ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0)),
        (65, 64, 64, 8, ffn.SiluActivation()),
        (65, 64, 64, 8, ffn.SiluActivation(alpha=1.3, limit=4.0, up_offset=-0.5)),
    ),
)
@xtest.requirements(device_count=1)
def test_packed_experts_keep_dynamic_routes_and_capacity_tails_without_capture_allocation(
    capacity: int,
    hidden_size: int,
    intermediate_size: int,
    experts: int,
    activation: ffn.ActivationSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(73)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    layer = packed_layer(experts, intermediate_size)
    layout = weights.ExpertLayout.from_layer(
        layer,
        hidden_size=hidden_size,
        local_intermediate_size=intermediate_size,
        payload_dtype=torch.bfloat16,
        bias_dtype=TritonKernelsExperts.bias_dtype,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    gate_up, down = layout.gate_up.allocate(device), layout.down.allocate(device)
    for projection in (gate_up, down):
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
        effective_topk=4,
        activation=activation,
        routed_scaling_factor=1.3,
        router=None,
        operation=execution.ExpertOperationKind.TRITON_KERNELS,
        gate_up=layout.gate_up,
        down=layout.down,
    )
    region_specs, offsets, extent = TritonKernelsExperts.workspace_layout(signature)
    assert all(
        offset + torch.Size(shape).numel() * dtype.itemsize <= extent
        for offset, (dtype, shape) in zip(offsets, region_specs, strict=True)
    )
    workspace = torch.empty(extent, dtype=torch.uint8, device=device)
    operation = TritonKernelsExperts(signature, workspace)
    hidden = torch.randn((capacity, hidden_size), device=device, dtype=torch.bfloat16)
    rows = torch.arange(capacity, device=device)[:, None]
    slots = torch.arange(4, device=device)[None, :]
    ids = ((rows * 3 + slots) % experts).to(torch.int32)
    probabilities = torch.tensor((0.1, 0.2, 0.3, 0.4), device=device).expand(capacity, 4).contiguous()
    output = torch.empty_like(hidden)
    selector, constraints, flags = matmul.make_opt_flags, opt_flags._opt_flags_constraints, opt_flags._opt_flags

    def run() -> None:
        operation.run(
            hidden_states=hidden, gate_up=gate_up, down=down, topk_ids=ids, topk_weights=probabilities, output=output
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
        probabilities.fill_(0.25)
        probabilities[live:] = 0
        expected = packed_moe_reference(
            hidden,
            gate_up=gate_up,
            down=down,
            ids=ids,
            probabilities=probabilities,
            activation=activation,
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
    assert matmul.make_opt_flags is selector
    assert opt_flags._opt_flags_constraints is constraints
    assert opt_flags._opt_flags is flags
    with pytest.raises(RuntimeError, match="construction failure"):
        with operation.configuration():
            raise RuntimeError("construction failure")
    assert matmul.make_opt_flags is selector
    assert opt_flags._opt_flags_constraints is constraints
    assert opt_flags._opt_flags is flags
