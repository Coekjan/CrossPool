import pytest
import torch

import xtest
from xpool import ffn, quantization
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.operators import linear
from xpool.runtime.ffnagent.operators.activation import Activation


@pytest.mark.parametrize(
    "rows,hidden,width,dtype,gated",
    (
        (1, 2053, 66, torch.bfloat16, True),
        (17, 67, 65, torch.float16, False),
        (257, 2053, 65, torch.float32, False),
        (129, 83, 130, torch.float32, True),
        (257, 67, 65, torch.bfloat16, False),
        (256, 2048, 512, torch.bfloat16, True),
    ),
)
@xtest.requirements(device_count=1)
def test_projection_replays_current_bias_and_activation_without_allocation(
    rows: int,
    hidden: int,
    width: int,
    dtype: torch.dtype,
    gated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    torch.manual_seed(101)
    projected_width = width * (2 if gated else 1)
    activation = (
        Activation(function=ffn.SiluActivation(alpha=1.3), gate=ffn.GateTransform(limit=4.0, up_offset=-0.5))
        if gated
        else None
    )
    layout = weights.ProjectionLayout(
        quantization=quantization.Unquantized(),
        gate_up_order=ffn.GateUpOrder.INTERLEAVED if gated else None,
        values=TensorLayout((projected_width, hidden), dtype, 0),
        scales=None,
        bias=TensorLayout((projected_width,), torch.float32, 0),
    )
    plan = linear.GemmPlan.prepare(
        input_layout=TensorLayout((rows, hidden), dtype, 0),
        weight_layout=layout,
        result_layout=TensorLayout((rows, width), dtype, 0),
        activation=activation,
    )
    inputs = torch.randn((rows, hidden), device="cuda", dtype=dtype) / hidden**0.5
    projection = layout.zero_probe(inputs.device)
    projection.values.normal_(std=0.1)
    assert projection.bias is not None
    projection.bias.normal_(std=0.1)
    output = torch.empty((rows, width), device="cuda", dtype=dtype)
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = linear.Gemm(plan, workspace)

    def run() -> None:
        operation.run_into(inputs, projection, output=output)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    projection.bias.mul_(0.5)
    projection.values.mul_(0.5)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    projected = torch.addmm(projection.bias.float(), inputs.float(), projection.values.float().t()).to(dtype)
    if gated:
        gate, up = projected.float()[..., ::2], projected.float()[..., 1::2]
        gate = gate.clamp_max(4.0)
        expected = (gate / (1 + torch.exp(-1.3 * gate)) * (up.clamp(-4.0, 4.0) - 0.5)).to(dtype)
    else:
        expected = projected
    torch.testing.assert_close(
        output, expected, rtol=1e-3 if dtype is torch.float32 else 0.02, atol=1e-4 if dtype is torch.float32 else 0.02
    )
