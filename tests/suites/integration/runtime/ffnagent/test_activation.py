import pytest
import torch

import xtest
from xpool import ffn
from xpool.runtime.ffnagent.operators import activation


@pytest.mark.parametrize(
    "spec,order,payload_dtype",
    (
        (ffn.SiluActivation(), ffn.GateUpOrder.CONCATENATED, torch.bfloat16),
        (ffn.SiluActivation(), ffn.GateUpOrder.INTERLEAVED, torch.float16),
        (
            ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0),
            ffn.GateUpOrder.CONCATENATED,
            torch.bfloat16,
        ),
        (
            ffn.SiluActivation(alpha=1.3, limit=4.0, up_offset=-0.5),
            ffn.GateUpOrder.INTERLEAVED,
            torch.bfloat16,
        ),
    ),
)
@xtest.requirements(device_count=1)
def test_silu_writes_declared_formula_and_replays_without_allocation(
    spec: ffn.ActivationSpec,
    order: ffn.GateUpOrder,
    payload_dtype: torch.dtype,
) -> None:
    inputs = torch.linspace(-12, 12, 6 * 80, device="cuda", dtype=payload_dtype).reshape(6, 80)
    output = torch.empty((6, 40), device="cuda", dtype=payload_dtype)

    def run() -> None:
        activation.apply_into(spec, inputs=inputs, output=output, order=order)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    inputs.neg_()
    graph.replay()
    torch.cuda.synchronize()

    gate, up = (
        inputs.float().chunk(2, dim=-1)
        if order is ffn.GateUpOrder.CONCATENATED
        else (inputs[:, ::2].float(), inputs[:, 1::2].float())
    )
    if spec.limit is not None:
        gate = gate.clamp(max=spec.limit)
        up = up.clamp(min=-spec.limit, max=spec.limit)
    expected = (gate * torch.sigmoid(spec.alpha * gate) * (up + spec.up_offset)).to(payload_dtype)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
