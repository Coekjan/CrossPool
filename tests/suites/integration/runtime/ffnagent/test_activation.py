import pytest
import torch

import xtest
from xpool import ffn
from xpool.runtime.ffnagent.operators import activation


@pytest.mark.parametrize("payload_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("in_place", (True, False))
@xtest.requirements(device_count=1)
def test_relu_preserves_all_16bit_float_patterns(payload_dtype: torch.dtype, in_place: bool) -> None:
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(payload_dtype)
    expected = values.relu().view(torch.int16)
    output = values if in_place else torch.empty_like(values)
    operation = activation.Activation(function=ffn.ReluActivation(), gate=None)
    operation.apply_into(inputs=values, output=output, order=None)
    assert torch.equal(output.view(torch.int16), expected)


@pytest.mark.parametrize("payload_dtype,in_place", ((torch.float16, True), (torch.bfloat16, False)))
@xtest.requirements(device_count=1)
def test_ungated_silu_replays_with_caller_owned_output(payload_dtype: torch.dtype, in_place: bool) -> None:
    inputs = torch.linspace(-12, 12, 480, device="cuda", dtype=payload_dtype).reshape(6, 80)
    output = inputs if in_place else torch.empty_like(inputs)
    operation = activation.Activation(function=ffn.SiluActivation(alpha=1.3), gate=None)

    def run() -> None:
        operation.apply_into(inputs=inputs, output=output, order=None)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    inputs.neg_()
    reference = inputs.float().clone()
    graph.replay()
    torch.cuda.synchronize()
    expected = (reference * torch.sigmoid(1.3 * reference)).to(payload_dtype)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "function,transform,order,payload_dtype",
    (
        (ffn.SiluActivation(), ffn.GateTransform(), ffn.GateUpOrder.CONCATENATED, torch.bfloat16),
        (ffn.SiluActivation(), ffn.GateTransform(), ffn.GateUpOrder.INTERLEAVED, torch.float16),
        (
            ffn.SiluActivation(alpha=1.702),
            ffn.GateTransform(limit=7.0, up_offset=1.0),
            ffn.GateUpOrder.CONCATENATED,
            torch.bfloat16,
        ),
        (
            ffn.SiluActivation(alpha=1.3),
            ffn.GateTransform(limit=4.0, up_offset=-0.5),
            ffn.GateUpOrder.INTERLEAVED,
            torch.bfloat16,
        ),
        (ffn.ReluActivation(), ffn.GateTransform(), ffn.GateUpOrder.CONCATENATED, torch.float16),
        (ffn.GeluActivation(approximate="none"), ffn.GateTransform(), ffn.GateUpOrder.CONCATENATED, torch.float16),
        (ffn.GeluActivation(approximate="tanh"), ffn.GateTransform(), ffn.GateUpOrder.INTERLEAVED, torch.bfloat16),
        (
            ffn.ReluActivation(),
            ffn.GateTransform(limit=4.0, up_offset=-0.5),
            ffn.GateUpOrder.INTERLEAVED,
            torch.bfloat16,
        ),
    ),
)
@xtest.requirements(device_count=1)
def test_gated_activation_writes_declared_formula_and_replays_without_allocation(
    function: ffn.ActivationSpec,
    transform: ffn.GateTransform,
    order: ffn.GateUpOrder,
    payload_dtype: torch.dtype,
) -> None:
    inputs = torch.linspace(-12, 12, 6 * 80, device="cuda", dtype=payload_dtype).reshape(6, 80)
    output = torch.empty((6, 40), device="cuda", dtype=payload_dtype)
    operation = activation.Activation(function=function, gate=transform)

    def run() -> None:
        operation.apply_into(inputs=inputs, output=output, order=order)

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
    if transform.limit is not None:
        gate = gate.clamp(max=transform.limit)
        up = up.clamp(min=-transform.limit, max=transform.limit)
    match function:
        case ffn.ReluActivation():
            activated = gate.relu()
        case ffn.SiluActivation(alpha=alpha):
            activated = gate * torch.sigmoid(alpha * gate)
        case ffn.GeluActivation(approximate=approximate):
            activated = torch.nn.functional.gelu(gate, approximate=approximate)
    expected = (activated * (up + transform.up_offset)).to(payload_dtype)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
