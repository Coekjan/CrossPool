import pytest
import torch

from xpool import ffn, quantization
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.operators import linear
from xpool.runtime.ffnagent.operators.activation import Activation


@pytest.mark.parametrize(
    "function",
    (
        ffn.ReluActivation(),
        ffn.SiluActivation(alpha=1.3),
        ffn.GeluActivation(approximate="none"),
        ffn.GeluActivation(approximate="tanh"),
    ),
)
@pytest.mark.parametrize("gate", (None, ffn.GateTransform()))
def test_gemm_plan_prepares_fused_result_without_device_work(
    function: ffn.ActivationSpec, gate: ffn.GateTransform | None
) -> None:
    input_layout = TensorLayout((4, 32), torch.bfloat16, 0)
    result_layout = TensorLayout((4, 16), torch.bfloat16, 0)
    projection = weights.ProjectionLayout(
        quantization=quantization.Unquantized(),
        gate_up_order=ffn.GateUpOrder.INTERLEAVED if gate is not None else None,
        values=TensorLayout((32 if gate is not None else 16, 32), torch.bfloat16, 0),
        scales=None,
        bias=None,
    )
    plan = linear.GemmPlan.prepare(
        input_layout=input_layout,
        weight_layout=projection,
        result_layout=result_layout,
        activation=Activation(function=function, gate=gate),
    )
    assert plan.input_layout == input_layout
    assert plan.weight_layout == projection
    assert plan.output_layout == result_layout
    assert plan.fused_activation is not None
    assert plan.fused_activation.specs.reduction_n == (2 if gate is not None else 1)
    assert not torch.cuda.is_initialized()


def test_gemm_plan_materializes_nonadjacent_gate_rows() -> None:
    projection = weights.ProjectionLayout(
        quantization=quantization.Unquantized(),
        gate_up_order=ffn.GateUpOrder.CONCATENATED,
        values=TensorLayout((32, 32), torch.float16, 0),
        scales=None,
        bias=None,
    )
    plan = linear.GemmPlan.prepare(
        input_layout=TensorLayout((4, 32), torch.float16, 0),
        weight_layout=projection,
        result_layout=TensorLayout((4, 16), torch.float16, 0),
        activation=Activation(function=ffn.SiluActivation(), gate=ffn.GateTransform()),
    )
    assert plan.fused_activation is None
    assert plan.output_layout == TensorLayout((4, 32), torch.float16, 0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "rows,width,dtype,gated,expected_bytes",
    (
        (1, 65, torch.bfloat16, False, 0),
        (32, 65, torch.float32, False, 0),
        (257, 65, torch.float32, False, 8 * 257 * 65 * 4),
        (257, 130, torch.float32, True, 4 * 257 * 260 * 4),
        (257, 65, torch.bfloat16, False, 0),
    ),
)
def test_gemm_plan_accounts_for_projection_scratch_before_activation(
    rows: int, width: int, dtype: torch.dtype, gated: bool, expected_bytes: int
) -> None:
    projected_width = width * (2 if gated else 1)
    plan = linear.GemmPlan.prepare(
        input_layout=TensorLayout((rows, 2053), dtype, 0),
        weight_layout=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.INTERLEAVED if gated else None,
            values=TensorLayout((projected_width, 2053), dtype, 0),
            scales=None,
            bias=TensorLayout((projected_width,), torch.float32, 0),
        ),
        result_layout=TensorLayout((rows, width), dtype, 0),
        activation=Activation(function=ffn.SiluActivation(), gate=ffn.GateTransform()) if gated else None,
    )
    assert plan.workspace.nbytes == expected_bytes
    assert not torch.cuda.is_initialized()
