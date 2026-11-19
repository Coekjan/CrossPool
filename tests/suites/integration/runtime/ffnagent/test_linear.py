import pytest
import torch
from triton_kernels import matmul, tensor
from triton_kernels.matmul_details import opt_flags

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
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    projection.bias.mul_(0.5)
    projection.values.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
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


@xtest.requirements(device_count=1)
def test_gemm_configuration_restores_public_flags_after_operation_failure() -> None:
    inputs = torch.empty((512, 64), device="cuda", dtype=torch.bfloat16)
    projection = weights.ProjectionLayout(
        quantization=quantization.Unquantized(),
        gate_up_order=None,
        values=TensorLayout((256, 64), inputs.dtype, 0),
        scales=None,
        bias=None,
    )
    plan = linear.GemmPlan.prepare(
        input_layout=TensorLayout(tuple(inputs.shape), inputs.dtype, 0),
        weight_layout=projection,
        result_layout=TensorLayout((512, 256), inputs.dtype, 0),
        activation=None,
    )
    operation = linear.Gemm(plan, torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8))
    operation.prepare_launch(inputs, projection.zero_probe(inputs.device))
    dtype = tensor.torch_dtype_to_dtype(inputs.dtype)

    def query_flags() -> opt_flags.OptFlags:
        return opt_flags.make_opt_flags(
            out_dtype=dtype,
            lhs_dtype=dtype,
            rhs_dtype=dtype,
            precision_config=matmul.PrecisionConfig(out_dtype=inputs.dtype, allow_tf32=False),
            batch_size=1,
            m=1,
            n=16,
            k=64,
            ragged_metadata=None,
            can_use_persistent_tma=False,
            can_use_split_k=False,
            epilogue_effective_itemsize=False,
            x_transpose=False,
            has_y_acc_in=False,
            block_k=None,
        )

    before = query_flags()
    with pytest.raises(RuntimeError, match="operation failed"):
        with operation.configuration():
            assert query_flags() != before
            raise RuntimeError("operation failed")
    assert query_flags() == before
