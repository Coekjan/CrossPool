import pytest
import torch

from xpool import ffn
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear


@pytest.mark.parametrize(
    "capacity,precision,dtype,bias,hidden,experts,capability,expected",
    (
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            True,
            2880,
            128,
            (9, 0),
            linear.LinearOperationKind.FLASHINFER_TINY,
        ),
        (
            256,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            True,
            2880,
            128,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            False,
            2880,
            128,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.FP32_LOGITS,
            torch.bfloat16,
            True,
            2880,
            128,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.float16,
            True,
            2880,
            128,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            True,
            2880,
            128,
            (8, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            True,
            2880,
            31,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
        (
            128,
            ffn.RouterProjectionPrecision.PAYLOAD,
            torch.bfloat16,
            True,
            2881,
            128,
            (9, 0),
            linear.LinearOperationKind.TORCH,
        ),
    ),
)
def test_router_projection_selection_preserves_existing_paths_and_tinygemm_domain(
    capacity: int,
    precision: ffn.RouterProjectionPrecision,
    dtype: torch.dtype,
    bias: bool,
    hidden: int,
    experts: int,
    capability: tuple[int, int],
    expected: linear.LinearOperationKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear, "get_device_capability", lambda device: capability)
    signature = execution.MoeRouterExecutionSignature(
        algorithm=ffn.RouterAlgorithm.SOFTMAX,
        projection_precision=precision,
        weight_rounding=ffn.TopKWeightRounding.LOGITS_DTYPE,
        projection=weights.ProjectionLayout(
            values=TensorLayout((experts, hidden), dtype, 0),
            scales=None,
            bias=TensorLayout(
                (experts,), dtype if precision is ffn.RouterProjectionPrecision.PAYLOAD else torch.float32, 0
            )
            if bias
            else None,
        ),
        routed_expert_count=experts,
        router_workspace_bytes=execution.router_workspace_bytes(
            precision,
            payload_dtype=dtype,
            payload_row_capacity=capacity,
            hidden_size=hidden,
            routed_expert_count=experts,
            routed_topk=4,
        ),
        correction_bias=None,
        renormalize=True,
    )
    assert (
        linear.select_router_projection_operation(
            signature, payload_dtype=dtype, row_capacity=capacity, device=torch.device("cuda", 0)
        )
        is expected
    )
    assert not torch.cuda.is_initialized()
