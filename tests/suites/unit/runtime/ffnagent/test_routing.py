from __future__ import annotations

import pytest
import torch

from xpool import ffn, quantization
from xpool.ffn import RouterProjectionPrecision
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import routing


@pytest.mark.parametrize(
    "precision, expected_dtype, expected_bytes",
    (
        (RouterProjectionPrecision.PAYLOAD, torch.bfloat16, 48),
        (RouterProjectionPrecision.FP32_LOGITS, torch.bfloat16, 64),
        (RouterProjectionPrecision.FP32, torch.float32, 128),
    ),
)
def test_router_resources_follow_execution_semantics(
    precision: RouterProjectionPrecision, expected_dtype: torch.dtype, expected_bytes: int
) -> None:
    assert precision.weight_dtype(payload_dtype=torch.bfloat16) is expected_dtype
    plan = routing.RouterExecutionPlan.prepare(
        execution.MoeRouterExecutionSignature(
            algorithm=ffn.RouterAlgorithm.SOFTMAX,
            projection_precision=precision,
            weight_rounding=ffn.TopKWeightRounding.NONE,
            projection=weights.ProjectionLayout(
                quantization=quantization.Unquantized(),
                values=TensorLayout((4, 8), expected_dtype, 0),
                scales=None,
                bias=None,
                gate_up_order=None,
            ),
            routed_expert_count=4,
            correction_bias=None,
            renormalize=True,
        ),
        input_layout=TensorLayout((2, 8), torch.bfloat16, 0),
        routed_topk=2,
    )
    assert plan.workspace.nbytes == expected_bytes
