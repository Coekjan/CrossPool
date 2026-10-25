"""Floating projections into caller-owned output, with startup kernel selection."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch.cuda import get_device_capability

from xpool import ffn
from xpool.runtime.ffnagent import execution, weights


class LinearOperationKind(IntEnum):
    """Concrete launch chosen once for a fixed capture environment."""

    TORCH = 1
    FLASHINFER_TINY = 2


def select_router_projection_operation(
    signature: execution.MoeRouterExecutionSignature,
    *,
    payload_dtype: torch.dtype,
    row_capacity: int,
    device: torch.device,
) -> LinearOperationKind:
    """Use the qualified biased BF16 Router fast path when its geometry fits.

    Existing unbiased Router paths remain unchanged. Device eligibility follows
    FlashInfer's SM90+ TMA requirement, not a model or device-name allowlist.
    This query runs before capture; replay never selects an implementation.
    """

    projection = signature.projection
    if (
        signature.projection_precision is ffn.RouterProjectionPrecision.PAYLOAD
        and payload_dtype is torch.bfloat16
        and projection.values.dtype is torch.bfloat16
        and projection.bias is not None
        and projection.bias.dtype is torch.bfloat16
        and row_capacity <= 128
        and projection.values.shape[0] % 16 == 0
        and projection.values.shape[1] % 64 == 0
        and device.type == "cuda"
        and get_device_capability(device)[0] >= 9
    ):
        return LinearOperationKind.FLASHINFER_TINY
    return LinearOperationKind.TORCH


def compute_projection(
    operation: LinearOperationKind,
    *,
    hidden_states: torch.Tensor,
    projection: weights.ProjectionWeights,
    output: torch.Tensor,
) -> None:
    """Write a validated floating projection without allocating output storage.

    The owning Dense/Router entry validates resources. An FP32 destination with
    payload-typed operands uses Torch's FP32-output GEMM rather than a separate
    result conversion. Optional bias is fused before destination rounding.
    """

    match operation:
        case LinearOperationKind.FLASHINFER_TINY:
            # FlashInfer import initializes CUDA; existing pre-capture warmup owns that side effect.
            from flashinfer.gemm import tinygemm_bf16

            tinygemm_bf16(hidden_states, projection.values, output, projection.bias, use_pdl=False)
        case LinearOperationKind.TORCH:
            if output.dtype is torch.float32 and hidden_states.dtype in (torch.bfloat16, torch.float16):
                if projection.bias is None:
                    torch.mm(hidden_states, projection.values.t(), out=output, out_dtype=torch.float32)
                else:
                    torch.addmm(
                        projection.bias, hidden_states, projection.values.t(), out=output, out_dtype=torch.float32
                    )
            elif projection.bias is None:
                torch.mm(hidden_states, projection.values.t(), out=output)
            else:
                torch.addmm(projection.bias, hidden_states, projection.values.t(), out=output)
