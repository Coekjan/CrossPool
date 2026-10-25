"""Closed selection shared by Expert preparation, sizing and capture."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.quantization import Mxfp4, Unquantized
from xpool.runtime.ffnagent import execution
from xpool.runtime.ffnagent.operators import sglang_moe, triton_moe


def select_expert_operation(
    layer: ffn.MoeFfnSpec,
    *,
    payload_dtype: torch.dtype,
) -> execution.ExpertOperationKind:
    """Select admitted encoding and payload storage without model dispatch.

    Unsupported combinations fail before weight loading or device work. Loading
    normalizes source rows into the selected operation's required order.
    """

    gate_up, down = layer.expert_projection_sources()
    projections = gate_up + down
    if all(isinstance(projection.quantization, Unquantized) for projection in projections):
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("ordinary SGLang Experts require BF16 or FP16 payloads")
        kind = execution.ExpertOperationKind.SGLANG_TRITON
    elif all(isinstance(projection.quantization, Mxfp4) for projection in projections):
        if payload_dtype is not torch.bfloat16:
            raise ValueError("packed Expert execution requires BF16 payloads")
        kind = execution.ExpertOperationKind.TRITON_KERNELS
    else:
        raise ValueError("Expert operation does not support mixed projection encodings")
    return kind


def expert_operation_type(
    kind: execution.ExpertOperationKind,
) -> type[sglang_moe.SglangTritonExperts] | type[triton_moe.TritonKernelsExperts]:
    """Resolve the same concrete operation for resource sizing and capture."""

    match kind:
        case execution.ExpertOperationKind.SGLANG_TRITON:
            return sglang_moe.SglangTritonExperts
        case execution.ExpertOperationKind.TRITON_KERNELS:
            return triton_moe.TritonKernelsExperts
    raise ValueError(f"unsupported Expert operation {kind!r}")
