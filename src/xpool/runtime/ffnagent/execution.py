"""Private FFN execution identities and exact resource geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum

import torch

from xpool import ffn
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent.weights import ProjectionLayout
from xpool.utils import align_up

OPERATOR_ALIGNMENT_BYTES = 16
QUALIFIED_MOE_BLOCK_SIZE_M_VALUES = (16, 32, 64, 128)


class ExpertOperationKind(IntEnum):
    """Concrete grouped computation captured into a reusable Graph."""

    SGLANG_TRITON = 1
    TRITON_KERNELS = 2


def dense_workspace_bytes(
    *,
    payload_dtype: torch.dtype,
    row_capacity: int,
    local_intermediate_size: int,
) -> int:
    """Return gate/up plus activated-up workspace bytes for one Dense branch."""

    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Dense workspace requires BF16 or FP16 payloads")
    if row_capacity <= 0 or local_intermediate_size <= 0:
        raise ValueError("Dense workspace dimensions must be positive")
    return row_capacity * 3 * local_intermediate_size * payload_dtype.itemsize


def moe_alignment_workspace_shapes(
    *,
    row_capacity: int,
    effective_topk: int,
    expert_count: int,
    block_size_m: int,
) -> tuple[int, int, int]:
    """Return exact sorted-id, block-id, and cumsum element counts."""

    if row_capacity <= 0 or effective_topk <= 0 or expert_count <= 0:
        raise ValueError("MoE alignment dimensions must be positive")
    if block_size_m not in QUALIFIED_MOE_BLOCK_SIZE_M_VALUES:
        raise ValueError(f"MoE BLOCK_SIZE_M {block_size_m} is outside the qualified domain")
    route_count = row_capacity * effective_topk
    maximum_padded = route_count + (expert_count + 1) * (block_size_m - 1)
    expert_block_count = (maximum_padded + block_size_m - 1) // block_size_m
    return maximum_padded, expert_block_count, expert_count + 2


@dataclass(frozen=True, slots=True)
class DenseFfnExecutionSignature:
    """Complete reusable gated-Dense Graph identity."""

    payload_dtype: torch.dtype
    payload_row_capacity: int
    hidden_size: int
    local_intermediate_size: int
    activation: ffn.ActivationSpec
    gate_up: ProjectionLayout
    down: ProjectionLayout

    def __post_init__(self) -> None:
        """Reject unsupported or empty Dense execution geometry."""

        if self.payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dense FFN execution requires BF16 or FP16 payloads")
        if min(self.payload_row_capacity, self.hidden_size, self.local_intermediate_size) <= 0:
            raise ValueError("Dense FFN execution dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MoeRouterExecutionSignature:
    """Router-owner Graph implementation and resource identity."""

    algorithm: ffn.RouterAlgorithm
    projection_precision: ffn.RouterProjectionPrecision
    weight_rounding: ffn.TopKWeightRounding
    projection: ProjectionLayout
    routed_expert_count: int
    router_workspace_bytes: int
    correction_bias: TensorLayout | None
    renormalize: bool

    def __post_init__(self) -> None:
        """Reject invalid Router semantics and resource geometry."""

        if not isinstance(self.algorithm, ffn.RouterAlgorithm) or not isinstance(
            self.projection_precision, ffn.RouterProjectionPrecision
        ):
            raise ValueError("MoE Router execution requires a supported algorithm and precision")
        if not isinstance(self.weight_rounding, ffn.TopKWeightRounding):
            raise ValueError("MoE Router execution requires a supported TopK weight rounding policy")
        if self.routed_expert_count <= 0 or self.router_workspace_bytes < 0:
            raise ValueError("MoE Router execution geometry is inconsistent")


@dataclass(frozen=True, slots=True)
class MoeFfnExecutionSignature:
    """Complete reusable MoE Graph identity for one TP role."""

    payload_dtype: torch.dtype
    payload_row_capacity: int
    hidden_size: int
    local_intermediate_size: int
    expert_count: int
    effective_topk: int
    activation: ffn.ActivationSpec
    routed_scaling_factor: float
    router: MoeRouterExecutionSignature | None
    operation: ExpertOperationKind
    gate_up: ProjectionLayout
    down: ProjectionLayout

    def __post_init__(self) -> None:
        """Reject unsupported or empty MoE execution geometry."""

        if self.payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("MoE FFN execution requires BF16 or FP16 payloads")
        if (
            min(
                self.payload_row_capacity,
                self.hidden_size,
                self.local_intermediate_size,
                self.expert_count,
                self.effective_topk,
            )
            <= 0
        ):
            raise ValueError("MoE FFN execution dimensions must be positive")
        if self.effective_topk > self.expert_count:
            raise ValueError("MoE effective TopK exceeds Expert count")
        if not isinstance(self.operation, ExpertOperationKind):
            raise ValueError("MoE execution requires a concrete Expert operation")
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("MoE routed scaling factor must be finite and positive")


type ExecutionSignature = DenseFfnExecutionSignature | MoeFfnExecutionSignature


def capture_weight_pair_key(signature: ExecutionSignature) -> ExecutionSignature:
    """Normalize Capacity and its derived Router extent for Probe sharing."""

    if isinstance(signature, MoeFfnExecutionSignature) and signature.router is not None:
        return replace(
            signature,
            payload_row_capacity=1,
            router=replace(signature.router, router_workspace_bytes=0),
        )
    return replace(signature, payload_row_capacity=1)


def derive_payload_row_capacities(maximum_row_capacity: int) -> tuple[int, ...]:
    """Return increasing power-of-two Graph capacities through one exact maximum."""

    if maximum_row_capacity <= 0:
        raise ValueError("maximum payload row capacity must be positive")
    capacities = []
    capacity = 1
    while capacity < maximum_row_capacity:
        capacities.append(capacity)
        capacity *= 2
    capacities.append(maximum_row_capacity)
    return tuple(capacities)


def router_workspace_bytes(
    projection_precision: ffn.RouterProjectionPrecision,
    *,
    payload_dtype: torch.dtype,
    payload_row_capacity: int,
    hidden_size: int,
    routed_expert_count: int,
    routed_topk: int,
) -> int:
    """Size caller-owned logits and any FP32 input conversion in bytes."""

    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Router requires BF16 or FP16 payloads")
    if (
        min(payload_row_capacity, hidden_size, routed_expert_count, routed_topk) <= 0
        or routed_topk > routed_expert_count
    ):
        raise ValueError("Router dimensions are inconsistent")
    match projection_precision:
        case ffn.RouterProjectionPrecision.PAYLOAD:
            return payload_row_capacity * routed_expert_count * payload_dtype.itemsize
        case ffn.RouterProjectionPrecision.FP32_LOGITS:
            return payload_row_capacity * routed_expert_count * torch.float32.itemsize
        case ffn.RouterProjectionPrecision.FP32:
            return payload_row_capacity * (hidden_size + routed_expert_count) * torch.float32.itemsize
    raise ValueError(f"unsupported Router projection precision {projection_precision!r}")


def moe_workspace_layout(
    signature: MoeFfnExecutionSignature,
    *,
    block_size_m: int,
) -> tuple[tuple[tuple[torch.dtype, tuple[int, ...]], ...], tuple[int, ...], int]:
    """Return selected MoE workspace regions, byte offsets, and final extent."""

    maximum_padded, expert_block_count, cumsum_count = moe_alignment_workspace_shapes(
        row_capacity=signature.payload_row_capacity,
        effective_topk=signature.effective_topk,
        expert_count=signature.expert_count,
        block_size_m=block_size_m,
    )
    route_count = signature.payload_row_capacity * signature.effective_topk
    region_specs: list[tuple[torch.dtype, tuple[int, ...]]] = [
        (torch.int32, (maximum_padded,)),
        (torch.int32, (expert_block_count,)),
        (torch.int32, (1,)),
        (torch.int32, (cumsum_count,)),
        (signature.payload_dtype, (route_count, signature.local_intermediate_size)),
        (
            torch.uint8,
            (
                max(
                    route_count * 2 * signature.local_intermediate_size * signature.payload_dtype.itemsize,
                    route_count * signature.hidden_size * signature.payload_dtype.itemsize,
                ),
            ),
        ),
    ]
    offsets = []
    cursor = 0
    for dtype, shape in region_specs:
        cursor = align_up(cursor, OPERATOR_ALIGNMENT_BYTES)
        offsets.append(cursor)
        cursor += math.prod(shape) * dtype.itemsize
    return tuple(region_specs), tuple(offsets), cursor


def control_capture_probe_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return physical byte requests of every Control Capture Probe storage."""

    result = signature.gate_up.allocation_bytes() + signature.down.allocation_bytes()
    if isinstance(signature, MoeFfnExecutionSignature) and signature.router is not None:
        result += signature.router.projection.allocation_bytes()
        correction = signature.router.correction_bias
        if correction is not None:
            result += (math.prod(correction.shape) * correction.dtype.itemsize,)
    return result


def graph_capture_initialization_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return transient route-ID arange and remainder requests during construction."""

    if isinstance(signature, DenseFfnExecutionSignature):
        return ()
    return (4 * signature.effective_topk,) * 2
