"""Private FFN execution identities and exact resource geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from xpool import ffn
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent.operators.activation import Activation
from xpool.runtime.ffnagent.weights import ProjectionLayout
from xpool.utils import align_up

OPERATOR_ALIGNMENT_BYTES = 16


@dataclass(frozen=True, slots=True)
class CapturePlaceholderLayout:
    """Default-domain capture Tensor geometry, owning no allocated storage."""

    hidden_states: TensorLayout
    partial: TensorLayout
    payload_rows: TensorLayout | None
    routing: TensorLayout | None

    def allocation_bytes(self) -> tuple[int, ...]:
        """Return independent requests in capture allocation order."""

        return tuple(
            layout.nbytes
            for layout in (self.hidden_states, self.partial, self.payload_rows, self.routing)
            if layout is not None
        )


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Prepared byte regions borrowed from one caller-owned backing."""

    regions: tuple[tuple[torch.dtype, tuple[int, ...]], ...]
    offsets: tuple[int, ...]
    nbytes: int

    @classmethod
    def from_regions(cls, regions: tuple[tuple[torch.dtype, tuple[int, ...]], ...]) -> WorkspaceLayout:
        """Prepare aligned regions once for admission and view construction."""

        offsets = []
        cursor = 0
        for dtype, shape in regions:
            size = math.prod(shape) * dtype.itemsize
            if size:
                cursor = align_up(cursor, OPERATOR_ALIGNMENT_BYTES)
            offsets.append(cursor)
            cursor += size
        return cls(regions=regions, offsets=tuple(offsets), nbytes=cursor)

    def views(self, backing: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Borrow the prepared views without allocating or choosing new geometry."""

        if backing.dtype is not torch.uint8 or backing.ndim != 1 or not backing.is_contiguous():
            raise ValueError("workspace backing must be a contiguous byte vector")
        if backing.numel() < self.nbytes or backing.data_ptr() % OPERATOR_ALIGNMENT_BYTES:
            raise ValueError("workspace backing does not cover its prepared aligned extent")
        return tuple(
            backing[offset : offset + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)
            for offset, (dtype, shape) in zip(self.offsets, self.regions, strict=True)
        )


@dataclass(frozen=True, slots=True)
class DenseFfnExecutionSignature:
    """Complete reusable Dense Graph identity."""

    payload_dtype: torch.dtype
    payload_row_capacity: int
    hidden_size: int
    local_intermediate_size: int
    activation: Activation
    i_proj: ProjectionLayout
    o_proj: ProjectionLayout

    @property
    def payload_shape(self) -> tuple[int, int]:
        """Return the shared hidden-state and partial-output geometry."""

        return self.payload_row_capacity, self.hidden_size

    def capture_placeholder_layout(self) -> CapturePlaceholderLayout:
        """Describe input and partial-output storages shared by Primary/Control."""

        payload = TensorLayout(self.payload_shape, self.payload_dtype, 0)
        return CapturePlaceholderLayout(hidden_states=payload, partial=payload, payload_rows=None, routing=None)


@dataclass(frozen=True, slots=True)
class MoeRouterExecutionSignature:
    """Router-owner Graph implementation and resource identity."""

    algorithm: ffn.RouterAlgorithm
    projection_precision: ffn.RouterProjectionPrecision
    weight_rounding: ffn.TopKWeightRounding
    projection: ProjectionLayout
    routed_expert_count: int
    correction_bias: TensorLayout | None
    renormalize: bool


@dataclass(frozen=True, slots=True)
class MoeFfnExecutionSignature:
    """Complete reusable MoE Graph identity for one TP role."""

    payload_dtype: torch.dtype
    payload_row_capacity: int
    hidden_size: int
    local_intermediate_size: int
    expert_count: int
    effective_topk: int
    activation: Activation
    routed_scaling_factor: float
    router: MoeRouterExecutionSignature | None
    i_proj: ProjectionLayout
    o_proj: ProjectionLayout

    @property
    def payload_shape(self) -> tuple[int, int]:
        """Return the shared hidden-state and partial-output geometry."""

        return self.payload_row_capacity, self.hidden_size

    @property
    def routing_shape(self) -> tuple[int, int]:
        """Return route IDs/weights geometry for this TP role."""

        return self.payload_row_capacity, self.effective_topk

    @property
    def routing_storage_bytes(self) -> int:
        """Return the packed int32 ID and float32 weight storage size."""

        return math.prod(self.routing_shape) * (torch.int32.itemsize + torch.float32.itemsize)

    def capture_placeholder_layout(self) -> CapturePlaceholderLayout:
        """Describe payload, optional live rows and packed ID/weight storages."""

        payload = TensorLayout(self.payload_shape, self.payload_dtype, 0)
        return CapturePlaceholderLayout(
            hidden_states=payload,
            partial=payload,
            payload_rows=TensorLayout((1,), torch.int64, 0) if self.router is not None else None,
            routing=TensorLayout((self.routing_storage_bytes,), torch.uint8, 0),
        )


type ExecutionSignature = DenseFfnExecutionSignature | MoeFfnExecutionSignature


def capture_weight_pair_key(signature: ExecutionSignature) -> ExecutionSignature:
    """Normalize Capacity for shape-equivalent Probe sharing."""

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


def control_capture_probe_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return physical byte requests of every Control Capture Probe storage."""

    if isinstance(signature, DenseFfnExecutionSignature):
        return signature.i_proj.allocation_bytes() + signature.o_proj.allocation_bytes()
    result = signature.i_proj.allocation_bytes() + signature.o_proj.allocation_bytes()
    if signature.router is not None:
        result += signature.router.projection.allocation_bytes()
        correction = signature.router.correction_bias
        if correction is not None:
            result += (correction.nbytes,)
    return result


def graph_capture_initialization_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return transient route-ID arange and remainder requests during construction."""

    if isinstance(signature, DenseFfnExecutionSignature):
        return ()
    return (4 * signature.effective_topk,) * 2
