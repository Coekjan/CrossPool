"""Private FFN execution identities and exact resource geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from xpool import ffn
from xpool.fabric import InstanceFfnProfile
from xpool.runtime.ffnagent import architecture, weights
from xpool.utils import align_up

OPERATOR_ALIGNMENT_BYTES = 16
QUALIFIED_MOE_BLOCK_SIZE_M_VALUES = (16, 32, 64, 128)


class MoeRoutingOperation(Protocol):
    """Normalized model-owned routing operation used during Graph capture."""

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        router_weights: weights.MoeRouterWeights,
        workspace: torch.Tensor,
        routed_ids: torch.Tensor,
        routed_weights: torch.Tensor,
        renormalize: bool,
    ) -> None:
        """Populate routed Expert IDs and weights for one input payload."""


@dataclass(frozen=True, slots=True)
class DenseFfnExecutionSignature:
    """Complete reusable gated-Dense Graph identity."""

    payload_dtype: torch.dtype
    payload_row_capacity: int
    hidden_size: int
    local_intermediate_size: int
    activation: ffn.ActivationKind

    def __post_init__(self) -> None:
        """Reject unsupported or empty Dense execution geometry."""

        if self.payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Dense FFN execution requires BF16 or FP16 payloads")
        if min(self.payload_row_capacity, self.hidden_size, self.local_intermediate_size) <= 0:
            raise ValueError("Dense FFN execution dimensions must be positive")
        if self.activation is not ffn.ActivationKind.SILU:
            raise ValueError("Dense FFN execution supports only SiLU activation")


@dataclass(frozen=True, slots=True)
class MoeRouterExecutionSignature:
    """Router-owner Graph implementation and resource identity."""

    compute_routed_topk: MoeRoutingOperation
    router_weight_dtype: torch.dtype
    routed_expert_count: int
    router_workspace_bytes: int
    correction_bias_present: bool
    renormalize: bool

    def __post_init__(self) -> None:
        """Reject invalid Router semantics and resource geometry."""

        if not callable(self.compute_routed_topk):
            raise ValueError("MoE Router execution requires one callable implementation")
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
    activation: ffn.ActivationKind
    routed_scaling_factor: float
    router: MoeRouterExecutionSignature | None

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
        if self.activation is not ffn.ActivationKind.SILU:
            raise ValueError("MoE FFN execution supports only SiLU activation")
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
    if signature.router is not None:
        shared_expert_count = signature.expert_count - signature.router.routed_expert_count
        routed_topk = signature.effective_topk - shared_expert_count
        if routed_topk <= 0:
            raise ValueError("Router-owner Execution Signature has invalid routed TopK")
        region_specs.extend(
            (
                (torch.uint8, (signature.router.router_workspace_bytes,)),
                (torch.int32, (signature.payload_row_capacity, routed_topk)),
                (torch.float32, (signature.payload_row_capacity, routed_topk)),
            )
        )

    offsets = []
    cursor = 0
    for dtype, shape in region_specs:
        cursor = align_up(cursor, OPERATOR_ALIGNMENT_BYTES)
        offsets.append(cursor)
        cursor += math.prod(shape) * dtype.itemsize
    return tuple(region_specs), tuple(offsets), cursor


def compute_workspace_bytes(signature: ExecutionSignature) -> int:
    """Return the selector-independent allocated Compute Workspace extent."""

    if isinstance(signature, DenseFfnExecutionSignature):
        return dense_workspace_bytes(
            payload_dtype=signature.payload_dtype,
            row_capacity=signature.payload_row_capacity,
            local_intermediate_size=signature.local_intermediate_size,
        )
    return max(moe_workspace_layout(signature, block_size_m=value)[2] for value in QUALIFIED_MOE_BLOCK_SIZE_M_VALUES)


def control_capture_probe_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return logical bytes of every Control Capture Probe storage."""

    gate_up_bytes = 2 * signature.payload_dtype.itemsize * signature.hidden_size * signature.local_intermediate_size
    down_bytes = signature.payload_dtype.itemsize * signature.hidden_size * signature.local_intermediate_size
    if isinstance(signature, DenseFfnExecutionSignature):
        return gate_up_bytes, down_bytes
    result = [signature.expert_count * gate_up_bytes, signature.expert_count * down_bytes]
    if signature.router is not None:
        result.append(
            signature.router.router_weight_dtype.itemsize * signature.router.routed_expert_count * signature.hidden_size
        )
        if signature.router.correction_bias_present:
            result.append(4 * signature.router.routed_expert_count)
    return tuple(result)


def graph_capture_capacity_storage_bytes(signature: ExecutionSignature) -> tuple[int, ...]:
    """Return logical bytes of every Capacity-dependent Graph Capture storage."""

    payload_bytes = signature.payload_dtype.itemsize * signature.payload_row_capacity * signature.hidden_size
    if isinstance(signature, DenseFfnExecutionSignature):
        return payload_bytes, payload_bytes, compute_workspace_bytes(signature)
    result = [
        payload_bytes,
        payload_bytes,
        8 * signature.payload_row_capacity * signature.effective_topk,
        compute_workspace_bytes(signature),
    ]
    if signature.router is not None:
        result.append(8)
    return tuple(result)


def required_execution_signatures(
    *,
    model_spec: ffn.FfnModelSpec,
    profile: InstanceFfnProfile,
    layer_ordinal: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[ExecutionSignature, ...]:
    """Derive every Capacity-specialized Signature for one prospective TP rank."""

    if not 0 <= layer_ordinal < len(model_spec.layers) or not 0 <= tp_rank < tp_size:
        raise ValueError("FFN Signature coordinate is outside the model or TP domain")
    layer = model_spec.layers[layer_ordinal]
    capacities = derive_payload_row_capacities(
        max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
    )
    if isinstance(layer, ffn.DenseFfnSpec):
        if layer.intermediate_size % tp_size:
            raise ValueError("Dense intermediate size is not divisible by FFN TP")
        local_intermediate_size = layer.intermediate_size // tp_size
        return tuple(
            DenseFfnExecutionSignature(
                payload_dtype=profile.payload_dtype,
                payload_row_capacity=capacity,
                hidden_size=model_spec.hidden_size,
                local_intermediate_size=local_intermediate_size,
                activation=model_spec.activation,
            )
            for capacity in capacities
        )

    if layer.expert_intermediate_size % tp_size:
        raise ValueError("MoE Expert intermediate size is not divisible by FFN TP")
    model_adapter = architecture.adapter_for(model_spec)
    if not issubclass(model_adapter, architecture.MoeFfnModelAdapter):
        raise ValueError("MoE Model Spec requires a MoE FFN Model Adapter")
    return tuple(
        MoeFfnExecutionSignature(
            payload_dtype=profile.payload_dtype,
            payload_row_capacity=capacity,
            hidden_size=model_spec.hidden_size,
            local_intermediate_size=layer.expert_intermediate_size // tp_size,
            expert_count=layer.routed_expert_count + layer.shared_expert_count,
            effective_topk=layer.routed_topk + layer.shared_expert_count,
            activation=model_spec.activation,
            routed_scaling_factor=layer.routed_scaling_factor,
            router=(
                MoeRouterExecutionSignature(
                    compute_routed_topk=model_adapter.compute_routed_topk,
                    router_weight_dtype=model_adapter.router_weight_dtype(payload_dtype=profile.payload_dtype),
                    routed_expert_count=layer.routed_expert_count,
                    router_workspace_bytes=model_adapter.router_workspace_bytes(
                        payload_dtype=profile.payload_dtype,
                        payload_row_capacity=capacity,
                        hidden_size=model_spec.hidden_size,
                        routed_expert_count=layer.routed_expert_count,
                        routed_topk=layer.routed_topk,
                    ),
                    correction_bias_present=(layer.checkpoint.router_correction_bias_key is not None),
                    renormalize=layer.renormalize,
                )
                if tp_rank == 0
                else None
            ),
        )
        for capacity in capacities
    )
