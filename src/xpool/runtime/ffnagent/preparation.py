"""Pure FFN resource preparation shared by loading, capture and admission."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xpool import ffn
from xpool.fabric import InstanceFfnProfile
from xpool.quantization import TensorLayout, Unquantized
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import moe
from xpool.utils import align_up


@dataclass(frozen=True, slots=True)
class DenseFfnLayerLayout:
    """One Dense layer's checked TP source interval and physical resources."""

    layer: ffn.DenseFfnSpec
    hidden_size: int
    payload_dtype: torch.dtype
    activation: ffn.ActivationSpec
    tp_size: int
    tp_rank: int
    source_start: int
    source_length: int
    local_intermediate_size: int
    gate_up: weights.ProjectionLayout
    down: weights.ProjectionLayout

    def weight_allocation_bytes(self) -> tuple[int, ...]:
        """Return the ordered physical storage requests for this prepared layer."""

        return self.gate_up.allocation_bytes() + self.down.allocation_bytes()


@dataclass(frozen=True, slots=True)
class MoeFfnLayerLayout:
    """One MoE TP role's source interval, physical resources and operation."""

    layer: ffn.MoeFfnSpec
    hidden_size: int
    payload_dtype: torch.dtype
    activation: ffn.SiluActivation
    tp_size: int
    tp_rank: int
    source_start: int
    source_length: int
    local_intermediate_size: int
    experts: weights.ExpertLayout
    router: weights.ProjectionLayout | None
    router_correction_bias: TensorLayout | None
    operation: execution.ExpertOperationKind

    def weight_allocation_bytes(self) -> tuple[int, ...]:
        """Return ordered Expert and optional Router storage requests, in bytes."""

        result = self.experts.gate_up.allocation_bytes() + self.experts.down.allocation_bytes()
        if self.router is not None:
            result += self.router.allocation_bytes()
        if self.router_correction_bias is not None:
            result += (math.prod(self.router_correction_bias.shape) * self.router_correction_bias.dtype.itemsize,)
        return result


type FfnLayerLayout = DenseFfnLayerLayout | MoeFfnLayerLayout


@dataclass(frozen=True, slots=True)
class RouterWorkspaceLayout:
    """Aligned byte regions in one caller-owned MoE root workspace."""

    projection: slice
    ids: slice
    weights: slice


@dataclass(frozen=True, slots=True)
class MoeWorkspaceLayout:
    """Compose private Expert scratch and common Router storage without allocation."""

    experts: slice
    router: RouterWorkspaceLayout | None
    nbytes: int

    @classmethod
    def from_signature(cls, signature: execution.MoeFfnExecutionSignature) -> MoeWorkspaceLayout:
        operation_type = moe.expert_operation_type(signature.operation)
        expert_bytes = operation_type.workspace_layout(signature)[2]
        experts = slice(0, expert_bytes)
        router = signature.router
        if router is None:
            return cls(experts=experts, router=None, nbytes=expert_bytes)
        routed_topk = signature.effective_topk - (signature.expert_count - router.routed_expert_count)
        if routed_topk <= 0:
            raise ValueError("Router-owner Execution Signature has invalid routed TopK")
        route_count = signature.payload_row_capacity * routed_topk
        cursor = align_up(expert_bytes, execution.OPERATOR_ALIGNMENT_BYTES)
        projection = slice(cursor, cursor + router.router_workspace_bytes)
        cursor = align_up(cursor + router.router_workspace_bytes, execution.OPERATOR_ALIGNMENT_BYTES)
        ids = slice(cursor, cursor + route_count * torch.int32.itemsize)
        cursor = align_up(cursor + route_count * torch.int32.itemsize, execution.OPERATOR_ALIGNMENT_BYTES)
        probabilities = slice(cursor, cursor + route_count * torch.float32.itemsize)
        return cls(
            experts=experts,
            router=RouterWorkspaceLayout(projection=projection, ids=ids, weights=probabilities),
            nbytes=cursor + route_count * torch.float32.itemsize,
        )


def prepare_ffn_layer(
    model_spec: ffn.FfnModelSpec,
    *,
    layer_ordinal: int,
    payload_dtype: torch.dtype,
    tp_size: int,
    tp_rank: int,
) -> FfnLayerLayout:
    """Resolve source ownership and final storage without I/O or device work.

    Encoding alignment determines a common bounded TP stride. It can exceed
    the mathematical width divided by TP; an empty final source interval still
    receives a positive, neutrally padded execution width. Unsupported
    operation/encoding combinations fail before loading or capture.
    """

    if any(isinstance(value, bool) or not isinstance(value, int) for value in (layer_ordinal, tp_size, tp_rank)):
        raise ValueError("FFN preparation coordinates must be integers")
    if not 0 <= layer_ordinal < len(model_spec.layers) or tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("FFN preparation coordinate is outside the model or TP domain")
    if payload_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FFN preparation requires BF16 or FP16 payloads")
    layer = model_spec.layers[layer_ordinal]
    hidden_size = model_spec.hidden_size
    intermediate_size = (
        layer.intermediate_size if isinstance(layer, ffn.DenseFfnSpec) else layer.expert_intermediate_size
    )
    if intermediate_size % tp_size:
        raise ValueError("FFN intermediate size is not divisible by FFN TP")

    if isinstance(layer, ffn.DenseFfnSpec):
        projections = layer.projections
        local_width = intermediate_size // tp_size
        if isinstance(projections, ffn.UnaryProjectionSpec):
            if any(
                not isinstance(projection.quantization, Unquantized) or projection.scales_key is not None
                for projection in (projections.fc1, projections.fc2)
            ):
                raise ValueError("Unary Dense execution requires unquantized projections without scales")
            if projections.fc1.bias_key is None or projections.fc2.bias_key is None:
                raise ValueError("Unary Dense execution requires fc1 and fc2 biases")
            fc1 = projections.fc1.quantization.weight_layout(
                logical_shape=(local_width, hidden_size), dtype=payload_dtype
            )
            fc2 = projections.fc2.quantization.weight_layout(
                logical_shape=(hidden_size, local_width), dtype=payload_dtype
            )
            return DenseFfnLayerLayout(
                layer=layer,
                hidden_size=hidden_size,
                payload_dtype=payload_dtype,
                activation=model_spec.activation,
                tp_size=tp_size,
                tp_rank=tp_rank,
                source_start=tp_rank * local_width,
                source_length=local_width,
                local_intermediate_size=local_width,
                gate_up=weights.ProjectionLayout(
                    values=fc1.values,
                    scales=fc1.scales,
                    bias=TensorLayout((local_width,), payload_dtype, 0),
                ),
                down=weights.ProjectionLayout(
                    values=fc2.values,
                    scales=fc2.scales,
                    bias=TensorLayout((hidden_size,), payload_dtype, 0),
                ),
            )
        if any(
            not isinstance(projection.quantization, Unquantized) or projection.scales_key is not None
            for projection in (projections.gate, projections.up, projections.down)
        ):
            raise ValueError("Dense execution requires unquantized projections without scales")
        gate_up = projections.gate.quantization.weight_layout(
            logical_shape=(2 * local_width, hidden_size), dtype=payload_dtype
        )
        down = projections.down.quantization.weight_layout(
            logical_shape=(hidden_size, local_width), dtype=payload_dtype
        )
        return DenseFfnLayerLayout(
            layer=layer,
            hidden_size=hidden_size,
            payload_dtype=payload_dtype,
            activation=model_spec.activation,
            tp_size=tp_size,
            tp_rank=tp_rank,
            source_start=tp_rank * local_width,
            source_length=local_width,
            local_intermediate_size=local_width,
            gate_up=weights.ProjectionLayout(
                values=gate_up.values,
                scales=gate_up.scales,
                bias=TensorLayout((2 * local_width,), payload_dtype, 0)
                if projections.gate.bias_key is not None or projections.up.bias_key is not None
                else None,
            ),
            down=weights.ProjectionLayout(
                values=down.values,
                scales=down.scales,
                bias=TensorLayout((hidden_size,), payload_dtype, 0) if projections.down.bias_key is not None else None,
            ),
        )

    if not isinstance(model_spec.activation, ffn.SiluActivation):
        raise ValueError("MoE FFN execution requires SiLU activation")
    operation = moe.select_expert_operation(layer, payload_dtype=payload_dtype)
    operation_type = moe.expert_operation_type(operation)
    source = layer.routed_experts
    if isinstance(source, ffn.StackedExpertSource):
        source_axes = (
            (source.gate_up, (layer.routed_expert_count, 2 * intermediate_size, hidden_size), 1),
            (source.down, (layer.routed_expert_count, hidden_size, intermediate_size), 2),
        )
    else:
        source_axes = tuple(
            entry
            for expert in source.experts
            for entry in (
                (expert.gate, (intermediate_size, hidden_size), 0),
                (expert.up, (intermediate_size, hidden_size), 0),
                (expert.down, (hidden_size, intermediate_size), 1),
            )
        )
    if layer.shared_expert is not None:
        shared = layer.shared_expert
        shared_width = intermediate_size * layer.shared_expert_count
        source_axes += (
            (shared.gate, (shared_width, hidden_size), 0),
            (shared.up, (shared_width, hidden_size), 0),
            (shared.down, (hidden_size, shared_width), 1),
        )
    quantum = math.lcm(
        *(
            projection.quantization.slice_alignment(logical_shape=shape, axis=axis)
            for projection, shape, axis in source_axes
        )
    )
    stride = (intermediate_size + tp_size * quantum - 1) // (tp_size * quantum) * quantum
    start = min(tp_rank * stride, intermediate_size)
    length = min((tp_rank + 1) * stride, intermediate_size) - start
    experts = weights.ExpertLayout.from_layer(
        layer,
        hidden_size=hidden_size,
        local_intermediate_size=stride,
        payload_dtype=payload_dtype,
        bias_dtype=operation_type.bias_dtype,
    )
    if not isinstance(layer.router.quantization, Unquantized) or layer.router.scales_key is not None:
        raise ValueError("Router execution requires an unquantized projection without scales")
    router_dtype = layer.router_projection_precision.weight_dtype(payload_dtype=payload_dtype)
    encoded_router = layer.router.quantization.weight_layout(
        logical_shape=(layer.routed_expert_count, hidden_size), dtype=router_dtype
    )
    return MoeFfnLayerLayout(
        layer=layer,
        hidden_size=hidden_size,
        payload_dtype=payload_dtype,
        activation=model_spec.activation,
        tp_size=tp_size,
        tp_rank=tp_rank,
        source_start=start,
        source_length=length,
        local_intermediate_size=stride,
        experts=experts,
        router=weights.ProjectionLayout(
            values=encoded_router.values,
            scales=encoded_router.scales,
            bias=TensorLayout(
                (layer.routed_expert_count,),
                payload_dtype
                if layer.router_projection_precision is ffn.RouterProjectionPrecision.PAYLOAD
                else torch.float32,
                0,
            )
            if layer.router.bias_key is not None
            else None,
        )
        if tp_rank == 0
        else None,
        router_correction_bias=TensorLayout((layer.routed_expert_count,), torch.float32, 0)
        if tp_rank == 0 and layer.router_correction_bias_key is not None
        else None,
        operation=operation,
    )


def required_execution_signatures(
    *,
    layer_layout: FfnLayerLayout,
    profile: InstanceFfnProfile,
) -> tuple[execution.ExecutionSignature, ...]:
    """Specialize prepared resources for the existing complete Capacity domain."""

    capacities = execution.derive_payload_row_capacities(
        max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
    )
    if isinstance(layer_layout, DenseFfnLayerLayout):
        return tuple(
            execution.DenseFfnExecutionSignature(
                payload_dtype=layer_layout.payload_dtype,
                payload_row_capacity=capacity,
                hidden_size=layer_layout.hidden_size,
                local_intermediate_size=layer_layout.local_intermediate_size,
                activation=layer_layout.activation,
                gate_up=layer_layout.gate_up,
                down=layer_layout.down,
            )
            for capacity in capacities
        )
    layer = layer_layout.layer
    return tuple(
        execution.MoeFfnExecutionSignature(
            payload_dtype=layer_layout.payload_dtype,
            payload_row_capacity=capacity,
            hidden_size=layer_layout.hidden_size,
            local_intermediate_size=layer_layout.local_intermediate_size,
            expert_count=layer.routed_expert_count + layer.shared_expert_count,
            effective_topk=layer.routed_topk + layer.shared_expert_count,
            activation=layer_layout.activation,
            routed_scaling_factor=layer.routed_scaling_factor,
            router=execution.MoeRouterExecutionSignature(
                algorithm=layer.router_algorithm,
                projection_precision=layer.router_projection_precision,
                weight_rounding=layer.topk_weight_rounding,
                projection=layer_layout.router,
                routed_expert_count=layer.routed_expert_count,
                router_workspace_bytes=execution.router_workspace_bytes(
                    layer.router_projection_precision,
                    payload_dtype=layer_layout.payload_dtype,
                    payload_row_capacity=capacity,
                    hidden_size=layer_layout.hidden_size,
                    routed_expert_count=layer.routed_expert_count,
                    routed_topk=layer.routed_topk,
                ),
                correction_bias=layer_layout.router_correction_bias,
                renormalize=layer.renormalize,
            )
            if layer_layout.router is not None
            else None,
            operation=layer_layout.operation,
            gate_up=layer_layout.experts.gate_up,
            down=layer_layout.experts.down,
        )
        for capacity in capacities
    )


def compute_workspace_bytes(signature: execution.ExecutionSignature) -> int:
    """Size the selected computation without configuration selection or allocation."""

    if isinstance(signature, execution.DenseFfnExecutionSignature):
        return execution.dense_workspace_bytes(
            payload_dtype=signature.payload_dtype,
            row_capacity=signature.payload_row_capacity,
            local_intermediate_size=signature.local_intermediate_size,
            activation=signature.activation,
        )
    return MoeWorkspaceLayout.from_signature(signature).nbytes


def graph_capture_capacity_storage_bytes(signature: execution.ExecutionSignature) -> tuple[int, ...]:
    """Return Capacity-dependent capture requests in their allocation order."""

    payload_bytes = signature.payload_dtype.itemsize * signature.payload_row_capacity * signature.hidden_size
    if isinstance(signature, execution.DenseFfnExecutionSignature):
        return payload_bytes, payload_bytes, compute_workspace_bytes(signature)
    result = (
        payload_bytes,
        payload_bytes,
        8 * signature.payload_row_capacity * signature.effective_topk,
        compute_workspace_bytes(signature),
    )
    return (*result, 8) if signature.router is not None else result
