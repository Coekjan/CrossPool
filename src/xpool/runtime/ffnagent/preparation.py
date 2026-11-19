"""Pure FFN resource preparation shared by loading, capture and admission."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xpool import ffn
from xpool.fabric import InstanceFfnProfile
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import dense, moe, routing
from xpool.runtime.ffnagent.operators.activation import Activation


@dataclass(frozen=True, slots=True)
class DenseFfnLayerLayout:
    """One Dense layer's checked TP source interval and physical resources."""

    layer: ffn.DenseFfnSpec
    hidden_size: int
    payload_dtype: torch.dtype
    activation: Activation
    tp_rank: int
    source_start: int
    source_length: int
    local_intermediate_size: int
    i_proj: weights.ProjectionLayout
    o_proj: weights.ProjectionLayout

    def weight_allocation_bytes(self) -> tuple[int, ...]:
        """Return the ordered physical storage requests for this prepared layer."""

        return self.i_proj.allocation_bytes() + self.o_proj.allocation_bytes()

    def execution_signatures(self, *, profile: InstanceFfnProfile) -> tuple[execution.DenseFfnExecutionSignature, ...]:
        """Specialize this prepared layout over the Instance Capacity domain."""

        capacities = execution.derive_payload_row_capacities(
            max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
        )
        return tuple(
            execution.DenseFfnExecutionSignature(
                payload_dtype=self.payload_dtype,
                payload_row_capacity=capacity,
                hidden_size=self.hidden_size,
                local_intermediate_size=self.local_intermediate_size,
                activation=self.activation,
                i_proj=self.i_proj,
                o_proj=self.o_proj,
            )
            for capacity in capacities
        )


@dataclass(frozen=True, slots=True)
class MoeFfnLayerLayout:
    """One MoE TP role's source interval, physical resources and operation."""

    layer: ffn.MoeFfnSpec
    hidden_size: int
    payload_dtype: torch.dtype
    activation: Activation
    tp_rank: int
    source_start: int
    source_length: int
    local_intermediate_size: int
    i_proj: weights.ProjectionLayout
    o_proj: weights.ProjectionLayout
    router: weights.ProjectionLayout | None
    router_correction_bias: TensorLayout | None

    def weight_allocation_bytes(self) -> tuple[int, ...]:
        """Return ordered Expert and optional Router storage requests, in bytes."""

        result = self.i_proj.allocation_bytes() + self.o_proj.allocation_bytes()
        if self.router is not None:
            result += self.router.allocation_bytes()
        if self.router_correction_bias is not None:
            result += (self.router_correction_bias.nbytes,)
        return result

    def execution_signatures(self, *, profile: InstanceFfnProfile) -> tuple[execution.MoeFfnExecutionSignature, ...]:
        """Specialize this prepared layout over the Instance Capacity domain."""

        capacities = execution.derive_payload_row_capacities(
            max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
        )
        layer = self.layer
        return tuple(
            execution.MoeFfnExecutionSignature(
                payload_dtype=self.payload_dtype,
                payload_row_capacity=capacity,
                hidden_size=self.hidden_size,
                local_intermediate_size=self.local_intermediate_size,
                expert_count=layer.routed_expert_count + layer.shared_expert_count,
                effective_topk=layer.routed_topk + layer.shared_expert_count,
                activation=self.activation,
                routed_scaling_factor=layer.routed_scaling_factor,
                router=execution.MoeRouterExecutionSignature(
                    algorithm=layer.router_algorithm,
                    projection_precision=layer.router_projection_precision,
                    weight_rounding=layer.topk_weight_rounding,
                    projection=self.router,
                    routed_expert_count=layer.routed_expert_count,
                    correction_bias=self.router_correction_bias,
                    renormalize=layer.renormalize,
                )
                if self.router is not None
                else None,
                i_proj=self.i_proj,
                o_proj=self.o_proj,
            )
            for capacity in capacities
        )


type FfnLayerLayout = DenseFfnLayerLayout | MoeFfnLayerLayout


@dataclass(frozen=True, slots=True)
class MoeExecutionPlan:
    """One root backing composed from retained Expert and optional Router plans."""

    experts: moe.ExpertExecutionPlan
    router: routing.RouterExecutionPlan | None
    workspace: execution.WorkspaceLayout

    @classmethod
    def prepare(cls, signature: execution.MoeFfnExecutionSignature) -> MoeExecutionPlan:
        """Prepare both consumers once and retain their exact aligned byte views."""

        experts = moe.ExpertExecutionPlan.prepare(signature)
        router = (
            routing.RouterExecutionPlan.prepare(
                signature.router,
                input_layout=TensorLayout(signature.payload_shape, signature.payload_dtype, 0),
                routed_topk=signature.effective_topk - (signature.expert_count - signature.router.routed_expert_count),
            )
            if signature.router is not None
            else None
        )
        regions = [(torch.uint8, (experts.workspace.nbytes,))]
        if router is not None:
            regions.append((torch.uint8, (router.workspace.nbytes,)))
        return cls(experts=experts, router=router, workspace=execution.WorkspaceLayout.from_regions(tuple(regions)))


def prepare_execution_plans(
    signatures: tuple[execution.ExecutionSignature, ...],
) -> tuple[dense.DenseExecutionPlan | MoeExecutionPlan, ...]:
    """Prepare each Signature's root backing for budgeting and capture."""

    return tuple(
        dense.DenseExecutionPlan.prepare(signature)
        if isinstance(signature, execution.DenseFfnExecutionSignature)
        else MoeExecutionPlan.prepare(signature)
        for signature in signatures
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
        if isinstance(projections, ffn.GatedProjectionSpec):
            gate = model_spec.gate_transform
            i_proj_sources = (projections.gate, projections.up)
        else:
            gate = None
            i_proj_sources = (projections.up,)
        if any(source.quantization != i_proj_sources[0].quantization for source in i_proj_sources):
            raise ValueError("input projection sources require a homogeneous encoding")
        local_width = intermediate_size // tp_size
        projected_width = len(i_proj_sources) * local_width
        i_proj = projections.up.quantization.weight_layout(
            logical_shape=(projected_width, hidden_size), dtype=payload_dtype
        )
        o_proj = projections.down.quantization.weight_layout(
            logical_shape=(hidden_size, local_width), dtype=payload_dtype
        )
        if any((source.scales_key is not None) != (i_proj.scales is not None) for source in i_proj_sources) or (
            (projections.down.scales_key is not None) != (o_proj.scales is not None)
        ):
            raise ValueError("Dense projection scale keys disagree with the encoding")
        return DenseFfnLayerLayout(
            layer=layer,
            hidden_size=hidden_size,
            payload_dtype=payload_dtype,
            activation=Activation(function=model_spec.activation, gate=gate),
            tp_rank=tp_rank,
            source_start=tp_rank * local_width,
            source_length=local_width,
            local_intermediate_size=local_width,
            i_proj=weights.ProjectionLayout(
                quantization=projections.up.quantization,
                gate_up_order=ffn.GateUpOrder.INTERLEAVED if gate is not None else None,
                values=i_proj.values,
                scales=i_proj.scales,
                bias=TensorLayout((projected_width,), torch.float32, 0)
                if any(projection.bias_key is not None for projection in i_proj_sources)
                else None,
            ),
            o_proj=weights.ProjectionLayout(
                quantization=projections.down.quantization,
                gate_up_order=None,
                values=o_proj.values,
                scales=o_proj.scales,
                bias=TensorLayout((hidden_size,), torch.float32, 0) if projections.down.bias_key is not None else None,
            ),
        )

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
    gate_up_sources, down_sources = layer.expert_projection_sources()
    expert_count = layer.routed_expert_count + layer.shared_expert_count
    projections = []
    for sources, shape, bias_shape, order in (
        (
            gate_up_sources,
            (expert_count, 2 * stride, hidden_size),
            (expert_count, 2 * stride),
            ffn.GateUpOrder.INTERLEAVED,
        ),
        (down_sources, (expert_count, hidden_size, stride), (expert_count, hidden_size), None),
    ):
        scheme = sources[0].quantization
        if any(source.quantization != scheme for source in sources):
            raise ValueError("Expert projection group requires a homogeneous encoding")
        encoded = scheme.weight_layout(logical_shape=shape, dtype=payload_dtype)
        if any((source.scales_key is not None) != (encoded.scales is not None) for source in sources):
            raise ValueError("Expert projection scale keys disagree with the encoding")
        projections.append(
            weights.ProjectionLayout(
                quantization=scheme,
                values=encoded.values,
                scales=encoded.scales,
                bias=TensorLayout(bias_shape, torch.float32, 0)
                if any(source.bias_key is not None for source in sources)
                else None,
                gate_up_order=order,
            )
        )
    router_dtype = layer.router_projection_precision.weight_dtype(payload_dtype=payload_dtype)
    encoded_router = layer.router.quantization.weight_layout(
        logical_shape=(layer.routed_expert_count, hidden_size), dtype=router_dtype
    )
    return MoeFfnLayerLayout(
        layer=layer,
        hidden_size=hidden_size,
        payload_dtype=payload_dtype,
        activation=Activation(function=model_spec.activation, gate=model_spec.gate_transform),
        tp_rank=tp_rank,
        source_start=start,
        source_length=length,
        local_intermediate_size=stride,
        i_proj=projections[0],
        o_proj=projections[1],
        router=weights.ProjectionLayout(
            quantization=layer.router.quantization,
            gate_up_order=None,
            values=encoded_router.values,
            scales=encoded_router.scales,
            bias=TensorLayout((layer.routed_expert_count,), torch.float32, 0)
            if layer.router.bias_key is not None
            else None,
        )
        if tp_rank == 0
        else None,
        router_correction_bias=TensorLayout((layer.routed_expert_count,), torch.float32, 0)
        if tp_rank == 0 and layer.router_correction_bias_key is not None
        else None,
    )
