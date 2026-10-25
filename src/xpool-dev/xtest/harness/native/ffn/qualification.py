"""Deterministic FFN structures and independent numerical references."""

from __future__ import annotations

import torch
import torch.nn.functional

from xpool import ffn
from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricGenerationId,
    FabricInstancePlan,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnModelPlan,
    FifoSchedulerPolicy,
    InstanceFfnLayerProfile,
    InstanceFfnProfile,
    InstanceRankTopology,
    MoeFfnLayerPlan,
)
from xpool.model import ModelId
from xpool.native.ffn import LayerKind, OutputRequirement
from xpool.quantization import Unquantized
from xpool.runtime.ffnagent.weights import DenseFfnWeights, MoeFfnWeights, MoeRouterWeights, ProjectionWeights

EXECUTION_HIDDEN_SIZE = 16
DENSE_INTERMEDIATE_SIZE = 32
MOE_INTERMEDIATE_SIZE = 16
MOE_EXPERT_COUNT = 4
MOE_TOPK = 2


def decode_mxfp4_weights(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Expand an isolated reference through the independent E2M1 codebook."""

    codebook = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6), device=values.device)
    codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).long()
    factors = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return codebook[codes] * factors


def packed_moe_reference(
    hidden_states: torch.Tensor,
    *,
    gate_up: ProjectionWeights,
    down: ProjectionWeights,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
    activation: ffn.ActivationSpec,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Evaluate isolated packed Experts without the production decoder or kernels."""

    assert gate_up.scales is not None and down.scales is not None
    assert gate_up.bias is not None and down.bias is not None
    first = decode_mxfp4_weights(gate_up.values, gate_up.scales)
    second = decode_mxfp4_weights(down.values, down.scales)
    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert in range(first.shape[0]):
        rows, slots = torch.where(ids == expert)
        projected = hidden_states[rows].float() @ first[expert].t() + gate_up.bias[expert]
        gate = projected[:, ::2]
        up = projected[:, 1::2]
        if activation.limit is not None:
            gate = gate.clamp(max=activation.limit)
            up = up.clamp(min=-activation.limit, max=activation.limit)
        activated = (gate * torch.sigmoid(activation.alpha * gate) * (up + activation.up_offset)).to(
            hidden_states.dtype
        )
        partial = activated.float() @ second[expert].t() + down.bias[expert]
        weighted = (partial * probabilities[rows, slots, None]).to(hidden_states.dtype).float()
        result.index_add_(0, rows, weighted)
    return (result * routed_scaling_factor).to(hidden_states.dtype)


def execution_model_specs(
    *,
    instance_count: int,
    execution_kind: LayerKind,
    layer_count: int,
) -> tuple[ffn.FfnModelSpec, ...]:
    """Build the co-indexed Model Specs used by deterministic qualification."""

    def gated_projections(prefix: str) -> ffn.GatedProjectionSpec:
        return ffn.GatedProjectionSpec(
            gate=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.gate.weight"),
            up=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.up.weight"),
            down=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.down.weight"),
        )

    result = []
    for instance_index in range(instance_count):
        layers: list[ffn.FfnLayerSpec] = []
        for layer_ordinal in range(layer_count):
            prefix = f"qualification.{instance_index}.layers.{layer_ordinal}.mlp"
            if execution_kind is LayerKind.DENSE:
                layers.append(
                    ffn.DenseFfnSpec(
                        kind=execution_kind,
                        layer_id=layer_ordinal,
                        intermediate_size=DENSE_INTERMEDIATE_SIZE,
                        projections=gated_projections(prefix),
                    )
                )
            else:
                layers.append(
                    ffn.MoeFfnSpec(
                        kind=execution_kind,
                        router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                        router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
                        layer_id=layer_ordinal,
                        expert_intermediate_size=MOE_INTERMEDIATE_SIZE,
                        routed_expert_count=MOE_EXPERT_COUNT,
                        shared_expert_count=0,
                        routed_topk=MOE_TOPK,
                        renormalize=True,
                        routed_scaling_factor=1.0,
                        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.router.weight"),
                        router_correction_bias_key=None,
                        routed_experts=ffn.PerExpertSource(
                            experts=tuple(
                                gated_projections(f"{prefix}.experts.{expert_index}")
                                for expert_index in range(MOE_EXPERT_COUNT)
                            ),
                        ),
                        shared_expert=None,
                    )
                )
        result.append(
            ffn.FfnModelSpec(
                model_id=ModelId(f"qualification/{execution_kind.name.lower()}-{instance_index}"),
                hidden_size=EXECUTION_HIDDEN_SIZE,
                activation=ffn.SiluActivation(),
                layers=tuple(layers),
            )
        )
    return tuple(result)


def execution_fabric_plan(
    *,
    uid: str,
    atnagent_count: int,
    ffnagent_count: int,
    ffn_tp_size: int,
    executor_lane_count: int,
    instance_count: int,
    execution_kind: LayerKind,
    layer_count: int = 1,
    decode_payload_row_capacity: int = 4,
    prefill_payload_row_capacity: int = 4,
    payload_dtype: torch.dtype,
) -> FabricPlan:
    """Build one explicit canonical Plan for deterministic FFN execution."""

    model_specs = execution_model_specs(
        instance_count=instance_count,
        execution_kind=execution_kind,
        layer_count=layer_count,
    )
    intermediate_size = DENSE_INTERMEDIATE_SIZE if execution_kind is LayerKind.DENSE else MOE_INTERMEDIATE_SIZE
    if intermediate_size % ffn_tp_size:
        raise ValueError("qualification FFN width must divide evenly across FfnAgents")
    profile = InstanceFfnProfile(
        payload_dtype=payload_dtype,
        hidden_size=EXECUTION_HIDDEN_SIZE,
        layers=tuple(InstanceFfnLayerProfile(layer_id=ordinal, kind=execution_kind) for ordinal in range(layer_count)),
        decode_payload_row_capacity=decode_payload_row_capacity,
        prefill_payload_row_capacity=prefill_payload_row_capacity,
        group_sum_complete_admitted=True,
    )
    execution_group = tuple(range(ffn_tp_size))
    layer = (
        DenseFfnLayerPlan(
            ffnagent_indices=execution_group,
            local_intermediate_size=DENSE_INTERMEDIATE_SIZE // ffn_tp_size,
        )
        if execution_kind is LayerKind.DENSE
        else MoeFfnLayerPlan(
            ffnagent_indices=execution_group,
            local_intermediate_size=MOE_INTERMEDIATE_SIZE // ffn_tp_size,
            effective_topk=MOE_TOPK,
        )
    )
    return FabricPlan(
        generation=FabricGenerationId(high=1, low=1),
        uid=FabricUid(uid),
        pe_placements=tuple(
            FabricPePlacement(role=FabricRole.ATNAGENT, device=index) for index in range(atnagent_count)
        )
        + tuple(
            FabricPePlacement(role=FabricRole.FFNAGENT, device=atnagent_count + index)
            for index in range(ffnagent_count)
        ),
        executor_lane_count=executor_lane_count,
        scheduler=FifoSchedulerPolicy(),
        model_plans=tuple(
            FfnModelPlan(model_spec_digest=model_spec.digest(), layers=(layer,) * layer_count)
            for model_spec in model_specs
        ),
        instance_plans=tuple(
            FabricInstancePlan(
                model_id=model_spec.model_id,
                ffn_profile=profile,
                instance_rank_topology=InstanceRankTopology(
                    atn_tp_size=atnagent_count,
                    atn_dp_size=1,
                    atnagent_indices=tuple(range(atnagent_count)),
                ),
            )
            for model_spec in model_specs
        ),
    )


def dense_layer_weights(
    *, ffnagent_index: int, ffnagent_count: int, device: int, payload_dtype: torch.dtype, layer_ordinal: int = 0
) -> DenseFfnWeights:
    """Materialize one deterministic true-TP Dense shard directly on a device."""

    local_width = DENSE_INTERMEDIATE_SIZE // ffnagent_count
    begin = ffnagent_index * local_width
    end = begin + local_width
    gate, up, down = dense_full_weights(device=device, payload_dtype=payload_dtype, layer_ordinal=layer_ordinal)
    return DenseFfnWeights(
        gate_up=ProjectionWeights(
            values=torch.cat((gate[begin:end], up[begin:end])).contiguous(), scales=None, bias=None
        ),
        down=ProjectionWeights(values=down[:, begin:end].contiguous(), scales=None, bias=None),
    )


def dense_reference(hidden_states: torch.Tensor, *, ffnagent_count: int, layer_ordinal: int = 0) -> torch.Tensor:
    """Evaluate the exact ordered sum of deterministic true-TP partials."""

    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    for ffnagent_index in range(ffnagent_count):
        result.add_(
            dense_partial_reference(
                hidden_states,
                ffnagent_index=ffnagent_index,
                ffnagent_count=ffnagent_count,
                layer_ordinal=layer_ordinal,
            )
        )
    return result.to(hidden_states.dtype)


def dense_partial_reference(
    hidden_states: torch.Tensor,
    *,
    ffnagent_index: int,
    ffnagent_count: int,
    layer_ordinal: int = 0,
) -> torch.Tensor:
    """Evaluate one deterministic rank-local Dense partial in payload dtype."""

    gate, up, down = dense_full_weights(
        device=hidden_states.device.index,
        payload_dtype=hidden_states.dtype,
        layer_ordinal=layer_ordinal,
    )
    local_width = DENSE_INTERMEDIATE_SIZE // ffnagent_count
    begin = ffnagent_index * local_width
    end = begin + local_width
    gate_values = torch.mm(hidden_states, gate[begin:end].t())
    up_values = torch.mm(hidden_states, up[begin:end].t())
    activated = (torch.nn.functional.silu(gate_values.float()) * up_values.float()).to(hidden_states.dtype)
    return torch.mm(activated, down[:, begin:end].t())


def dense_full_weights(
    *, device: int | None, payload_dtype: torch.dtype, layer_ordinal: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic full Dense tensors shared by producers and Reference."""

    def values(shape: tuple[int, ...], offset: int) -> torch.Tensor:
        count = torch.Size(shape).numel()
        data = torch.arange(count, device=device, dtype=torch.float32)
        return (((data + offset).remainder(29) - 14) / 64).reshape(shape).to(payload_dtype)

    return (
        values((DENSE_INTERMEDIATE_SIZE, EXECUTION_HIDDEN_SIZE), 3 + layer_ordinal * 5),
        values((DENSE_INTERMEDIATE_SIZE, EXECUTION_HIDDEN_SIZE), 11 + layer_ordinal * 7),
        values((EXECUTION_HIDDEN_SIZE, DENSE_INTERMEDIATE_SIZE), 19 + layer_ordinal * 11),
    )


def moe_layer_weights(
    *, ffnagent_index: int, ffnagent_count: int, device: int, payload_dtype: torch.dtype, layer_ordinal: int = 0
) -> MoeFfnWeights:
    """Materialize one deterministic true-TP MoE shard directly on a device."""

    local_width = MOE_INTERMEDIATE_SIZE // ffnagent_count
    begin = ffnagent_index * local_width
    end = begin + local_width
    router, gate, up, down = moe_full_weights(device=device, payload_dtype=payload_dtype, layer_ordinal=layer_ordinal)
    return MoeFfnWeights(
        gate_up=ProjectionWeights(
            values=torch.cat((gate[:, begin:end], up[:, begin:end]), dim=1).contiguous(), scales=None, bias=None
        ),
        down=ProjectionWeights(values=down[:, :, begin:end].contiguous(), scales=None, bias=None),
        router=MoeRouterWeights(
            projection=ProjectionWeights(values=router, scales=None, bias=None), correction_bias=None
        )
        if ffnagent_index == 0
        else None,
    )


def moe_reference(hidden_states: torch.Tensor, *, ffnagent_count: int, layer_ordinal: int = 0) -> torch.Tensor:
    """Evaluate deterministic routing and ordered true-TP MoE partials."""

    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    for ffnagent_index in range(ffnagent_count):
        result.add_(
            moe_partial_reference(
                hidden_states,
                ffnagent_index=ffnagent_index,
                ffnagent_count=ffnagent_count,
                layer_ordinal=layer_ordinal,
            )
        )
    return result.to(hidden_states.dtype)


def moe_partial_reference(
    hidden_states: torch.Tensor,
    *,
    ffnagent_index: int,
    ffnagent_count: int,
    layer_ordinal: int = 0,
) -> torch.Tensor:
    """Evaluate one deterministic rank-local MoE partial in payload dtype."""

    router, gate, up, down = moe_full_weights(
        device=hidden_states.device.index,
        payload_dtype=hidden_states.dtype,
        layer_ordinal=layer_ordinal,
    )
    scores = torch.softmax(torch.mm(hidden_states, router.t()).float(), dim=-1)
    route_weights, route_ids = torch.topk(scores, MOE_TOPK, dim=-1)
    route_weights /= route_weights.sum(dim=-1, keepdim=True)
    local_width = MOE_INTERMEDIATE_SIZE // ffnagent_count
    begin = ffnagent_index * local_width
    end = begin + local_width
    partial = torch.zeros_like(hidden_states, dtype=torch.float32)
    for row in range(hidden_states.shape[0]):
        for route in range(MOE_TOPK):
            expert = int(route_ids[row, route])
            gate_values = torch.mv(gate[expert, begin:end], hidden_states[row])
            up_values = torch.mv(up[expert, begin:end], hidden_states[row])
            activated = (torch.nn.functional.silu(gate_values.float()) * up_values.float()).to(hidden_states.dtype)
            expert_output = torch.mv(down[expert, :, begin:end], activated)
            partial[row].add_(expert_output.float() * route_weights[row, route])
    return partial.to(hidden_states.dtype)


def execution_reference(
    hidden_states: torch.Tensor,
    *,
    execution_kind: LayerKind,
    ffn_tp_size: int,
    output_count: int,
    output_rank: int,
    output_requirement: OutputRequirement,
    layer_ordinal: int = 0,
) -> torch.Tensor:
    """Project one complete or direct-partial result to an Output rank."""

    complete = dense_reference if execution_kind is LayerKind.DENSE else moe_reference
    partial = dense_partial_reference if execution_kind is LayerKind.DENSE else moe_partial_reference
    if output_requirement is OutputRequirement.PER_RANK_COMPLETE or ffn_tp_size > output_count:
        return (
            complete(hidden_states, ffnagent_count=ffn_tp_size, layer_ordinal=layer_ordinal)
            if output_rank == 0 or output_requirement is OutputRequirement.PER_RANK_COMPLETE
            else torch.zeros_like(hidden_states)
        )
    if output_rank >= ffn_tp_size:
        return torch.zeros_like(hidden_states)
    return partial(
        hidden_states,
        ffnagent_index=output_rank,
        ffnagent_count=ffn_tp_size,
        layer_ordinal=layer_ordinal,
    )


def moe_full_weights(
    *, device: int | None, payload_dtype: torch.dtype, layer_ordinal: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic full MoE tensors shared by producers and Reference."""

    def values(shape: tuple[int, ...], offset: int) -> torch.Tensor:
        count = torch.Size(shape).numel()
        data = torch.arange(count, device=device, dtype=torch.float32)
        return (((data + offset).remainder(31) - 15) / 96).reshape(shape).to(payload_dtype)

    return (
        values((MOE_EXPERT_COUNT, EXECUTION_HIDDEN_SIZE), 5 + layer_ordinal * 3),
        values((MOE_EXPERT_COUNT, MOE_INTERMEDIATE_SIZE, EXECUTION_HIDDEN_SIZE), 7 + layer_ordinal * 5),
        values((MOE_EXPERT_COUNT, MOE_INTERMEDIATE_SIZE, EXECUTION_HIDDEN_SIZE), 13 + layer_ordinal * 7),
        values((MOE_EXPERT_COUNT, EXECUTION_HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE), 23 + layer_ordinal * 11),
    )
