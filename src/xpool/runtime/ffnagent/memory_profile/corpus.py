"""Fixed model-independent FFN memory-calibration corpus."""

from __future__ import annotations

import hashlib

import torch

from xpool import ffn
from xpool.config import XpoolConfig
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
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import weights

DECODE_PAYLOAD_ROW_CAPACITY = 2048
PREFILL_PAYLOAD_ROW_CAPACITY = 4096
FIT_COORDINATES = ("C0a", "C1", "C2", "C3", "D-T")
HELD_OUT_COORDINATE = "H0"


def calibration_gated_keys(prefix: str) -> ffn.GatedFfnCheckpointKeys:
    """Create unique semantic checkpoint keys without reading a model directory."""

    return ffn.GatedFfnCheckpointKeys(
        gate_weight_key=f"{prefix}.gate.weight",
        up_weight_key=f"{prefix}.up.weight",
        down_weight_key=f"{prefix}.down.weight",
    )


def calibration_moe_layer(
    *,
    member: str,
    layer_id: int,
    expert_intermediate_size: int,
    routed_expert_count: int,
    shared_expert_count: int,
    routed_topk: int,
    correction_bias_present: bool,
    renormalize: bool,
    routed_scaling_factor: float,
) -> ffn.MoeFfnSpec:
    """Create one accepted calibration-corpus MoE layer."""

    prefix = f"calibration.{member}.layers.{layer_id}.mlp"
    correction_bias_key = f"{prefix}.router.correction_bias" if correction_bias_present else None
    return ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=layer_id,
        expert_intermediate_size=expert_intermediate_size,
        shared_expert_count=shared_expert_count,
        routed_topk=routed_topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        checkpoint=ffn.MoeFfnCheckpointKeys(
            router_weight_key=f"{prefix}.router.weight",
            router_correction_bias_key=correction_bias_key,
            routed_experts=tuple(
                calibration_gated_keys(f"{prefix}.experts.{expert_id}") for expert_id in range(routed_expert_count)
            ),
            shared_expert=(calibration_gated_keys(f"{prefix}.shared") if shared_expert_count else None),
        ),
    )


def calibration_corpus_spec(member: str) -> ffn.FfnModelSpec:
    """Build one fixed implementation-domain Corpus member without checkpoint I/O."""

    if member == "gated-dense":
        hidden_size = 5120
        layers: tuple[ffn.FfnLayerSpec, ...] = (
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=17408,
                checkpoint=calibration_gated_keys(f"calibration.{member}.layers.0.mlp"),
            ),
        )
    elif member == "softmax-shared-moe64":
        hidden_size = 2048
        layers = (
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10944,
                checkpoint=calibration_gated_keys(f"calibration.{member}.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member=member,
                layer_id=1,
                expert_intermediate_size=1408,
                routed_expert_count=64,
                shared_expert_count=2,
                routed_topk=6,
                correction_bias_present=False,
                renormalize=False,
                routed_scaling_factor=1.0,
            ),
        )
    elif member == "corrected-shared-moe64":
        hidden_size = 2048
        layers = (
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10240,
                checkpoint=calibration_gated_keys(f"calibration.{member}.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member=member,
                layer_id=1,
                expert_intermediate_size=1536,
                routed_expert_count=64,
                shared_expert_count=1,
                routed_topk=4,
                correction_bias_present=True,
                renormalize=True,
                routed_scaling_factor=1.8,
            ),
        )
    elif member == "softmax-moe128":
        hidden_size = 2048
        layers = (
            calibration_moe_layer(
                member=member,
                layer_id=0,
                expert_intermediate_size=768,
                routed_expert_count=128,
                shared_expert_count=0,
                routed_topk=8,
                correction_bias_present=False,
                renormalize=True,
                routed_scaling_factor=1.0,
            ),
        )
    else:
        raise ValueError(f"unsupported FFN Calibration Corpus member {member!r}")
    architecture_names = {
        "gated-dense": "Qwen3ForCausalLM",
        "softmax-shared-moe64": "DeepseekV2ForCausalLM",
        "corrected-shared-moe64": "Glm4MoeLiteForCausalLM",
        "softmax-moe128": "Qwen3MoeForCausalLM",
    }
    return ffn.FfnModelSpec(
        model_id=member,
        architecture_name=architecture_names[member],
        model_config_digest=hashlib.sha256(f"calibration:{member}".encode()).hexdigest(),
        hidden_size=hidden_size,
        activation=ffn.ActivationKind.SILU,
        layers=layers,
    )


def coordinate_members(
    coordinate: str,
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> tuple[tuple[str, int, int, bool], ...]:
    """Return fixed Corpus members, TP widths, Output counts, and admission modes."""

    if coordinate == "C0a":
        return (("gated-dense", 1, 1, True),)
    if coordinate == "C1":
        return (("softmax-shared-moe64", 1, 1, True),)
    if coordinate == "C2":
        return (("corrected-shared-moe64", min(ffnagent_count, 2), 1, True),)
    if coordinate == "C3":
        return (("softmax-moe128", min(ffnagent_count, 2), 1, True),)
    if coordinate == "D-T" and ffnagent_count >= 2:
        return (("gated-dense", 2, 1, True),)
    if coordinate == HELD_OUT_COORDINATE:
        return (
            ("softmax-moe128", min(ffnagent_count, 4), min(atnagent_count, 2), True),
            ("gated-dense", min(ffnagent_count, 2), min(atnagent_count, 2), False),
        )
    raise ValueError(f"coordinate {coordinate!r} is unreachable for F={ffnagent_count}")


def build_instance_profile(spec: ffn.FfnModelSpec, *, group_sum_complete: bool) -> InstanceFfnProfile:
    """Project one Calibration Corpus Spec into the fixed 4K profiling Profile."""

    return InstanceFfnProfile(
        model_config_digest=spec.model_config_digest,
        payload_dtype=torch.bfloat16,
        hidden_size=spec.hidden_size,
        layers=tuple(InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in spec.layers),
        decode_payload_row_capacity=DECODE_PAYLOAD_ROW_CAPACITY,
        prefill_payload_row_capacity=PREFILL_PAYLOAD_ROW_CAPACITY,
        group_sum_complete_admitted=group_sum_complete,
    )


def build_model_plan(spec: ffn.FfnModelSpec, *, tp_size: int) -> FfnModelPlan:
    """Place every Corpus layer on one stable primary FFN TP group."""

    group = tuple(range(tp_size))
    layers = []
    for layer in spec.layers:
        if isinstance(layer, ffn.DenseFfnSpec):
            if layer.intermediate_size % tp_size:
                raise ValueError(f"Dense layer {layer.layer_id} is not divisible by TP={tp_size}")
            layers.append(
                DenseFfnLayerPlan(
                    ffnagent_indices=group,
                    local_intermediate_size=layer.intermediate_size // tp_size,
                )
            )
            continue
        if layer.expert_intermediate_size % tp_size:
            raise ValueError(f"MoE layer {layer.layer_id} is not divisible by TP={tp_size}")
        layers.append(
            MoeFfnLayerPlan(
                ffnagent_indices=group,
                local_intermediate_size=layer.expert_intermediate_size // tp_size,
                effective_topk=layer.routed_topk + layer.shared_expert_count,
            )
        )
    return FfnModelPlan(model_spec_digest=spec.digest(), layers=tuple(layers))


def build_fabric_plan(
    *,
    uid: str,
    coordinate: str,
    config: XpoolConfig,
    model_specs: tuple[ffn.FfnModelSpec, ...],
) -> FabricPlan:
    """Construct one complete configured-fleet calibration Fabric Plan."""

    members = coordinate_members(
        coordinate,
        atnagent_count=len(config.devices.atn_cuda_devices),
        ffnagent_count=len(config.devices.ffn_cuda_devices),
    )
    model_plans = tuple(
        build_model_plan(spec, tp_size=member[1]) for spec, member in zip(model_specs, members, strict=True)
    )
    instance_plans = tuple(
        FabricInstancePlan(
            instance_id=member[0],
            ffn_profile=build_instance_profile(spec, group_sum_complete=member[3]),
            instance_rank_topology=InstanceRankTopology(
                atn_tp_size=member[2],
                atn_dp_size=1,
                atnagent_indices=tuple(range(member[2])),
            ),
        )
        for spec, member in zip(model_specs, members, strict=True)
    )
    return FabricPlan(
        generation=FabricGenerationId.create(),
        uid=FabricUid(uid),
        pe_placements=tuple(
            FabricPePlacement(role=FabricRole.ATNAGENT, cuda_device=device)
            for device in config.devices.atn_cuda_devices
        )
        + tuple(
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=device)
            for device in config.devices.ffn_cuda_devices
        ),
        executor_lane_count=config.scheduler.ffn_concurrency,
        scheduler=FifoSchedulerPolicy(),
        model_plans=model_plans,
        instance_plans=instance_plans,
    )


def materialize_calibration_weights(
    *,
    fabric_plan: FabricPlan,
    model_specs: tuple[ffn.FfnModelSpec, ...],
    ffnagent_index: int,
) -> tuple[tuple[weights.FfnLayerWeights | None, ...], ...]:
    """Allocate exact production-shaped CUDA weights without checkpoint I/O."""

    materialized = []
    for model_plan, model_spec in zip(fabric_plan.model_plans, model_specs, strict=True):
        model_weights = []
        for layer_plan, layer_spec in zip(model_plan.layers, model_spec.layers, strict=True):
            if ffnagent_index not in layer_plan.ffnagent_indices:
                model_weights.append(None)
                continue
            tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
            if isinstance(layer_plan, DenseFfnLayerPlan):
                model_weights.append(
                    weights.DenseFfnWeights(
                        gate_up_weight=torch.empty(
                            2 * layer_plan.local_intermediate_size,
                            model_spec.hidden_size,
                            dtype=torch.bfloat16,
                            device="cuda",
                        ),
                        down_weight=torch.empty(
                            model_spec.hidden_size,
                            layer_plan.local_intermediate_size,
                            dtype=torch.bfloat16,
                            device="cuda",
                        ),
                    )
                )
                continue
            if not isinstance(layer_spec, ffn.MoeFfnSpec):
                raise TypeError("calibration MoE Plan requires a MoE Model Spec")
            expert_count = layer_spec.routed_expert_count + layer_spec.shared_expert_count
            router = None
            if tp_rank == 0:
                router = weights.MoeRouterWeights(
                    weight=torch.zeros(
                        layer_spec.routed_expert_count,
                        model_spec.hidden_size,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    correction_bias=(
                        torch.zeros(layer_spec.routed_expert_count, dtype=torch.float32, device="cuda")
                        if layer_spec.checkpoint.router_correction_bias_key is not None
                        else None
                    ),
                )
            model_weights.append(
                weights.MoeFfnWeights(
                    expert_gate_up_weight=torch.empty(
                        expert_count,
                        2 * layer_plan.local_intermediate_size,
                        model_spec.hidden_size,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    expert_down_weight=torch.empty(
                        expert_count,
                        model_spec.hidden_size,
                        layer_plan.local_intermediate_size,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    router=router,
                )
            )
        materialized.append(tuple(model_weights))
    return tuple(materialized)
