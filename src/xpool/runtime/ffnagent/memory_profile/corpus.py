"""Fixed model-independent FFN memory-calibration corpus."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import FabricPlan, InstanceRankTopology
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.memory_profile.experiment import CalibrationExperiment, CalibrationModel


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
    router_algorithm: ffn.RouterAlgorithm,
    router_precision: ffn.RouterPrecision,
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
        router_algorithm=router_algorithm,
        router_precision=router_precision,
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


def calibration_experiments(
    config: XpoolConfig,
) -> tuple[tuple[CalibrationExperiment, ...], CalibrationExperiment]:
    """Construct the ordered fitting matrix and its independent mixed experiment.

    Specifications are immutable and shared across experiment definitions;
    measurement worlds allocate their resources independently.
    """

    dense = ffn.FfnModelSpec(
        model_id=ModelId("calibration/gated-dense"),
        hidden_size=5120,
        activation=ffn.ActivationKind.SILU,
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=17408,
                checkpoint=calibration_gated_keys("calibration.gated-dense.layers.0.mlp"),
            ),
        ),
    )
    softmax_shared = ffn.FfnModelSpec(
        model_id=ModelId("calibration/softmax-shared-moe64"),
        hidden_size=2048,
        activation=ffn.ActivationKind.SILU,
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10944,
                checkpoint=calibration_gated_keys("calibration.softmax-shared-moe64.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member="softmax-shared-moe64",
                layer_id=1,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_precision=ffn.RouterPrecision.FP32_LOGITS,
                expert_intermediate_size=1408,
                routed_expert_count=64,
                shared_expert_count=2,
                routed_topk=6,
                correction_bias_present=False,
                renormalize=False,
                routed_scaling_factor=1.0,
            ),
        ),
    )
    corrected_shared = ffn.FfnModelSpec(
        model_id=ModelId("calibration/corrected-shared-moe64"),
        hidden_size=2048,
        activation=ffn.ActivationKind.SILU,
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10240,
                checkpoint=calibration_gated_keys("calibration.corrected-shared-moe64.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member="corrected-shared-moe64",
                layer_id=1,
                router_algorithm=ffn.RouterAlgorithm.SIGMOID,
                router_precision=ffn.RouterPrecision.FP32,
                expert_intermediate_size=1536,
                routed_expert_count=64,
                shared_expert_count=1,
                routed_topk=4,
                correction_bias_present=True,
                renormalize=True,
                routed_scaling_factor=1.8,
            ),
        ),
    )
    softmax = ffn.FfnModelSpec(
        model_id=ModelId("calibration/softmax-moe128"),
        hidden_size=2048,
        activation=ffn.ActivationKind.SILU,
        layers=(
            calibration_moe_layer(
                member="softmax-moe128",
                layer_id=0,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_precision=ffn.RouterPrecision.PAYLOAD,
                expert_intermediate_size=768,
                routed_expert_count=128,
                shared_expert_count=0,
                routed_topk=8,
                correction_bias_present=False,
                renormalize=True,
                routed_scaling_factor=1.0,
            ),
        ),
    )
    ffnagent_count = len(config.ffn.devices)
    single_output = InstanceRankTopology(atn_tp_size=1, atn_dp_size=1, atnagent_indices=(0,))
    fit_models = (
        CalibrationModel(
            spec=dense,
            ffn_tp_size=1,
            topology=single_output,
            group_sum_complete_admitted=True,
        ),
        CalibrationModel(
            spec=softmax_shared,
            ffn_tp_size=1,
            topology=single_output,
            group_sum_complete_admitted=True,
        ),
        CalibrationModel(
            spec=corrected_shared,
            ffn_tp_size=min(ffnagent_count, 2),
            topology=single_output,
            group_sum_complete_admitted=True,
        ),
        CalibrationModel(
            spec=softmax,
            ffn_tp_size=min(ffnagent_count, 2),
            topology=single_output,
            group_sum_complete_admitted=True,
        ),
    )
    if ffnagent_count >= 2:
        fit_models += (
            CalibrationModel(
                spec=dense,
                ffn_tp_size=2,
                topology=single_output,
                group_sum_complete_admitted=True,
            ),
        )
    output_count = min(len(config.atn.devices), 2)
    mixed_topology = InstanceRankTopology(
        atn_tp_size=output_count,
        atn_dp_size=1,
        atnagent_indices=tuple(range(output_count)),
    )
    held_out = CalibrationExperiment(
        models=(
            CalibrationModel(
                spec=softmax,
                ffn_tp_size=min(ffnagent_count, 4),
                topology=mixed_topology,
                group_sum_complete_admitted=True,
            ),
            CalibrationModel(
                spec=dense,
                ffn_tp_size=min(ffnagent_count, 2),
                topology=mixed_topology,
                group_sum_complete_admitted=False,
            ),
        ),
    )
    return tuple(CalibrationExperiment(models=(model,)) for model in fit_models), held_out


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
            device = torch.device("cuda")
            if isinstance(layer_spec, ffn.DenseFfnSpec):
                layer_weights = weights.DenseFfnWeights.allocate(
                    hidden_size=model_spec.hidden_size,
                    local_intermediate_size=layer_plan.local_intermediate_size,
                    payload_dtype=torch.bfloat16,
                    device=device,
                )
            else:
                layer_weights = weights.MoeFfnWeights.allocate(
                    layer_spec,
                    hidden_size=model_spec.hidden_size,
                    local_intermediate_size=layer_plan.local_intermediate_size,
                    payload_dtype=torch.bfloat16,
                    router_weight_dtype=(
                        layer_spec.router_precision.weight_dtype(payload_dtype=torch.bfloat16) if tp_rank == 0 else None
                    ),
                    device=device,
                )
                router = layer_weights.router
                if router is not None:
                    router.weight.zero_()
                    if router.correction_bias is not None:
                        router.correction_bias.zero_()
            model_weights.append(layer_weights)
        materialized.append(tuple(model_weights))
    return tuple(materialized)
