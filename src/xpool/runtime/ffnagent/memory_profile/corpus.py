"""Fixed model-independent FFN memory-calibration corpus."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import FabricPlan, InstanceRankTopology
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, Unquantized
from xpool.runtime.ffnagent import preparation, weights
from xpool.runtime.ffnagent.memory_profile.experiment import CalibrationExperiment, CalibrationModel


def calibration_projections(prefix: str) -> ffn.GatedProjectionSpec:
    """Create unique semantic checkpoint keys without reading a model directory."""

    return ffn.GatedProjectionSpec(
        gate=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.gate.weight"),
        up=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.up.weight"),
        down=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.down.weight"),
    )


def calibration_moe_layer(
    *,
    member: str,
    layer_id: int,
    router_algorithm: ffn.RouterAlgorithm,
    router_projection_precision: ffn.RouterProjectionPrecision,
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
        router_projection_precision=router_projection_precision,
        layer_id=layer_id,
        expert_intermediate_size=expert_intermediate_size,
        routed_expert_count=routed_expert_count,
        shared_expert_count=shared_expert_count,
        routed_topk=routed_topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.router.weight"),
        router_correction_bias_key=correction_bias_key,
        routed_experts=ffn.PerExpertSource(
            experts=tuple(
                calibration_projections(f"{prefix}.experts.{expert_id}") for expert_id in range(routed_expert_count)
            ),
        ),
        shared_expert=(calibration_projections(f"{prefix}.shared") if shared_expert_count else None),
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
        activation=ffn.SiluActivation(),
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=17408,
                projections=calibration_projections("calibration.gated-dense.layers.0.mlp"),
            ),
        ),
    )
    softmax_shared = ffn.FfnModelSpec(
        model_id=ModelId("calibration/softmax-shared-moe64"),
        hidden_size=2048,
        activation=ffn.SiluActivation(),
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10944,
                projections=calibration_projections("calibration.softmax-shared-moe64.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member="softmax-shared-moe64",
                layer_id=1,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_projection_precision=ffn.RouterProjectionPrecision.FP32_LOGITS,
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
        activation=ffn.SiluActivation(),
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=10240,
                projections=calibration_projections("calibration.corrected-shared-moe64.layers.0.mlp"),
            ),
            calibration_moe_layer(
                member="corrected-shared-moe64",
                layer_id=1,
                router_algorithm=ffn.RouterAlgorithm.SIGMOID,
                router_projection_precision=ffn.RouterProjectionPrecision.FP32,
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
        activation=ffn.SiluActivation(),
        layers=(
            calibration_moe_layer(
                member="softmax-moe128",
                layer_id=0,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
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
    packed = ffn.FfnModelSpec(
        model_id=ModelId("calibration/packed-moe"),
        hidden_size=2048,
        activation=ffn.SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0),
        layers=(
            ffn.MoeFfnSpec(
                kind=LayerKind.MOE,
                layer_id=0,
                router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
                expert_intermediate_size=3072,
                routed_expert_count=48,
                shared_expert_count=0,
                routed_topk=4,
                topk_weight_rounding=ffn.TopKWeightRounding.LOGITS_DTYPE,
                renormalize=True,
                routed_scaling_factor=1.0,
                router=ffn.ProjectionSpec(
                    quantization=Unquantized(),
                    values_key="calibration.packed-moe.router.weight",
                    bias_key="calibration.packed-moe.router.bias",
                ),
                routed_experts=ffn.StackedExpertSource(
                    gate_up=ffn.ProjectionSpec(
                        quantization=Mxfp4(),
                        values_key="calibration.packed-moe.gate_up.blocks",
                        scales_key="calibration.packed-moe.gate_up.scales",
                        values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                        bias_key="calibration.packed-moe.gate_up.bias",
                    ),
                    down=ffn.ProjectionSpec(
                        quantization=Mxfp4(),
                        values_key="calibration.packed-moe.down.blocks",
                        scales_key="calibration.packed-moe.down.scales",
                        values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                        bias_key="calibration.packed-moe.down.bias",
                    ),
                    gate_up_order=ffn.GateUpOrder.INTERLEAVED,
                ),
                shared_expert=None,
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
    for tp_size in dict.fromkeys((min(ffnagent_count, 2), min(ffnagent_count, 4))):
        fit_models += (
            CalibrationModel(
                spec=packed,
                ffn_tp_size=tp_size,
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
            CalibrationModel(
                spec=packed,
                ffn_tp_size=min(ffnagent_count, 4),
                topology=mixed_topology,
                group_sum_complete_admitted=True,
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
        model_weights: list[weights.FfnLayerWeights | None] = []
        for ordinal, layer_plan in enumerate(model_plan.layers):
            if ffnagent_index not in layer_plan.ffnagent_indices:
                model_weights.append(None)
                continue
            layout = preparation.prepare_ffn_layer(
                model_spec,
                layer_ordinal=ordinal,
                payload_dtype=torch.bfloat16,
                tp_size=model_plan.tp_size,
                tp_rank=layer_plan.ffnagent_indices.index(ffnagent_index),
            )
            if layer_plan.local_intermediate_size != layout.local_intermediate_size:
                raise ValueError("calibration Plan disagrees with prepared execution width")
            device = torch.device("cuda", torch.cuda.current_device())
            layer_weights = (
                weights.DenseFfnWeights.allocate(gate_up=layout.gate_up, down=layout.down, device=device)
                if isinstance(layout, preparation.DenseFfnLayerLayout)
                else weights.MoeFfnWeights.allocate(
                    gate_up=layout.experts.gate_up,
                    down=layout.experts.down,
                    router=layout.router,
                    correction_bias=layout.router_correction_bias,
                    device=device,
                )
            )
            model_weights.append(layer_weights)
        materialized.append(tuple(model_weights))
    return tuple(materialized)
