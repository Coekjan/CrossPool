"""GLM-4 MoE Lite FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Glm4MoeLiteAdapter(architecture.FfnModelAdapter):
    """Compile the strict mixed Dense/MoE GLM-4 Lite FFN profile."""

    architecture_name = "Glm4MoeLiteForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile GLM main layers and corrected-sigmoid routing semantics."""

        model_config.validate_family_profile(model_type="glm4_moe_lite", dtype_field="dtype")
        explicit_frequency = model_config.optional("moe_layer_freq", int, allow_none=True)
        if explicit_frequency is not None and explicit_frequency != 1:
            raise ValueError("explicit moe_layer_freq must equal the GLM family constant 1")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("intermediate_size", int, ge=1)
        expert_intermediate_size = model_config.get("moe_intermediate_size", int, ge=1)
        routed_expert_count = model_config.get("n_routed_experts", int, ge=1)
        shared_expert_count = model_config.get("n_shared_experts", int, ge=0)
        routed_topk = model_config.get("num_experts_per_tok", int, ge=1)
        first_moe_layer = model_config.get("first_k_dense_replace", int, ge=0)
        expert_group_count = model_config.get("n_group", int, ge=1)
        selected_expert_group_count = model_config.get("topk_group", int, ge=1)
        if model_config.get("topk_method", str) != "noaux_tc":
            raise ValueError("topk_method must equal 'noaux_tc'")
        if expert_group_count != 1 or selected_expert_group_count != 1:
            raise ValueError("first-production GLM routing requires n_group == topk_group == 1")
        renormalize = model_config.get("norm_topk_prob", bool)
        routed_scaling_factor = model_config.get("routed_scaling_factor", float, gt=0)

        layers: list[ffn.FfnLayerSpec] = []
        for layer_id in range(layer_count):
            if layer_id < first_moe_layer:
                layers.append(
                    ffn.DenseFfnSpec(
                        kind=LayerKind.DENSE,
                        layer_id=layer_id,
                        intermediate_size=intermediate_size,
                        projections=model_config.gated_projections(f"model.layers.{layer_id}.mlp"),
                    )
                )
                continue
            layers.append(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    router_algorithm=ffn.RouterAlgorithm.SIGMOID,
                    router_projection_precision=ffn.RouterProjectionPrecision.FP32,
                    layer_id=layer_id,
                    expert_intermediate_size=expert_intermediate_size,
                    routed_expert_count=routed_expert_count,
                    shared_expert_count=shared_expert_count,
                    routed_topk=routed_topk,
                    renormalize=renormalize,
                    routed_scaling_factor=routed_scaling_factor,
                    router=model_config.projection(f"model.layers.{layer_id}.mlp.gate"),
                    router_correction_bias_key=f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias",
                    routed_experts=ffn.PerExpertSource(
                        experts=tuple(
                            model_config.gated_projections(f"model.layers.{layer_id}.mlp.experts.{expert_id}")
                            for expert_id in range(routed_expert_count)
                        ),
                    ),
                    shared_expert=(
                        model_config.gated_projections(f"model.layers.{layer_id}.mlp.shared_experts")
                        if shared_expert_count > 0
                        else None
                    ),
                )
            )
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=ffn.SiluActivation(),
            layers=tuple(layers),
        )
