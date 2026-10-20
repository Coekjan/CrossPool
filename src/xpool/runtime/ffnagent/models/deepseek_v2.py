"""DeepSeek-V2 FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class DeepseekV2Adapter(architecture.FfnModelAdapter):
    """Compile the strict mixed Dense/MoE DeepSeek-V2 FFN profile."""

    architecture_name = "DeepseekV2ForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile DeepSeek layer placement and routing semantics."""

        model_config.validate_family_profile(model_type="deepseek_v2", dtype_field="torch_dtype")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("intermediate_size", int, ge=1)
        expert_intermediate_size = model_config.get("moe_intermediate_size", int, ge=1)
        routed_expert_count = model_config.get("n_routed_experts", int, ge=1)
        shared_expert_count = model_config.get("n_shared_experts", int, ge=0)
        routed_topk = model_config.get("num_experts_per_tok", int, ge=1)
        first_moe_layer = model_config.get("first_k_dense_replace", int, ge=0)
        moe_layer_frequency = model_config.get("moe_layer_freq", int, ge=1)
        expert_group_count = model_config.get("n_group", int, ge=1)
        selected_expert_group_count = model_config.get("topk_group", int, ge=1)
        if model_config.get("scoring_func", str) != "softmax":
            raise ValueError("scoring_func must equal 'softmax'")
        if model_config.get("topk_method", str) != "greedy":
            raise ValueError("topk_method must equal 'greedy'")
        if expert_group_count != 1 or selected_expert_group_count != 1:
            raise ValueError("first-production DeepSeek routing requires n_group == topk_group == 1")
        renormalize = model_config.get("norm_topk_prob", bool)
        routed_scaling_factor = model_config.get("routed_scaling_factor", float, gt=0)

        layers: list[ffn.FfnLayerSpec] = []
        for layer_id in range(layer_count):
            if layer_id < first_moe_layer or layer_id % moe_layer_frequency != 0:
                layers.append(
                    ffn.DenseFfnSpec(
                        kind=LayerKind.DENSE,
                        layer_id=layer_id,
                        intermediate_size=intermediate_size,
                        checkpoint=architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp"),
                    )
                )
                continue
            layers.append(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                    router_precision=ffn.RouterPrecision.FP32_LOGITS,
                    layer_id=layer_id,
                    expert_intermediate_size=expert_intermediate_size,
                    shared_expert_count=shared_expert_count,
                    routed_topk=routed_topk,
                    renormalize=renormalize,
                    routed_scaling_factor=routed_scaling_factor,
                    checkpoint=ffn.MoeFfnCheckpointKeys(
                        router_weight_key=f"model.layers.{layer_id}.mlp.gate.weight",
                        router_correction_bias_key=None,
                        routed_experts=tuple(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.experts.{expert_id}")
                            for expert_id in range(routed_expert_count)
                        ),
                        shared_expert=(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.shared_experts")
                            if shared_expert_count > 0
                            else None
                        ),
                    ),
                )
            )
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(layers),
        )
