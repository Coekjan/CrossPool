"""Qwen3-MoE FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Qwen3MoeAdapter(architecture.FfnModelAdapter):
    """Compile the strict all-MoE Qwen3 FFN profile."""

    architecture_name = "Qwen3MoeForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile all main Qwen3-MoE decoder layers and routing semantics."""

        model_config.validate_family_profile(model_type="qwen3_moe", dtype_field="torch_dtype")
        if model_config.get("decoder_sparse_step", int, ge=1) != 1:
            raise ValueError("decoder_sparse_step must equal 1")
        mlp_only_layers = model_config.optional("mlp_only_layers", list)
        if mlp_only_layers is None or mlp_only_layers:
            raise ValueError("mlp_only_layers must be an empty JSON array")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        expert_intermediate_size = model_config.get("moe_intermediate_size", int, ge=1)
        routed_expert_count = model_config.get("num_experts", int, ge=1)
        routed_topk = model_config.get("num_experts_per_tok", int, ge=1)
        renormalize = model_config.get("norm_topk_prob", bool)
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                    router_precision=ffn.RouterPrecision.PAYLOAD,
                    layer_id=layer_id,
                    expert_intermediate_size=expert_intermediate_size,
                    shared_expert_count=0,
                    routed_topk=routed_topk,
                    renormalize=renormalize,
                    routed_scaling_factor=1.0,
                    checkpoint=ffn.MoeFfnCheckpointKeys(
                        router_weight_key=f"model.layers.{layer_id}.mlp.gate.weight",
                        router_correction_bias_key=None,
                        routed_experts=tuple(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.experts.{expert_id}")
                            for expert_id in range(routed_expert_count)
                        ),
                        shared_expert=None,
                    ),
                )
                for layer_id in range(layer_count)
            ),
        )
