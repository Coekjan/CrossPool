"""GPT-OSS source semantics for stacked MXFP4 Experts."""

from __future__ import annotations

from xpool import ffn, quantization
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class GptOssAdapter(architecture.FfnModelAdapter):
    """Compile original GPT-OSS checkpoint identities and parameterized SiLU."""

    architecture_name = "GptOssForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Describe Router bias, interleaved Expert rows and block-shaped source values."""

        model_config.validate_family_profile(model_type="gpt_oss", dtype_field=None)
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("intermediate_size", int, ge=1)
        expert_count = model_config.get("num_local_experts", int, ge=1)
        topk = model_config.get("num_experts_per_tok", int, ge=1)
        encoding = model_config.get("quantization_config", dict)
        alpha = model_config.optional("hidden_act_alpha", float)
        activation = ffn.SiluActivation(
            alpha=1.702 if alpha is None else alpha,
            limit=model_config.get("swiglu_limit", float, gt=0),
            up_offset=1.0,
        )
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=activation,
            layers=tuple(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    layer_id=layer_id,
                    router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
                    router_projection_precision=ffn.RouterProjectionPrecision.PAYLOAD,
                    topk_weight_rounding=ffn.TopKWeightRounding.LOGITS_DTYPE,
                    expert_intermediate_size=intermediate_size,
                    routed_expert_count=expert_count,
                    shared_expert_count=0,
                    routed_topk=topk,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    router=ffn.ProjectionSpec(
                        quantization=quantization.resolve(encoding, module_name=f"model.layers.{layer_id}.mlp.router"),
                        values_key=f"model.layers.{layer_id}.mlp.router.weight",
                        bias_key=f"model.layers.{layer_id}.mlp.router.bias",
                    ),
                    routed_experts=ffn.StackedExpertSource(
                        gate_up=ffn.ProjectionSpec(
                            quantization=quantization.resolve(
                                encoding, module_name=f"model.layers.{layer_id}.mlp.experts.gate_up_proj"
                            ),
                            values_key=f"model.layers.{layer_id}.mlp.experts.gate_up_proj_blocks",
                            scales_key=f"model.layers.{layer_id}.mlp.experts.gate_up_proj_scales",
                            values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                            bias_key=f"model.layers.{layer_id}.mlp.experts.gate_up_proj_bias",
                        ),
                        down=ffn.ProjectionSpec(
                            quantization=quantization.resolve(
                                encoding, module_name=f"model.layers.{layer_id}.mlp.experts.down_proj"
                            ),
                            values_key=f"model.layers.{layer_id}.mlp.experts.down_proj_blocks",
                            scales_key=f"model.layers.{layer_id}.mlp.experts.down_proj_scales",
                            values_layout=ffn.ValuesSourceLayout.SCALE_BLOCKED,
                            bias_key=f"model.layers.{layer_id}.mlp.experts.down_proj_bias",
                        ),
                        gate_up_order=ffn.GateUpOrder.INTERLEAVED,
                    ),
                    shared_expert=None,
                )
                for layer_id in range(layer_count)
            ),
        )
