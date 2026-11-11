"""Original multimodal Qwen3.5 Dense decoder FFN specification."""

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Qwen35Adapter(architecture.FfnModelAdapter):
    """Compile only the language decoder's main Dense layers."""

    architecture_name = "Qwen3_5ForConditionalGeneration"
    decoder_prefix = "model.language_model.layers"

    @classmethod
    def compile(cls, *, model_id: ModelId, model_config: architecture.FfnSourceConfig) -> ffn.FfnModelSpec:
        if model_config.get("model_type", str) != "qwen3_5":
            raise ValueError("Qwen3.5 requires its original multimodal configuration")
        text = architecture.FfnSourceConfig(model_config.get("text_config", dict))
        text.validate_family_profile(model_type="qwen3_5_text", dtype_field="dtype")
        if model_config.optional("quantization_config", dict, allow_none=True) is not None:
            raise ValueError("Qwen3.5 requires unquantized source weights")
        if text.optional("quantization_config", dict, allow_none=True) is not None:
            raise ValueError("Qwen3.5 requires unquantized language weights")
        layer_count = text.get("num_hidden_layers", int, ge=1)
        kinds = text.get("layer_types", list)
        if len(kinds) != layer_count or any(kind not in ("linear_attention", "full_attention") for kind in kinds):
            raise ValueError("Qwen3.5 requires one supported attention declaration per main layer")
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=text.get("hidden_size", int, ge=1),
            activation=ffn.SiluActivation(),
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layer_id=layer,
                    intermediate_size=text.get("intermediate_size", int, ge=1),
                    projections=text.gated_projections(f"{cls.decoder_prefix}.{layer}.mlp"),
                )
                for layer in range(layer_count)
            ),
        )
