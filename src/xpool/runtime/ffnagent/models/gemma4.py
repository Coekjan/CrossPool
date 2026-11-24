"""Dense FFN architecture compilation for the two Gemma 4 checkpoints."""

from __future__ import annotations

from collections.abc import Collection

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Gemma4E2BAdapter(architecture.FfnModelAdapter):
    """Compile the Gemma 4 E2B text decoder's gated Dense FFNs."""

    architecture_name = "Gemma4ForConditionalGeneration"
    model_name = "google/gemma-4-E2B-it"
    model_type = "gemma4"
    text_model_type = "gemma4_text"

    @classmethod
    def checkpoint_prefixes(cls, layer: ffn.FfnLayerSpec) -> tuple[str, ...]:
        return (f"model.language_model.layers.{layer.layer_id}.mlp.",)

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
        checkpoint_keys: Collection[str],
    ) -> ffn.FfnModelSpec:
        """Resolve the decoder FFN keys and layer widths from the text config."""

        if str(model_id) != cls.model_name or model_config.get("model_type", str) != cls.model_type:
            raise ValueError(f"{cls.architecture_name} requires the original {cls.model_name} checkpoint")
        text = architecture.FfnSourceConfig(model_config.get("text_config", dict))
        if text.get("model_type", str) != cls.text_model_type:
            raise ValueError(f"Gemma 4 text_config.model_type must equal {cls.text_model_type!r}")
        if model_config.get("dtype", str) != "bfloat16" or text.optional("dtype", str) not in (None, "bfloat16"):
            raise ValueError("Gemma 4 requires a BF16 checkpoint")
        if text.get("hidden_activation", str) != "gelu_pytorch_tanh":
            raise ValueError("Gemma 4 requires gelu_pytorch_tanh on the text decoder")
        if text.get("enable_moe_block", bool) or any(
            config.optional("quantization_config", dict, allow_none=True) is not None for config in (model_config, text)
        ):
            raise ValueError("Gemma 4 requires an unquantized Dense text decoder")

        hidden_size = text.get("hidden_size", int, ge=1)
        layer_count = text.get("num_hidden_layers", int, ge=1)
        intermediate_size = text.get("intermediate_size", int, ge=1)
        shared_layer_count = text.get("num_kv_shared_layers", int, ge=0)
        if shared_layer_count >= layer_count:
            raise ValueError("Gemma 4 KV sharing requires an earlier donor layer")
        wide_layer_start = layer_count - shared_layer_count if text.get("use_double_wide_mlp", bool) else layer_count
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=ffn.GeluActivation(approximate="tanh"),
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layer_id=layer_id,
                    intermediate_size=intermediate_size * (2 if layer_id >= wide_layer_start else 1),
                    projections=model_config.gated_projections(f"model.language_model.layers.{layer_id}.mlp"),
                )
                for layer_id in range(layer_count)
            ),
        )


class Gemma4UnifiedAdapter(Gemma4E2BAdapter):
    """Compile the encoder-free Gemma 4 12B wrapper's text decoder."""

    architecture_name = "Gemma4UnifiedForConditionalGeneration"
    model_name = "google/gemma-4-12B-it"
    model_type = "gemma4_unified"
    text_model_type = "gemma4_unified_text"
