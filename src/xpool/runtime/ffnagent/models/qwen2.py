"""Dense Qwen2 FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Qwen2Adapter(architecture.FfnModelAdapter):
    """Compile the gated Dense Qwen2 FFN profile."""

    architecture_name = "Qwen2ForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile all main Qwen2 decoder layers as gated Dense FFNs."""

        model_config.validate_family_profile(model_type="qwen2", dtype_field="torch_dtype")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("intermediate_size", int, ge=1)
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layer_id=layer_id,
                    intermediate_size=intermediate_size,
                    checkpoint=architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp"),
                )
                for layer_id in range(layer_count)
            ),
        )
