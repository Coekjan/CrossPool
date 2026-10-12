"""Dense Llama FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class LlamaAdapter(architecture.FfnModelAdapter):
    """Compile the gated Dense Llama FFN profile."""

    architecture_name = "LlamaForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile all main Llama decoder layers as gated Dense FFNs."""

        if model_config.get("model_type", str) != "llama":
            raise ValueError("model_type must equal 'llama'")
        if model_config.get("hidden_act", str) != "silu":
            raise ValueError("hidden_act must equal 'silu'")
        if model_config.get("torch_dtype", str) != "float16":
            raise ValueError("torch_dtype must equal 'float16'")
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
