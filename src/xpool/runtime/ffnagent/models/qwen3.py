"""Dense Qwen3 FFN architecture compiler."""

from __future__ import annotations

from collections.abc import Mapping

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Qwen3Adapter(architecture.FfnModelAdapter):
    """Compile the strict all-Dense Qwen3 FFN profile."""

    architecture_name = "Qwen3ForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: Mapping[str, object],
        model_config_digest: str,
    ) -> ffn.FfnModelSpec:
        """Compile all main Qwen3 decoder layers as gated Dense FFNs."""

        architecture.require_family_profile(
            model_config,
            architecture_name="Qwen3ForCausalLM",
            model_type="qwen3",
            dtype_field="torch_dtype",
        )
        hidden_size = architecture.require_integer(model_config, "hidden_size", minimum=1)
        layer_count = architecture.require_integer(model_config, "num_hidden_layers", minimum=1)
        intermediate_size = architecture.require_integer(model_config, "intermediate_size", minimum=1)
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            model_config_digest=model_config_digest,
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
