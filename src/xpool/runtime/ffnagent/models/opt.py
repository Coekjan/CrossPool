"""Biased unary Dense OPT FFN architecture compiler."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class OPTAdapter(architecture.FfnModelAdapter):
    """Compile the biased unary Dense OPT FFN profile."""

    architecture_name = "OPTForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile all main OPT decoder layers as biased unary Dense FFNs."""

        dtype_field = "torch_dtype" if model_config.values.get("torch_dtype") is not None else "dtype"
        model_config.validate_family_profile(
            model_type="opt",
            dtype_field=dtype_field,
            activation_field="activation_function",
            allowed_activations=("relu",),
            allowed_dtypes=("float16",),
        )
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("ffn_dim", int, ge=1)
        enable_bias = model_config.optional("enable_bias", bool, allow_none=True)
        if enable_bias is False:
            raise ValueError("OPT FFN requires enable_bias=true")
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.RELU,
            checkpoint_namespace="model.decoder.layers.{layer_id}",
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layout=ffn.DenseFfnLayout.BIASED_RELU,
                    layer_id=layer_id,
                    intermediate_size=intermediate_size,
                    checkpoint=architecture.biased_checkpoint_keys(f"model.decoder.layers.{layer_id}"),
                )
                for layer_id in range(layer_count)
            ),
        )
