"""Dense OPT FFN architecture compiler."""

from __future__ import annotations

import re
from collections.abc import Collection

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class OptAdapter(architecture.FfnModelAdapter):
    """Compile OPT's two projections and ReLU from its checkpoint namespace."""

    architecture_name = "OPTForCausalLM"

    @classmethod
    def compile(
        cls,
        *,
        model_id: ModelId,
        model_config: architecture.FfnSourceConfig,
        checkpoint_keys: Collection[str],
    ) -> ffn.FfnModelSpec:
        dtype_field = next(
            (field for field in ("torch_dtype", "dtype") if model_config.values.get(field) is not None), None
        )
        # OPTConfig defaults an absent activation_function to ReLU.
        activation_field = "activation_function" if "activation_function" in model_config.values else None
        model_config.validate_family_profile(
            model_type="opt", dtype_field=dtype_field, activation_field=activation_field, activation="relu"
        )
        pattern = re.compile(r"^(?P<prefix>(?:model\.)?decoder\.layers)\.\d+\.fc[12]\.")
        prefixes = {match.group("prefix") for key in checkpoint_keys if (match := pattern.match(key)) is not None}
        if len(prefixes) != 1:
            raise ValueError("OPT checkpoint must contain exactly one FFN decoder-layer namespace")
        prefix = prefixes.pop()
        enable_bias = model_config.optional("enable_bias", bool)
        bias = True if enable_bias is None else enable_bias
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("ffn_dim", int, ge=1)
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden_size,
            activation=ffn.ReluActivation(),
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layer_id=layer_id,
                    intermediate_size=intermediate_size,
                    projections=ffn.UngatedProjectionSpec(
                        up=model_config.projection(f"{prefix}.{layer_id}.fc1", bias=bias),
                        down=model_config.projection(f"{prefix}.{layer_id}.fc2", bias=bias),
                    ),
                )
                for layer_id in range(layer_count)
            ),
        )

    @classmethod
    def checkpoint_prefixes(cls, layer: ffn.FfnLayerSpec) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f"{key.rsplit('.', 1)[0]}." for key in layer.tensor_keys()))
