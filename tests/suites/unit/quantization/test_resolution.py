from __future__ import annotations

import pytest

from xpool.quantization import Mxfp4, Unquantized, resolve


def test_resolution_uses_source_exclusions_without_model_identity() -> None:
    config = {"quant_method": "mxfp4", "modules_to_not_convert": ["gate", r"model\.layers\.\d+\.self_attn"]}
    assert resolve(None, module_name="model.layers.0.mlp.experts") == Unquantized()
    assert resolve(config, module_name="model.layers.0.mlp.experts") == Mxfp4()
    assert resolve(config, module_name="model.layers.0.mlp.gate") == Unquantized()
    assert resolve(config, module_name="model.layers.5.self_attn.q_proj") == Unquantized()


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "other", "modules_to_not_convert": ["gate"]},
        {"quant_method": "mxfp4", "modules_to_not_convert": "gate"},
        {"quant_method": "mxfp4", "modules_to_not_convert": [1]},
    ],
)
def test_resolution_rejects_unsupported_or_malformed_metadata(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        resolve(config, module_name="model.layers.0.mlp.gate")
