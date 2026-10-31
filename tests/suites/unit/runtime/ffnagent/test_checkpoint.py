from __future__ import annotations

from pathlib import Path

import numpy
import pytest
from safetensors.numpy import save_file
from xpool.native.ffn import LayerKind

from xpool import ffn
from xpool.model import ModelId
from xpool.quantization import Unquantized
from xpool.runtime.ffnagent import checkpoint


def unary_spec() -> ffn.FfnModelSpec:
    """Return one OPT-shaped biased unary Model Spec."""

    return ffn.FfnModelSpec(
        model_id=ModelId("facebook/opt-test"),
        hidden_size=16,
        activation=ffn.ReluActivation(),
        checkpoint_namespace="model.decoder.layers.{layer_id}",
        layers=(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=32,
                projections=ffn.UnaryProjectionSpec(
                    fc1=ffn.ProjectionSpec(
                        quantization=Unquantized(),
                        values_key="model.decoder.layers.0.fc1.weight",
                        bias_key="model.decoder.layers.0.fc1.bias",
                    ),
                    fc2=ffn.ProjectionSpec(
                        quantization=Unquantized(),
                        values_key="model.decoder.layers.0.fc2.weight",
                        bias_key="model.decoder.layers.0.fc2.bias",
                    ),
                ),
            ),
        ),
    )


def unary_key_view(*extra: str) -> dict[str, Path]:
    keys = {
        "model.decoder.layers.0.fc1.weight",
        "model.decoder.layers.0.fc1.bias",
        "model.decoder.layers.0.fc2.weight",
        "model.decoder.layers.0.fc2.bias",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.self_attn_layer_norm.weight",
    }
    keys.update(extra)
    return {key: Path("model.safetensors") for key in keys}


def test_unary_ffn_coverage_ignores_non_ffn_layer_keys() -> None:
    checkpoint.validate_ffn_coverage(unary_spec(), unary_key_view(), allow_trailing_ffn_layers=False)


def test_unary_ffn_coverage_rejects_missing_bias_and_extra_fc_key() -> None:
    missing = unary_key_view()
    del missing["model.decoder.layers.0.fc1.bias"]
    with pytest.raises(ValueError, match=r"missing=.*fc1\.bias"):
        checkpoint.validate_ffn_coverage(unary_spec(), missing, allow_trailing_ffn_layers=False)

    extra = unary_key_view("model.decoder.layers.0.fc1.unowned")
    with pytest.raises(ValueError, match=r"extra=.*unowned"):
        checkpoint.validate_ffn_coverage(unary_spec(), extra, allow_trailing_ffn_layers=False)


def test_unary_ffn_coverage_rejects_trailing_layer_fc_key() -> None:
    trailing = unary_key_view("model.decoder.layers.1.fc1.weight")
    with pytest.raises(ValueError, match="outside main decoder layers"):
        checkpoint.validate_ffn_coverage(unary_spec(), trailing, allow_trailing_ffn_layers=False)


def test_single_file_checkpoint_uses_safetensors_header(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    save_file({"model.layers.0.mlp.gate_proj.weight": numpy.zeros((1,), dtype=numpy.float32)}, checkpoint_path)

    assert checkpoint.read_checkpoint_key_view(tmp_path) == {
        "model.layers.0.mlp.gate_proj.weight": checkpoint_path,
    }


def test_unindexed_checkpoint_rejects_ambiguous_files(tmp_path: Path) -> None:
    for name in ("model.safetensors", "other.safetensors"):
        save_file({name: numpy.zeros((1,), dtype=numpy.float32)}, tmp_path / name)

    with pytest.raises(ValueError, match="exactly one unindexed canonical"):
        checkpoint.read_checkpoint_key_view(tmp_path)
