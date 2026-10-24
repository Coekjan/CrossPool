from __future__ import annotations

from pathlib import Path

import numpy
import pytest
from safetensors.numpy import save_file

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import checkpoint
from xtest.harness.support.config import TEST_MODEL_ID


def opt_spec(layer_ids: tuple[int, ...] = (0, 1)) -> ffn.FfnModelSpec:
    return ffn.FfnModelSpec(
        model_id=TEST_MODEL_ID,
        architecture_name="OPTForCausalLM",
        hidden_size=16,
        activation=ffn.ActivationKind.RELU,
        checkpoint_namespace="model.decoder.layers.{layer_id}",
        layers=tuple(
            ffn.DenseFfnSpec(
                kind=LayerKind.DENSE,
                layout=ffn.DenseFfnLayout.BIASED_RELU,
                layer_id=layer_id,
                intermediate_size=32,
                checkpoint=ffn.BiasedFfnCheckpointKeys(
                    fc1_weight_key=f"model.decoder.layers.{layer_id}.fc1.weight",
                    fc1_bias_key=f"model.decoder.layers.{layer_id}.fc1.bias",
                    fc2_weight_key=f"model.decoder.layers.{layer_id}.fc2.weight",
                    fc2_bias_key=f"model.decoder.layers.{layer_id}.fc2.bias",
                ),
            )
            for layer_id in layer_ids
        ),
    )


def opt_keys(layer_id: int) -> tuple[str, ...]:
    prefix = f"model.decoder.layers.{layer_id}"
    return (
        f"{prefix}.fc1.weight",
        f"{prefix}.fc1.bias",
        f"{prefix}.fc2.weight",
        f"{prefix}.fc2.bias",
    )


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


def test_opt_coverage_accepts_fc_keys_and_ignores_attention(tmp_path: Path) -> None:
    spec = opt_spec()
    keys = [key for layer_id in (0, 1) for key in opt_keys(layer_id)]
    keys.extend(("model.decoder.layers.0.self_attn.q_proj.weight", "model.decoder.layers.0.final_layer_norm.weight"))
    key_view = {key: tmp_path / "shard.safetensors" for key in keys}

    checkpoint.validate_ffn_coverage(spec, key_view, allow_trailing_ffn_layers=False)


def test_opt_coverage_rejects_missing_bias(tmp_path: Path) -> None:
    spec = opt_spec((0,))
    keys = opt_keys(0)[:-1]
    key_view = {key: tmp_path / "shard.safetensors" for key in keys}

    with pytest.raises(ValueError, match=r"missing=.*fc2\.bias"):
        checkpoint.validate_ffn_coverage(spec, key_view, allow_trailing_ffn_layers=False)


def test_opt_coverage_rejects_trailing_fc_keys(tmp_path: Path) -> None:
    spec = opt_spec((0,))
    keys = list(opt_keys(0))
    keys.append("model.decoder.layers.1.fc1.weight")
    key_view = {key: tmp_path / "shard.safetensors" for key in keys}

    with pytest.raises(ValueError, match="outside main decoder layers"):
        checkpoint.validate_ffn_coverage(spec, key_view, allow_trailing_ffn_layers=False)
