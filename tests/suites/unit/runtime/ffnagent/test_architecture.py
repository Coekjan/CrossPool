from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from xpool.runtime.ffnagent import architecture


def dense_keys(layer_id: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_id}.mlp"
    return tuple(f"{prefix}.{projection}_proj.weight" for projection in ("gate", "up", "down"))


def write_indexed_model(path: Path, config: Mapping[str, object], keys: tuple[str, ...]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "model-00001-of-00001.safetensors")}),
        encoding="utf-8",
    )


def common_config(*, architecture_name: str, model_type: str, dtype_field: str) -> dict[str, object]:
    return {
        "architectures": [architecture_name],
        "model_type": model_type,
        "hidden_act": "silu",
        dtype_field: "bfloat16",
        "hidden_size": 16,
        "num_hidden_layers": 3,
    }


def test_load_rejects_extra_main_layer_ffn_key(tmp_path: Path) -> None:
    config = {
        **common_config(architecture_name="Qwen3ForCausalLM", model_type="qwen3", dtype_field="torch_dtype"),
        "intermediate_size": 32,
    }
    keys = tuple(key for layer_id in range(3) for key in dense_keys(layer_id))
    write_indexed_model(tmp_path / "model", config, (*keys, "model.layers.0.mlp.unowned.weight"))

    with pytest.raises(RuntimeError, match=r"extra=.*unowned"):
        architecture.load(model_id="model", model_path=tmp_path / "model")


def test_load_rejects_duplicate_config_member(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b'{"architectures":[],"architectures":[]}')

    with pytest.raises(RuntimeError, match="duplicate JSON member 'architectures'"):
        architecture.load(model_id="model", model_path=model_path)


def test_source_config_reads_strict_types_and_constraints() -> None:
    config = architecture.FfnSourceConfig(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 16,
            "scale": 1,
            "optional": None,
        }
    )

    assert config.architecture_name == "Qwen3ForCausalLM"
    assert config.get("hidden_size", int, ge=1) == 16
    assert config.get("scale", float, gt=0) == 1.0
    assert config.optional("optional", int, allow_none=True) is None

    with pytest.raises(ValueError, match="hidden_size must be greater than or equal to 17"):
        config.get("hidden_size", int, ge=17)
    with pytest.raises(ValueError, match="missing must be a JSON integer"):
        config.get("missing", int)
    with pytest.raises(ValueError, match="optional must be a JSON integer"):
        config.optional("optional", int)
