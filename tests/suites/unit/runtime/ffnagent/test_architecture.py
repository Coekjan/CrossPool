from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from xpool.ffn import MoeFfnSpec
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


def dense_keys(layer_id: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_id}.mlp"
    return tuple(f"{prefix}.{projection}_proj.weight" for projection in ("gate", "up", "down"))


def moe_keys(layer_id: int, *, routed_experts: int, shared_experts: int, correction: bool) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_id}.mlp"
    keys = [f"{prefix}.gate.weight"]
    if correction:
        keys.append(f"{prefix}.gate.e_score_correction_bias")
    for expert_id in range(routed_experts):
        keys.extend(f"{prefix}.experts.{expert_id}.{projection}_proj.weight" for projection in ("gate", "up", "down"))
    if shared_experts:
        keys.extend(f"{prefix}.shared_experts.{projection}_proj.weight" for projection in ("gate", "up", "down"))
    return tuple(keys)


def write_indexed_model(path: Path, config: Mapping[str, object], keys: tuple[str, ...]) -> bytes:
    path.mkdir()
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    (path / "config.json").write_bytes(config_bytes)
    (path / "model-00001-of-00001.safetensors").write_bytes(b"")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "model-00001-of-00001.safetensors")}),
        encoding="utf-8",
    )
    return config_bytes


def common_config(*, architecture_name: str, model_type: str, dtype_field: str) -> dict[str, object]:
    return {
        "architectures": [architecture_name],
        "model_type": model_type,
        "hidden_act": "silu",
        dtype_field: "bfloat16",
        "hidden_size": 16,
        "num_hidden_layers": 3,
    }


@pytest.mark.parametrize("family", ["qwen3", "deepseek", "glm", "qwen3_moe"])
def test_load_compiles_exact_first_production_family(tmp_path: Path, family: str) -> None:
    if family == "qwen3":
        config = {
            **common_config(architecture_name="Qwen3ForCausalLM", model_type="qwen3", dtype_field="torch_dtype"),
            "intermediate_size": 32,
        }
        keys = tuple(key for layer_id in range(3) for key in dense_keys(layer_id))
    elif family == "deepseek":
        config = {
            **common_config(
                architecture_name="DeepseekV2ForCausalLM",
                model_type="deepseek_v2",
                dtype_field="torch_dtype",
            ),
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": False,
            "routed_scaling_factor": 1.0,
        }
        keys = dense_keys(0) + tuple(
            key
            for layer_id in (1, 2)
            for key in moe_keys(layer_id, routed_experts=2, shared_experts=1, correction=False)
        )
    elif family == "glm":
        config = {
            **common_config(
                architecture_name="Glm4MoeLiteForCausalLM",
                model_type="glm4_moe_lite",
                dtype_field="dtype",
            ),
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
            "topk_method": "noaux_tc",
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.8,
        }
        keys = dense_keys(0) + tuple(
            key
            for layer_id in (1, 2)
            for key in moe_keys(layer_id, routed_experts=2, shared_experts=1, correction=True)
        )
        keys += moe_keys(3, routed_experts=2, shared_experts=1, correction=True)
    else:
        config = {
            **common_config(
                architecture_name="Qwen3MoeForCausalLM",
                model_type="qwen3_moe",
                dtype_field="torch_dtype",
            ),
            "moe_intermediate_size": 8,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        }
        keys = tuple(
            key
            for layer_id in range(3)
            for key in moe_keys(layer_id, routed_experts=2, shared_experts=0, correction=False)
        )

    config_bytes = write_indexed_model(tmp_path / family, config, keys)
    spec = architecture.load(model_id=family, model_path=tmp_path / family)

    assert spec.model_config_digest == hashlib.sha256(config_bytes).hexdigest()
    assert spec.digest() == spec.model_copy().digest()
    assert (
        tuple(layer.kind for layer in spec.layers)
        == {
            "qwen3": (LayerKind.DENSE,) * 3,
            "deepseek": (LayerKind.DENSE, LayerKind.MOE, LayerKind.MOE),
            "glm": (LayerKind.DENSE, LayerKind.MOE, LayerKind.MOE),
            "qwen3_moe": (LayerKind.MOE,) * 3,
        }[family]
    )
    if family == "glm":
        assert isinstance(spec.layers[1], MoeFfnSpec)
        assert spec.layers[1].checkpoint.router_correction_bias_key is not None


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
