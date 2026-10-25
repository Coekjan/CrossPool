from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from xpool.ffn import (
    DenseFfnSpec,
    FfnModelSpec,
    MoeFfnSpec,
    RouterAlgorithm,
    RouterProjectionPrecision,
    TopKWeightRounding,
)
from xpool.quantization import Mxfp4, Unquantized
from xpool.runtime.ffnagent import architecture
from xpool.runtime.ffnagent.models.deepseek_v2 import DeepseekV2Adapter
from xpool.runtime.ffnagent.models.glm4_moe_lite import Glm4MoeLiteAdapter
from xpool.runtime.ffnagent.models.qwen3_moe import Qwen3MoeAdapter
from xtest.harness.support.config import TEST_MODEL_ID


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
        architecture.load(model_id=TEST_MODEL_ID, model_path=tmp_path / "model")


def test_load_rejects_duplicate_config_member(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b'{"architectures":[],"architectures":[]}')

    with pytest.raises(RuntimeError, match="duplicate JSON member 'architectures'"):
        architecture.load(model_id=TEST_MODEL_ID, model_path=model_path)


@pytest.mark.parametrize(
    ("dtype", "hidden_size", "intermediate_size", "layer_count"),
    (
        pytest.param("bfloat16", 16, 32, 3, id="small-bf16"),
        pytest.param("float16", 16, 32, 3, id="small-fp16"),
    ),
)
def test_load_compiles_llama_dense_layers(
    tmp_path: Path, dtype: str, hidden_size: int, intermediate_size: int, layer_count: int
) -> None:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_act": "silu",
        "torch_dtype": dtype,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layer_count,
    }
    keys = tuple(key for layer_id in range(layer_count) for key in dense_keys(layer_id))
    model_path = tmp_path / "model"
    write_indexed_model(model_path, config, keys)

    spec = architecture.load(model_id=TEST_MODEL_ID, model_path=model_path)

    assert spec.model_id == TEST_MODEL_ID
    assert spec.hidden_size == hidden_size
    assert len(spec.layers) == layer_count
    for layer_id, layer in enumerate(spec.layers):
        assert isinstance(layer, DenseFfnSpec)
        assert layer.layer_id == layer_id
        assert layer.intermediate_size == intermediate_size
        assert layer.tensor_keys() == dense_keys(layer_id)


@pytest.mark.parametrize("dtype_field", ("dtype", "torch_dtype"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_family_profile_accepts_admitted_dtypes(dtype_field: str, dtype: str) -> None:
    config = architecture.FfnSourceConfig({"model_type": "llama", "hidden_act": "silu", dtype_field: dtype})
    config.validate_family_profile(model_type="llama", dtype_field=dtype_field)


@pytest.mark.parametrize("dtype", ("float32", "half", "torch.float16", "int8", None, 16))
def test_family_profile_rejects_other_dtypes(dtype: object) -> None:
    config = architecture.FfnSourceConfig({"model_type": "llama", "hidden_act": "silu", "torch_dtype": dtype})
    with pytest.raises(ValueError, match="torch_dtype"):
        config.validate_family_profile(model_type="llama", dtype_field="torch_dtype")


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


def test_source_config_binds_quantization_exclusions_to_individual_projections() -> None:
    config = architecture.FfnSourceConfig(
        {
            "quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": ["down_proj"]},
        }
    )
    projections = config.gated_projections("model.layers.0.mlp")

    assert projections.gate.quantization == Mxfp4()
    assert projections.up.quantization == Mxfp4()
    assert projections.down.quantization == Unquantized()
    assert projections.tensor_keys() == dense_keys(0)


@pytest.mark.parametrize(
    "adapter, algorithm, precision, source",
    (
        (
            DeepseekV2Adapter,
            RouterAlgorithm.SOFTMAX,
            RouterProjectionPrecision.FP32_LOGITS,
            {
                "model_type": "deepseek_v2",
                "torch_dtype": "bfloat16",
                "scoring_func": "softmax",
                "topk_method": "greedy",
                "moe_layer_freq": 1,
            },
        ),
        (
            Glm4MoeLiteAdapter,
            RouterAlgorithm.SIGMOID,
            RouterProjectionPrecision.FP32,
            {"model_type": "glm4_moe_lite", "dtype": "bfloat16", "topk_method": "noaux_tc"},
        ),
        (
            Qwen3MoeAdapter,
            RouterAlgorithm.SOFTMAX,
            RouterProjectionPrecision.PAYLOAD,
            {"model_type": "qwen3_moe", "torch_dtype": "bfloat16", "decoder_sparse_step": 1, "mlp_only_layers": []},
        ),
    ),
)
def test_compilers_select_serializable_router_semantics(
    adapter: type[architecture.FfnModelAdapter],
    algorithm: RouterAlgorithm,
    precision: RouterProjectionPrecision,
    source: dict[str, object],
) -> None:
    config = architecture.FfnSourceConfig(
        {
            "hidden_act": "silu",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "num_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.8,
            **source,
        }
    )
    spec = adapter.compile(model_id=TEST_MODEL_ID, model_config=config)
    assert isinstance(spec.layers[0], MoeFfnSpec)
    assert spec.layers[0].router_algorithm is algorithm
    assert spec.layers[0].router_projection_precision is precision
    assert spec.layers[0].topk_weight_rounding is TopKWeightRounding.NONE
    assert (spec.layers[0].router_correction_bias_key is not None) == (algorithm is RouterAlgorithm.SIGMOID)
    assert FfnModelSpec.model_validate(spec.model_dump(mode="json")) == spec

    payload = spec.model_dump(mode="json")
    payload["layers"][0]["router_correction_bias_key"] = (
        None if algorithm is RouterAlgorithm.SIGMOID else "model.layers.0.mlp.gate.bias"
    )
    changed = FfnModelSpec.model_validate(payload)
    assert changed.digest() != spec.digest()
