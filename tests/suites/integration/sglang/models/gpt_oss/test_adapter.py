from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from sglang.srt.models.gpt_oss import GptOssForCausalLM
from transformers import GptOssConfig

from xpool.integrations.sglang.models.gpt_oss import (
    GptOssShimAdapter,
    XpoolGptOssSparseMoeBlock,
    around_load_weights,
)
from xpool.integrations.sglang.models.qwen3 import Qwen3ShimAdapter
from xpool.integrations.sglang.shim import ShimUnavailableError
from xpool.native.ffn import LayerKind
from xtest.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture


def test_gpt_oss_keeps_parameter_free_shims_and_original_full_boundaries() -> None:
    config = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=96)
    shims = [XpoolGptOssSparseMoeBlock(layer_id, config) for layer_id in range(2)]
    model = loaded_model(
        GptOssForCausalLM,
        config,
        [FakeDecoderLayer(shim, allow_reduce_scatter=True) for shim in shims],
    )
    runner = runner_with_architecture("GptOssForCausalLM")
    runner.model = model
    runner.model_config.quantization = "mxfp4"
    adapter = GptOssShimAdapter()

    assert adapter.matches(runner.as_model_runner())
    adapter.validate_before_load(runner.as_model_runner())
    adapter.validate_after_load(runner.as_model_runner())
    assert all(shim.layer_kind is LayerKind.MOE and not tuple(shim.parameters()) for shim in shims)
    assert all(shim.get_moe_weights() == [] for shim in shims)


def test_quantization_admission_uses_the_resolved_model_and_adapter() -> None:
    runner = runner_with_architecture("GptOssForCausalLM")
    with pytest.raises(ShimUnavailableError, match="original MXFP4"):
        GptOssShimAdapter().validate_before_load(runner.as_model_runner())
    runner.model_config.quantization = "mxfp4"
    GptOssShimAdapter().validate_before_load(runner.as_model_runner())
    with pytest.raises(ShimUnavailableError, match="unquantized"):
        Qwen3ShimAdapter().validate_before_load(runner.as_model_runner())


def test_gpt_oss_loader_keeps_attention_sinks_and_mapping_but_no_ffn_tensors() -> None:
    retained: list[str] = []
    mappings: list[dict[str, str] | None] = []

    def original(
        model: GptOssForCausalLM,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_nextn: bool,
        weight_name_mapping: dict[str, str] | None,
    ) -> None:
        retained.extend(name for name, tensor in weights)
        mappings.append(weight_name_mapping)

    model = loaded_model(GptOssForCausalLM, GptOssConfig(), [])
    names = (
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.down_proj_bias",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.sinks",
        "model.embed_tokens.weight",
        "lm_head.weight",
    )
    weights = tuple((name, torch.empty(1)) for name in names)
    mapping = {"source": "target"}

    around_load_weights(original, model, iter(weights), weight_name_mapping=mapping)

    assert retained == list(names[-4:])
    assert mappings == [mapping]
    with pytest.raises(ShimUnavailableError, match="MTP"):
        around_load_weights(original, model, iter(weights), is_nextn=True)
