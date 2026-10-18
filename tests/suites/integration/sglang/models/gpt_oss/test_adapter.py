from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.models.gpt_oss import GptOssConfig, GptOssForCausalLM

from xpool.integrations.sglang.models.gpt_oss import GptOssShimAdapter, XpoolGptOssSparseMoeBlock, around_load_weights
from xpool.integrations.sglang.shim import ShimUnavailableError
from xtest.harness.support.sglang.fakes import loaded_model, runner_with_architecture


def test_gpt_oss_shim_filters_packed_scales_and_all_biases() -> None:
    config = GptOssConfig(hidden_size=2880, num_hidden_layers=2)
    quantization = Mxfp4Config.from_config({"quant_method": "mxfp4"})
    shim = XpoolGptOssSparseMoeBlock(0, config, quant_config=quantization)
    assert list(shim.parameters()) == [] and shim.get_moe_weights() == []
    assert not hasattr(shim, "experts") and not hasattr(shim, "router")
    model = loaded_model(GptOssForCausalLM, config, [])
    observed = []

    def original(
        model: GptOssForCausalLM,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        is_nextn: bool,
        weight_name_mapping: dict[str, str] | None,
    ) -> None:
        observed.extend(name for name, _ in weights)

    tensors = [
        (f"model.layers.0.mlp.{suffix}", torch.empty(1))
        for suffix in (
            "router.weight",
            "router.bias",
            "experts.gate_up_proj_blocks",
            "experts.gate_up_proj_scales",
            "experts.gate_up_proj_bias",
            "experts.down_proj_blocks",
            "experts.down_proj_scales",
            "experts.down_proj_bias",
        )
    ]
    tensors.append(("model.layers.0.self_attn.sinks", torch.empty(1)))
    around_load_weights(original, model, tensors)
    assert observed == ["model.layers.0.self_attn.sinks"]
    with pytest.raises(ShimUnavailableError, match="nextn"):
        around_load_weights(original, model, tensors, is_nextn=True)
    adapter = GptOssShimAdapter()
    assert adapter.matches(runner_with_architecture("GptOssForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen3MoeForCausalLM").as_model_runner())
