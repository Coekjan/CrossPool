from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM, PretrainedConfig
from sglang.srt.plugins.hook_registry import HookType

from tests.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from xpool.integrations.sglang.models.glm4_moe_lite import (
    Glm4MoeLiteShimAdapter,
    XpoolGlm4MoeLiteMLP,
    XpoolGlm4MoeLiteSparseMoeBlock,
    around_load_weights,
)
from xpool.integrations.sglang.shim import ShimUnavailableError
from xpool.native.ffn import LayerKind


def glm_config() -> PretrainedConfig:
    config = PretrainedConfig(architectures=["Glm4MoeLiteForCausalLM"])
    config.num_hidden_layers = 2
    config.hidden_size = 2048
    config.hidden_act = "silu"
    config.n_routed_experts = 8
    config.first_k_dense_replace = 1
    config.moe_layer_freq = 1
    return config


def test_glm_adapter_declares_exact_hooks_and_capability() -> None:
    adapter = Glm4MoeLiteShimAdapter()
    hooks = {(hook.target, hook.kind): hook.handler for hook in adapter.hooks()}

    assert adapter.supports_dp_attention
    assert hooks[("sglang.srt.models.glm4_moe_lite.Glm4MoeLiteMLP", HookType.REPLACE)] is XpoolGlm4MoeLiteMLP
    assert (
        hooks[("sglang.srt.models.glm4_moe_lite.Glm4MoeLiteSparseMoeBlock", HookType.REPLACE)]
        is XpoolGlm4MoeLiteSparseMoeBlock
    )
    assert ("sglang.srt.models.glm4_moe_lite.Glm4MoeLiteForCausalLM.load_weights", HookType.AROUND) in hooks


def test_glm_adapter_matches_only_exact_architecture() -> None:
    adapter = Glm4MoeLiteShimAdapter()

    assert adapter.matches(runner_with_architecture("Glm4MoeLiteForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Glm4ForCausalLM").as_model_runner())


def test_glm_replacements_are_parameter_free_and_preserve_layer_kinds() -> None:
    dense = XpoolGlm4MoeLiteMLP(2048, 8192, "silu", prefix="model.layers.0.mlp")
    sparse = XpoolGlm4MoeLiteSparseMoeBlock(glm_config(), 1)

    assert list(dense.parameters()) == []
    assert list(sparse.parameters()) == []
    assert dense.layer_kind is LayerKind.DENSE
    assert sparse.layer_kind is LayerKind.MOE
    assert not hasattr(sparse, "experts")
    assert sparse.get_moe_weights() == []


def test_glm_weight_loader_filters_ffn_and_rejects_nextn() -> None:
    observed: list[str] = []

    def original(
        model: Glm4MoeLiteForCausalLM,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_nextn: bool,
        params_dict: dict[str, torch.nn.Parameter] | None,
    ) -> None:
        observed.extend(name for name, tensor in weights)

    weights = [
        ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
    ]
    model = loaded_model(Glm4MoeLiteForCausalLM, glm_config(), [])

    around_load_weights(original, model, weights)
    assert observed == ["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(ShimUnavailableError, match="next-token draft"):
        around_load_weights(original, model, weights, is_nextn=True)


def test_glm_loaded_model_uses_effective_config_policy() -> None:
    config = glm_config()
    model = loaded_model(
        Glm4MoeLiteForCausalLM,
        config,
        [
            FakeDecoderLayer(
                XpoolGlm4MoeLiteMLP(2048, 8192, "silu", prefix="model.layers.0.mlp"),
                allow_reduce_scatter=True,
            ),
            FakeDecoderLayer(XpoolGlm4MoeLiteSparseMoeBlock(config, 1), allow_reduce_scatter=True),
        ],
    )
    runner = runner_with_architecture("Glm4MoeLiteForCausalLM")
    runner.model = model

    Glm4MoeLiteShimAdapter().validate_after_load(runner.as_model_runner())

    assert runner.xpool_ffn_shim_count == 2
