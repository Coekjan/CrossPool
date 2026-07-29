from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from transformers import Qwen3MoeConfig

from tests.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.models.qwen3_moe import (
    Qwen3MoeAdapter,
    XpoolQwen3MoeSparseMoeBlock,
    around_load_weights,
)
from xpool.integrations.sglang.shim import ShimUnavailableError


def qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=2,
        hidden_size=2048,
        hidden_act="silu",
    )


def test_qwen3_moe_adapter_declares_exact_hooks_and_capability() -> None:
    adapter = Qwen3MoeAdapter()
    hooks = {(hook.target, hook.kind): hook.handler for hook in adapter.hooks()}

    assert adapter.supports_dp_attention
    assert (
        hooks[("sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock", HookType.REPLACE)] is XpoolQwen3MoeSparseMoeBlock
    )
    assert ("sglang.srt.models.qwen3_moe.Qwen3MoeForCausalLM.load_weights", HookType.AROUND) in hooks


def test_qwen3_moe_adapter_matches_only_exact_architecture() -> None:
    adapter = Qwen3MoeAdapter()

    assert adapter.matches(runner_with_architecture("Qwen3MoeForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen3ForCausalLM").as_model_runner())


def test_qwen3_moe_replacement_is_parameter_free_sparse_shim() -> None:
    shim = XpoolQwen3MoeSparseMoeBlock(0, qwen3_moe_config())

    assert list(shim.parameters()) == []
    assert shim.layer_kind is FfnLayerKind.SPARSE
    assert not hasattr(shim, "experts")
    assert shim.get_moe_weights() == []


def test_qwen3_moe_weight_loader_filters_ffn_and_rejects_mtp() -> None:
    observed: list[str] = []

    def original(
        model: Qwen3MoeForCausalLM,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_mtp: bool,
    ) -> None:
        observed.extend(name for name, tensor in weights)

    weights = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.empty(1)),
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
    ]
    model = loaded_model(Qwen3MoeForCausalLM, qwen3_moe_config(), [])

    around_load_weights(original, model, weights)
    assert observed == ["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(ShimUnavailableError, match="MTP"):
        around_load_weights(original, model, weights, is_mtp=True)


def test_qwen3_moe_loaded_model_requires_all_sparse_full_boundaries() -> None:
    config = qwen3_moe_config()
    model = loaded_model(
        Qwen3MoeForCausalLM,
        config,
        [
            FakeDecoderLayer(XpoolQwen3MoeSparseMoeBlock(0, config), allow_reduce_scatter=True),
            FakeDecoderLayer(XpoolQwen3MoeSparseMoeBlock(1, config), allow_reduce_scatter=True),
        ],
    )
    runner = runner_with_architecture("Qwen3MoeForCausalLM")
    runner.model = model

    Qwen3MoeAdapter().validate_after_load(runner.as_model_runner())

    assert runner.xpool_ffn_shim_count == 2
