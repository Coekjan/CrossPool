from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from tests.harness.sglang.fakes import FakeDecoderLayer, runner_with_architecture
from xpool.integrations.sglang.models.qwen3 import (
    Qwen3Adapter,
    XpoolQwen3MLP,
    filter_ffn_weights,
)


def qwen_shim(layer_id: int) -> XpoolQwen3MLP:
    return XpoolQwen3MLP(
        hidden_size=5120,
        intermediate_size=17408,
        hidden_act="silu",
        prefix=f"model.layers.{layer_id}.mlp",
    )


def loaded_qwen_model(*, mlp_mode: ScatterMode = ScatterMode.FULL) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            FakeDecoderLayer(qwen_shim(0), mlp_mode=mlp_mode, allow_reduce_scatter=False),
            FakeDecoderLayer(qwen_shim(1), mlp_mode=mlp_mode, allow_reduce_scatter=False),
        ]
    )
    return model


def test_qwen3_adapter_declares_dense_sglang_hooks() -> None:
    hooks = {(hook.target, hook.kind): hook.handler for hook in Qwen3Adapter().hooks()}

    assert hooks[("sglang.srt.models.qwen3.Qwen3MLP", HookType.REPLACE)] is XpoolQwen3MLP
    assert ("sglang.srt.models.qwen3.Qwen3ForCausalLM.load_weights", HookType.AROUND) in hooks


def test_qwen3_adapter_matches_only_dense_qwen3_architecture() -> None:
    adapter = Qwen3Adapter()

    assert adapter.matches(runner_with_architecture("Qwen3ForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen3MoeForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("DeepseekV2ForCausalLM").as_model_runner())


def test_qwen3_weight_filter_removes_only_decoder_mlp_tensors() -> None:
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.empty(1)),
        ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ("model.layers.1.mlp.down_proj.weight", torch.empty(1)),
        ("model.norm.weight", torch.empty(1)),
    ]

    assert [name for name, tensor in filter_ffn_weights(weights)] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]


def test_qwen3_loaded_model_requires_full_non_reduce_scatter_boundary() -> None:
    runner = runner_with_architecture("Qwen3ForCausalLM")
    runner.model = loaded_qwen_model()

    Qwen3Adapter().validate_after_load(runner.as_model_runner())

    assert runner.xpool_ffn_shim_count == 2


def test_qwen3_loaded_model_rejects_non_full_mlp_boundary() -> None:
    runner = runner_with_architecture("Qwen3ForCausalLM")
    runner.model = loaded_qwen_model(mlp_mode=ScatterMode.TP_ATTN_FULL)

    with pytest.raises(RuntimeError, match=r"requires ScatterMode\.FULL"):
        Qwen3Adapter().validate_after_load(runner.as_model_runner())
