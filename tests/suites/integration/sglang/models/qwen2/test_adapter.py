from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from transformers import Qwen2Config

from tests.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture
from xpool.integrations.sglang.models.qwen2 import (
    Qwen2ShimAdapter,
    XpoolQwen2MLP,
    around_load_weights,
    qwen2_decoder_forward,
)


def test_qwen2_adapter_declares_shared_mlp_and_decoder_hooks() -> None:
    hooks = {(hook.target, hook.kind): hook.handler for hook in Qwen2ShimAdapter().hooks()}

    assert hooks[("sglang.srt.models.qwen2.Qwen2MLP", HookType.REPLACE)] is XpoolQwen2MLP
    assert hooks[("sglang.srt.models.qwen2.Qwen2DecoderLayer.forward", HookType.REPLACE)] is qwen2_decoder_forward
    assert hooks[("sglang.srt.models.qwen2.Qwen2ForCausalLM.load_weights", HookType.AROUND)] is around_load_weights


def test_qwen2_and_qwen3_hooks_install_one_shared_mlp_class() -> None:
    script = """
from sglang.srt.plugins.hook_registry import HookRegistry
from xpool.integrations.sglang.models.qwen2 import Qwen2ShimAdapter, XpoolQwen2MLP
from xpool.integrations.sglang.models.qwen3 import Qwen3ShimAdapter

for adapter in (Qwen2ShimAdapter(), Qwen3ShimAdapter()):
    for hook in adapter.hooks():
        HookRegistry.register(hook.target, hook.handler, hook.kind)
HookRegistry.apply_hooks()

from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.models.qwen3 import Qwen3MLP

assert Qwen2MLP is Qwen3MLP is XpoolQwen2MLP
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_qwen2_adapter_matches_only_qwen2_architecture() -> None:
    adapter = Qwen2ShimAdapter()

    assert adapter.matches(runner_with_architecture("Qwen2ForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen3ForCausalLM").as_model_runner())


def test_qwen2_loaded_model_requires_complete_dense_shims() -> None:
    config = Qwen2Config(architectures=["Qwen2ForCausalLM"], num_hidden_layers=2)
    model = loaded_model(
        Qwen2ForCausalLM,
        config,
        [
            FakeDecoderLayer(XpoolQwen2MLP(896, 4864, "silu", prefix="model.layers.0.mlp"), allow_reduce_scatter=False),
            FakeDecoderLayer(XpoolQwen2MLP(896, 4864, "silu", prefix="model.layers.1.mlp"), allow_reduce_scatter=False),
        ],
    )
    runner = runner_with_architecture("Qwen2ForCausalLM")
    runner.model = model

    Qwen2ShimAdapter().validate_after_load(runner.as_model_runner())


def test_qwen2_decoder_preserves_bf16_attention_and_norm_order() -> None:
    events: list[str] = []
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]], dtype=torch.bfloat16)
    forward_batch = cast(ForwardBatch, object())

    def input_norm(states: torch.Tensor, residual: torch.Tensor | None = None, *, quant_linear: object) -> object:
        events.append("input_norm")
        return states if residual is None else (states, residual)

    def attention(*, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: object) -> torch.Tensor:
        events.append("attention")
        return hidden_states + 1

    def post_norm(states: torch.Tensor, residual: torch.Tensor, *, quant_linear: object | None = None) -> object:
        events.append("post_norm")
        return states + 1, residual

    def original_mlp(states: torch.Tensor) -> torch.Tensor:
        events.append("mlp")
        return states + 1

    def shim_mlp(states: torch.Tensor, *, forward_batch: object) -> torch.Tensor:
        assert forward_batch is expected_batch
        events.append("mlp")
        return states + 1

    expected_batch = forward_batch
    original_attention = Mock(side_effect=attention)
    original_attention.qkv_proj = object()
    original_ffn = Mock(side_effect=original_mlp)
    original_ffn.gate_up_proj = object()
    original = cast(
        Qwen2DecoderLayer,
        SimpleNamespace(
            input_layernorm=input_norm,
            self_attn=original_attention,
            post_attention_layernorm=post_norm,
            mlp=original_ffn,
        ),
    )
    replacement = cast(
        Qwen2DecoderLayer,
        SimpleNamespace(
            input_layernorm=input_norm,
            self_attn=original.self_attn,
            post_attention_layernorm=post_norm,
            mlp=shim_mlp,
        ),
    )

    original_result = Qwen2DecoderLayer.forward(original, positions, hidden_states, forward_batch, None)
    original_events = tuple(events)
    events.clear()
    replacement_result = qwen2_decoder_forward(replacement, positions, hidden_states, forward_batch, None)

    assert all(torch.equal(left, right) for left, right in zip(original_result, replacement_result, strict=True))
    assert tuple(events) == original_events == ("input_norm", "attention", "post_norm", "mlp")
