from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from transformers import LlamaConfig

from xpool.integrations.sglang.models.llama import (
    LlamaShimAdapter,
    XpoolLlamaMLP,
    around_load_weights,
    llama_decoder_forward,
)
from xtest.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture


def test_llama_adapter_declares_dense_sglang_hooks() -> None:
    hooks = {(hook.target, hook.kind): hook.handler for hook in LlamaShimAdapter().hooks()}

    assert hooks[("sglang.srt.models.llama.LlamaMLP", HookType.REPLACE)] is XpoolLlamaMLP
    assert hooks[("sglang.srt.models.llama.LlamaDecoderLayer.forward", HookType.REPLACE)] is llama_decoder_forward
    assert hooks[("sglang.srt.models.llama.LlamaForCausalLM.load_weights", HookType.AROUND)] is around_load_weights


def test_llama_adapter_matches_only_llama_architecture() -> None:
    adapter = LlamaShimAdapter()

    assert adapter.matches(runner_with_architecture("LlamaForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen2ForCausalLM").as_model_runner())


def test_llama_loaded_model_requires_complete_dense_shims() -> None:
    config = LlamaConfig(architectures=["LlamaForCausalLM"], num_hidden_layers=2)
    model = loaded_model(
        LlamaForCausalLM,
        config,
        [
            FakeDecoderLayer(
                XpoolLlamaMLP(4096, 11008, "silu", prefix="model.layers.0.mlp"),
                allow_reduce_scatter=False,
            ),
            FakeDecoderLayer(
                XpoolLlamaMLP(4096, 11008, "silu", prefix="model.layers.1.mlp"),
                allow_reduce_scatter=False,
            ),
        ],
    )
    runner = runner_with_architecture("LlamaForCausalLM")
    runner.model = model

    LlamaShimAdapter().validate_after_load(runner.as_model_runner())


def test_llama_decoder_preserves_fp16_attention_and_norm_order() -> None:
    events: list[str] = []
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]], dtype=torch.float16)
    forward_batch = object()

    def input_norm(
        states: torch.Tensor, residual: torch.Tensor | None = None, *, quant_linear: object
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        events.append("input_norm")
        return states if residual is None else (states, residual)

    def attention(*, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: object) -> torch.Tensor:
        events.append("attention")
        return hidden_states + 1

    def post_norm(
        states: torch.Tensor, residual: torch.Tensor, *, quant_linear: object | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    original = SimpleNamespace(
        input_layernorm=input_norm,
        self_attn=original_attention,
        post_attention_layernorm=post_norm,
        mlp=original_ffn,
    )
    replacement = SimpleNamespace(
        input_layernorm=input_norm,
        self_attn=original.self_attn,
        post_attention_layernorm=post_norm,
        mlp=shim_mlp,
    )

    original_result = LlamaDecoderLayer.forward(
        cast(LlamaDecoderLayer, original), positions, hidden_states, cast(ForwardBatch, forward_batch), None
    )
    original_events = tuple(events)
    events.clear()
    replacement_result = llama_decoder_forward(
        cast(LlamaDecoderLayer, replacement), positions, hidden_states, cast(ForwardBatch, forward_batch), None
    )

    assert all(torch.equal(left, right) for left, right in zip(original_result, replacement_result, strict=True))
    assert tuple(events) == original_events == ("input_norm", "attention", "post_norm", "mlp")
