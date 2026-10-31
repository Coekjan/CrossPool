from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.opt import OPTDecoderLayer, OPTForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from torch import nn
from transformers import OPTConfig

from xpool.integrations.sglang.models import opt as opt_module
from xpool.integrations.sglang.models.opt import (
    OPTShimAdapter,
    XpoolOPTDecoderLayer,
    XpoolOPTFFN,
    around_load_weights,
    opt_decoder_forward,
    required_attention_backend,
)
from xpool.integrations.sglang.shim import ShimUnavailableError
from xtest.harness.support.sglang.fakes import FakeDecoderLayer, loaded_model, runner_with_architecture


def opt_config(*, do_layer_norm_before: bool = True) -> OPTConfig:
    return OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        word_embed_proj_dim=16,
        activation_function="relu",
        enable_bias=True,
        do_layer_norm_before=do_layer_norm_before,
    )


def test_opt_adapter_declares_layer_and_loader_hooks() -> None:
    hooks = {(hook.target, hook.kind): hook.handler for hook in OPTShimAdapter().hooks()}

    assert hooks[("sglang.srt.models.opt.OPTDecoderLayer", HookType.REPLACE)] is XpoolOPTDecoderLayer
    assert hooks[("sglang.srt.models.opt.OPTForCausalLM.load_weights", HookType.AROUND)] is around_load_weights


def test_opt_hooks_replace_the_decoder_layer_class() -> None:
    script = """
from sglang.srt.plugins.hook_registry import HookRegistry
from xpool.integrations.sglang.models.opt import OPTShimAdapter, XpoolOPTDecoderLayer

for hook in OPTShimAdapter().hooks():
    HookRegistry.register(hook.target, hook.handler, hook.kind)
HookRegistry.apply_hooks()

from sglang.srt.models.opt import OPTDecoderLayer

assert OPTDecoderLayer is XpoolOPTDecoderLayer
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_opt_adapter_matches_only_opt_architecture() -> None:
    adapter = OPTShimAdapter()

    assert adapter.matches(runner_with_architecture("OPTForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen2ForCausalLM").as_model_runner())


@pytest.mark.parametrize(
    ("hidden_size", "num_attention_heads", "expected_backend"),
    ((2560, 32, "triton"), (2048, 32, None)),
)
def test_opt_adapter_requires_backend_for_unsupported_head_dim(
    hidden_size: int,
    num_attention_heads: int,
    expected_backend: str | None,
) -> None:
    config = OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=hidden_size,
        ffn_dim=4 * hidden_size,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        word_embed_proj_dim=hidden_size,
        activation_function="relu",
        enable_bias=True,
        do_layer_norm_before=True,
    )

    assert required_attention_backend(config) == expected_backend


def test_opt_weight_filter_drops_only_inline_ffn_parameters() -> None:
    names = (
        "model.decoder.layers.0.fc1.weight",
        "model.decoder.layers.0.fc1.bias",
        "model.decoder.layers.0.fc2.weight",
        "model.decoder.layers.0.fc2.bias",
        "decoder.layers.1.fc1.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.final_layer_norm.weight",
        "model.decoder.project_in.weight",
        "model.decoder.embed_tokens.weight",
        "lm_head.weight",
    )
    seen: list[str] = []

    model = nn.Module()
    language_model = nn.Module()
    decoder = nn.Module()
    language_model.decoder = decoder
    model.model = language_model
    model.register_parameter("weight", nn.Parameter(torch.empty(0)))
    assert model.weight.requires_grad

    def original(_model: object, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        seen.extend(name for name, _ in weights)

    around_load_weights(original, cast(OPTForCausalLM, model), ((name, torch.empty(0)) for name in names))

    assert "model.decoder.layers.0.fc1.weight" not in seen
    assert "model.decoder.layers.0.fc2.bias" not in seen
    assert "decoder.layers.1.fc1.weight" not in seen
    assert "model.decoder.layers.0.self_attn.q_proj.weight" in seen
    assert "model.decoder.layers.0.final_layer_norm.weight" in seen
    assert "model.decoder.project_in.weight" in seen
    assert "model.decoder.embed_tokens.weight" in seen
    assert "lm_head.weight" in seen
    assert not model.weight.requires_grad
    assert language_model.model is decoder
    assert "model" not in dict(language_model.named_children())


def test_opt_adapter_accepts_complete_dense_shims() -> None:
    config = opt_config()
    model = loaded_model(
        OPTForCausalLM,
        config,
        [
            FakeDecoderLayer(XpoolOPTFFN(layer_id=0, config=config), allow_reduce_scatter=False),
            FakeDecoderLayer(XpoolOPTFFN(layer_id=1, config=config), allow_reduce_scatter=False),
        ],
    )
    runner = runner_with_architecture("OPTForCausalLM")
    runner.model = model

    OPTShimAdapter().validate_after_load(runner.as_model_runner())


def test_opt_adapter_rejects_missing_shim_and_leftover_ffn_parameters() -> None:
    config = opt_config()
    incomplete = loaded_model(
        OPTForCausalLM,
        config,
        [FakeDecoderLayer(XpoolOPTFFN(layer_id=0, config=config), allow_reduce_scatter=False)],
    )
    runner = runner_with_architecture("OPTForCausalLM")
    runner.model = incomplete
    with_missing = OPTShimAdapter()
    try:
        with_missing.validate_after_load(runner.as_model_runner())
    except RuntimeError as error:
        assert "requires FFN shim layer ids" in str(error)
    else:
        raise AssertionError("missing OPT shim must fail validation")

    leftover = loaded_model(
        OPTForCausalLM,
        config,
        [
            FakeDecoderLayer(XpoolOPTFFN(layer_id=0, config=config), allow_reduce_scatter=False),
            FakeDecoderLayer(XpoolOPTFFN(layer_id=1, config=config), allow_reduce_scatter=False),
        ],
    )
    leftover.add_module("fc1", nn.Linear(1, 1))
    runner.model = leftover
    try:
        OPTShimAdapter().validate_after_load(runner.as_model_runner())
    except RuntimeError as error:
        assert "left local FFN parameters" in str(error)
    else:
        raise AssertionError("leftover OPT FFN parameter must fail validation")


def test_opt_shim_rejects_unsupported_activation_and_bias() -> None:
    config = opt_config()
    config.activation_function = "gelu"
    with pytest.raises(ShimUnavailableError, match="only relu"):
        XpoolOPTFFN(layer_id=0, config=config)

    config = opt_config()
    config.enable_bias = False
    with pytest.raises(ShimUnavailableError, match="enable_bias=True"):
        XpoolOPTFFN(layer_id=0, config=config)


def test_opt_decoder_layer_omits_local_ffn_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAttention(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

    monkeypatch.setattr(opt_module, "OPTAttention", FakeAttention)
    config = opt_config()
    layer = XpoolOPTDecoderLayer(config, layer_id=1, prefix="model.layers.1")

    assert isinstance(layer.mlp, XpoolOPTFFN)
    assert not hasattr(layer, "fc1")
    assert not hasattr(layer, "fc2")
    assert not hasattr(layer, "activation_fn")
    assert list(layer.mlp.parameters()) == []


def test_opt_decoder_preserves_pre_ln_and_post_ln_order() -> None:
    for do_layer_norm_before in (True, False):
        events: list[str] = []
        hidden_states = torch.tensor([[1.0]], dtype=torch.float16)
        forward_batch = cast(ForwardBatch, object())

        def input_norm(states: torch.Tensor) -> torch.Tensor:
            events.append("attn_norm")
            return states + 1

        def final_norm(states: torch.Tensor) -> torch.Tensor:
            events.append("final_norm")
            return states + 1

        def attention(*, hidden_states: torch.Tensor, forward_batch: object) -> torch.Tensor:
            assert forward_batch is expected_batch
            events.append("attention")
            return hidden_states + 1

        def mlp(states: torch.Tensor, *, forward_batch: object) -> torch.Tensor:
            assert forward_batch is expected_batch
            events.append("mlp")
            return states + 3

        def fc1(states: torch.Tensor) -> tuple[torch.Tensor, None]:
            events.append("fc1")
            return states + 1, None

        def activation(states: torch.Tensor) -> torch.Tensor:
            events.append("activation")
            return states + 1

        def fc2(states: torch.Tensor) -> tuple[torch.Tensor, None]:
            events.append("fc2")
            return states + 1, None

        expected_batch = forward_batch
        attention_module = Mock(side_effect=attention)
        original = cast(
            OPTDecoderLayer,
            SimpleNamespace(
                do_layer_norm_before=do_layer_norm_before,
                self_attn_layer_norm=input_norm,
                self_attn=attention_module,
                final_layer_norm=final_norm,
                fc1=fc1,
                activation_fn=activation,
                fc2=fc2,
            ),
        )
        replacement = cast(
            XpoolOPTDecoderLayer,
            SimpleNamespace(
                do_layer_norm_before=do_layer_norm_before,
                self_attn_layer_norm=input_norm,
                self_attn=attention_module,
                final_layer_norm=final_norm,
                mlp=mlp,
            ),
        )

        original_result = OPTDecoderLayer.forward(original, hidden_states, forward_batch)
        original_events = tuple(events)
        events.clear()
        replacement_result = opt_decoder_forward(replacement, hidden_states, forward_batch)

        assert torch.equal(original_result, replacement_result)
        non_ffn_events = tuple(event for event in original_events if event not in {"fc1", "activation", "fc2"})
        replacement_events = tuple(event for event in events if event != "mlp")
        assert replacement_events == non_ffn_events
