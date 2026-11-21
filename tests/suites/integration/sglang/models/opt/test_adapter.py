from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.opt import OPTDecoderLayer, OPTForCausalLM
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.integrations.sglang.models.opt import (
    OptShimAdapter,
    XpoolOptDecoderLayer,
    XpoolOptModel,
    around_load_weights,
)
from xtest.harness.support.sglang.fakes import runner_with_architecture


def test_opt_adapter_hooks_and_model_matching() -> None:
    adapter = OptShimAdapter()
    hooks = {(hook.target, hook.kind): hook.handler for hook in adapter.hooks()}
    assert hooks[("sglang.srt.models.opt.OPTDecoderLayer", HookType.REPLACE)] is XpoolOptDecoderLayer
    assert hooks[("sglang.srt.models.opt.OPTModel", HookType.REPLACE)] is XpoolOptModel
    assert hooks[("sglang.srt.models.opt.OPTForCausalLM.load_weights", HookType.AROUND)] is around_load_weights
    assert adapter.matches(runner_with_architecture("OPTForCausalLM").as_model_runner())
    assert not adapter.matches(runner_with_architecture("Qwen2ForCausalLM").as_model_runner())


@pytest.mark.parametrize("pre_norm", (True, False))
def test_opt_decoder_preserves_upstream_residual_and_norm_order(pre_norm: bool) -> None:
    states = torch.tensor([[-1.0, 2.0]], dtype=torch.float16)
    batch = cast(ForwardBatch, object())
    common = {
        "do_layer_norm_before": pre_norm,
        "self_attn_layer_norm": lambda x: x + 1,
        "self_attn": lambda *, hidden_states, forward_batch: hidden_states * 2,
        "final_layer_norm": lambda x: x - 1,
    }
    original = cast(
        OPTDecoderLayer,
        SimpleNamespace(
            **common,
            fc1=lambda x: (x * 3, None),
            activation_fn=torch.relu,
            fc2=lambda x: (x + 4, None),
        ),
    )
    replacement = cast(
        XpoolOptDecoderLayer,
        SimpleNamespace(
            **common,
            mlp=lambda x, *, forward_batch: torch.relu(x * 3) + 4,
        ),
    )
    torch.testing.assert_close(
        XpoolOptDecoderLayer.forward(replacement, states, batch),
        OPTDecoderLayer.forward(original, states, batch),
        rtol=0,
        atol=0,
    )


def test_opt_weight_loader_filters_both_source_prefixes_and_freezes() -> None:
    model = nn.Linear(1, 1)
    source = [
        ("model.decoder.layers.0.fc1.weight", torch.ones(1)),
        ("decoder.layers.0.fc2.bias", torch.ones(1)),
        ("model.decoder.layers.0.self_attn.q_proj.weight", torch.ones(1)),
    ]
    retained: list[str] = []

    def load(_: OPTForCausalLM, items: Iterable[tuple[str, torch.Tensor]]) -> None:
        retained.extend(name for name, _ in items)

    around_load_weights(load, cast(OPTForCausalLM, model), source)
    assert retained == ["model.decoder.layers.0.self_attn.q_proj.weight"]
    assert not any(parameter.requires_grad for parameter in model.parameters())
