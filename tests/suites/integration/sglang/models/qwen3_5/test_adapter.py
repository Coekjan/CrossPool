from collections.abc import Iterable
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from xpool.integrations.sglang.models.qwen3_5 import XpoolQwen35MLP, around_load_weights, around_silu_fusion


def test_shim_initialization_preserves_language_identity_without_projection_parameters() -> None:
    for prefix in ("model.layers.7.mlp", "model.language_model.layers.7.mlp"):
        shim = XpoolQwen35MLP(128, 256, "silu", prefix=prefix)
        assert shim.layer_id == 7
        assert not tuple(shim.parameters())
        original = Mock()
        around_silu_fusion(original, shim)
        original.assert_not_called()
    local = torch.nn.Linear(4, 4)
    around_silu_fusion(original, local)
    original.assert_called_once_with(local)


@pytest.mark.parametrize("prefix", ("model.visual.layers.0.mlp", "model.language_model.mtp.layers.0.mlp"))
def test_shim_rejects_non_decoder_prefixes(prefix: str) -> None:
    with pytest.raises(RuntimeError, match="cannot identify"):
        XpoolQwen35MLP(128, 256, "silu", prefix=prefix)


def test_loader_filter_preserves_visual_attention_and_auxiliary_weights() -> None:
    names = (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
    )
    tensors = tuple(torch.tensor([index]) for index in range(len(names)))
    consumed = []

    def load(model: Qwen3_5ForConditionalGeneration, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        consumed.extend(weights)

    model = object.__new__(Qwen3_5ForConditionalGeneration)
    around_load_weights(load, model, zip(names, tensors, strict=True))
    assert tuple(name for name, _ in consumed) == names[3:]
    assert all(tensor is original for (_, tensor), original in zip(consumed, tensors[3:], strict=True))
