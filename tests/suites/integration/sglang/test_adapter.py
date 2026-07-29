"""Common SGLang model-adapter behavior."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from torch import nn

from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.adapter import SglangHook, SglangModelAdapter, filter_decoder_ffn_weights
from xpool.integrations.sglang.shim import FfnShimModule


class CommonShim(FfnShimModule):
    """Concrete shim type used to exercise common coverage validation."""


class CommonAdapter(SglangModelAdapter):
    """Minimal adapter exposing the common validation helpers."""

    name = "test"

    def hooks(self) -> tuple[SglangHook, ...]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False


def test_decoder_ffn_weight_filter_is_lazy_and_path_exact() -> None:
    weights = iter(
        [
            ("model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
            ("model.layers.0.self_attn.mlp_probe.weight", torch.empty(1)),
            ("draft.model.layers.0.mlp.gate_proj.weight", torch.empty(1)),
        ]
    )

    filtered = filter_decoder_ffn_weights(weights)

    assert iter(filtered) is filtered
    assert [name for name, tensor in filtered] == [
        "model.layers.0.self_attn.mlp_probe.weight",
        "draft.model.layers.0.mlp.gate_proj.weight",
    ]


def test_require_ffn_shims_validates_exact_ids_and_kinds() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            CommonShim(layer_id=0, hidden_size=4, layer_kind=FfnLayerKind.DENSE),
            CommonShim(layer_id=1, hidden_size=4, layer_kind=FfnLayerKind.SPARSE),
        ]
    )

    shims = CommonAdapter().require_ffn_shims(
        model,
        expected_layer_kinds=(FfnLayerKind.DENSE, FfnLayerKind.SPARSE),
        allowed_shim_types=(CommonShim,),
    )

    assert tuple(shim.layer_id for shim in shims) == (0, 1)


def test_require_ffn_shims_rejects_wrong_kind() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([CommonShim(layer_id=0, hidden_size=4, layer_kind=FfnLayerKind.DENSE)])

    with pytest.raises(RuntimeError, match="mismatched FFN layer kinds"):
        CommonAdapter().require_ffn_shims(
            model,
            expected_layer_kinds=(FfnLayerKind.SPARSE,),
            allowed_shim_types=(CommonShim,),
        )
