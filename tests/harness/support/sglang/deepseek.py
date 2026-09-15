"""Provide pinned DeepSeek-V2 model fixtures for SGLang integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from transformers import DeepseekV2Config

from tests.harness.support.config import install_test_config
from xpool.config import XpoolConfig
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.native.ffn import LayerKind


@pytest.fixture
def install_adapter_config(
    reset_global_config: None,
) -> Iterator[None]:
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )
    yield


def deepseek_config() -> DeepseekV2Config:
    config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        num_hidden_layers=2,
        hidden_size=2048,
        hidden_act="silu",
        n_shared_experts=1,
        n_routed_experts=8,
        first_k_dense_replace=1,
    )
    setattr(config, "moe_layer_freq", 1)
    return config


def bound_shim(*, layer_id: int = 0, atn_dp_size: int = 1) -> FfnShimModule:
    shim = FfnShimModule(layer_id=layer_id, hidden_size=2048, layer_kind=LayerKind.DENSE)
    shim.bind_runtime(
        layer_ordinal=0,
        model_architecture="DeepseekV2ForCausalLM",
        atn_dp_size=atn_dp_size,
    )
    return shim
