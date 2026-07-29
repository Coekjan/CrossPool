"""Provide pinned DeepSeek-V2 model fixtures for SGLang integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sglang.srt.model_executor import forward_batch_info
from transformers import PretrainedConfig

from tests.harness.support.config import install_test_config
from xpool.config import XpoolConfig
from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.shim import FfnShimModule


@pytest.fixture
def install_adapter_config(
    reset_global_config: None,
) -> Iterator[None]:
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )
    yield


def deepseek_config() -> PretrainedConfig:
    config = PretrainedConfig(architectures=["DeepseekV2ForCausalLM"])
    setattr(config, "num_hidden_layers", 2)
    setattr(config, "hidden_size", 2048)
    setattr(config, "hidden_act", "silu")
    setattr(config, "n_shared_experts", 1)
    setattr(config, "n_routed_experts", 8)
    setattr(config, "first_k_dense_replace", 1)
    setattr(config, "moe_layer_freq", 1)
    return config


def bound_shim(*, layer_id: int = 0, atn_dp_size: int = 1) -> FfnShimModule:
    shim = FfnShimModule(layer_id=layer_id, hidden_size=2048, layer_kind=FfnLayerKind.DENSE)
    shim.bind_runtime(
        layer_ordinal=0,
        model_architecture="DeepseekV2ForCausalLM",
        atn_dp_size=atn_dp_size,
    )
    return shim


def decode_forward_batch() -> object:
    return type("FakeForwardBatch", (), {"forward_mode": forward_batch_info.ForwardMode.DECODE})()
