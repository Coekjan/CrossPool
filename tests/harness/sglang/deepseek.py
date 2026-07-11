from __future__ import annotations

from collections.abc import Iterator

import pytest
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode
from transformers import PretrainedConfig

from xpool.config import (
    XpoolConfig,
    init_global_config,
)
from xpool.integrations.sglang.shim import (
    FfnLayerKind,
    FfnShimModule,
)


@pytest.fixture(autouse=True)
def install_adapter_config(
    reset_global_config: None,
) -> Iterator[None]:
    init_global_config(
        config=XpoolConfig.from_mapping(
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
    return config


def bound_shim(*, layer_id: int = 0, instance_index: int = 0, sglang_rank: int = 0) -> FfnShimModule:
    shim = FfnShimModule(layer_id=layer_id, hidden_size=2048, layer_kind=FfnLayerKind.DENSE)
    shim.bind_identity(
        instance_index=instance_index,
        sglang_rank=sglang_rank,
        model_architecture="DeepseekV2ForCausalLM",
    )
    return shim


def decode_forward_batch() -> object:
    return type("FakeForwardBatch", (), {"forward_mode": SglangForwardMode.DECODE})()
