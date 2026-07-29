"""Build concrete pinned-SGLang objects with controlled lightweight state."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import torch
from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from torch import nn
from transformers import PretrainedConfig

from xpool.integrations.sglang.adapter import XpoolModelRuntime


def fake_hf_config() -> PretrainedConfig:
    """Return a minimal HF config carrying the fields xpool reads in tests."""

    config = PretrainedConfig(architectures=["FakeForCausalLM"])
    setattr(config, "hidden_size", 4)
    return config


def server_args(**overrides: object) -> ServerArgs:
    """Return real SGLang server args with a dummy model path for tests."""

    args = ServerArgs(model_path="dummy")
    for name, value in overrides.items():
        if not hasattr(args, name):
            raise AttributeError(f"SGLang ServerArgs has no field {name!r}")
        setattr(args, name, value)
    return args


@dataclass(slots=True)
class FakeModelConfig:
    """Minimal SGLang model-config fake used by plugin and adapter tests."""

    hf_config: PretrainedConfig | None = field(default_factory=fake_hf_config)
    model_path: str = "/tmp/xpool/fake-model"
    dtype: torch.dtype = torch.float16


@dataclass(slots=True)
class FakeModelRunner:
    """Minimal SGLang ModelRunner fake shared by integration tests."""

    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)
    server_args: ServerArgs = field(default_factory=server_args)
    gpu_id: int = 0
    tp_rank: int = 0
    tp_size: int = 1
    dp_rank: int | None = None
    dp_size: int = 1
    attn_cp_rank: int | None = None
    attn_cp_size: int = 1
    max_running_requests: int = 1
    model: nn.Module | None = None
    xpool_ffn_shim_count: int = 0
    xpool_runtime: XpoolModelRuntime | None = None

    def as_model_runner(self) -> ModelRunner:
        """Cast this fake runner to SGLang's ``ModelRunner`` type."""

        return cast(ModelRunner, self)


class FakeDecoderLayer(nn.Module):
    """Minimal loaded decoder layer exposing SGLang's MLP boundary facts."""

    def __init__(
        self,
        mlp: nn.Module,
        *,
        mlp_mode: ScatterMode = ScatterMode.FULL,
        allow_reduce_scatter: bool,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.layer_scatter_modes = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=mlp_mode,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        self.layer_communicator = SimpleNamespace(allow_reduce_scatter=allow_reduce_scatter)


def loaded_model[M: nn.Module](model_type: type[M], config: PretrainedConfig, layers: list[nn.Module]) -> M:
    """Create a lightweight instance of a pinned SGLang model type.

    The real constructors allocate model weights and initialize CUDA. Adapter
    post-load tests need only the concrete type, effective config, and decoder
    module tree that form the validation boundary.
    """

    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.config = config
    model.layers = nn.ModuleList(layers)
    return model


def runner_with_architecture(
    architecture: str,
    *,
    runner_server_args: ServerArgs | None = None,
    model_path: str = "/tmp/xpool/fake-model",
) -> FakeModelRunner:
    """Return a fake SGLang model runner with one HF architecture string."""

    hf_config = PretrainedConfig(architectures=[architecture])
    setattr(hf_config, "num_hidden_layers", 2)
    return FakeModelRunner(
        model_config=FakeModelConfig(hf_config=hf_config, model_path=model_path),
        server_args=runner_server_args or server_args(),
    )
