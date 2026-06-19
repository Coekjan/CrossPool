from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from torch import nn
from transformers import PretrainedConfig

from xpool.integrations.sglang.adapter import XpoolModelBinding


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

    hf_config: PretrainedConfig | None = field(
        default_factory=lambda: PretrainedConfig(architectures=["FakeForCausalLM"])
    )
    model_path: str = "/tmp/xpool/fake-model"


@dataclass(slots=True)
class FakeModelRunner:
    """Minimal SGLang ModelRunner fake shared by integration tests."""

    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)
    server_args: ServerArgs | None = field(default_factory=server_args)
    model: nn.Module | None = None
    xpool_ffn_shim_count: int = 0
    xpool_model_binding: XpoolModelBinding | None = None

    def as_model_runner(self) -> ModelRunner:
        """Cast this fake runner to SGLang's ``ModelRunner`` type."""

        return cast(ModelRunner, self)


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
