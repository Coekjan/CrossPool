from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType

import xpool.config as config_module
from xpool.config import XpoolConfig
from xpool.integrations.sglang import plugin as sglang_plugin
from xpool.integrations.sglang import topology as sglang_topology
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangHookHandler,
    SglangModelAdapter,
    XpoolModelBinding,
)
from xpool.integrations.sglang.topology import AtnKind, SglangModelMetadata


@pytest.fixture(autouse=True)
def reset_plugin_required_hook_targets(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    monkeypatch.setattr(sglang_plugin.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(sglang_plugin.devkit, "install", lambda: None)
    sglang_plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()
    yield
    sglang_plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()


def configure_xpool_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    *,
    atn_cuda_devices: tuple[int, ...] = (0,),
    ffn_cuda_devices: tuple[int, ...] = (1,),
    atn_kind: AtnKind = AtnKind.GQA,
    num_key_value_heads: int = 2,
    physical_kv_lanes: int = 2,
) -> None:
    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_model_path.mkdir(parents=True, exist_ok=True)
    (resolved_model_path / "config.json").write_text(
        """
{
  "model_type": "deepseek_v2",
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "intermediate_size": 8192,
  "moe_intermediate_size": 8192
}
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "xpool.toml"
    atn_devices = ", ".join(str(device) for device in atn_cuda_devices)
    ffn_devices = ", ".join(str(device) for device in ffn_cuda_devices)
    config_path.write_text(
        f"""
[daemon]
host = "127.0.0.1"
port = 9810

[scheduler]
atn_concurrency = 1
ffn_concurrency = 1

[devices]
atn_cuda_devices = [{atn_devices}]
ffn_cuda_devices = [{ffn_devices}]

[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=num_key_value_heads,
            atn_kind=atn_kind,
            physical_kv_lanes=physical_kv_lanes,
        )

    monkeypatch.setattr(sglang_topology, "load_sglang_model_metadata", fake_sglang_metadata)
    config_module.init_global_config()


def binding() -> XpoolModelBinding:
    return XpoolModelBinding(
        instance_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        model_path=Path("/tmp/xpool/fake-model"),
        instance_index=0,
        sglang_rank=0,
        cuda_device=0,
        sglang_tp_size=1,
        sglang_dp_size=1,
        sglang_base_gpu_id=0,
        sglang_gpu_id_step=1,
        enable_dp_atn=False,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


class FakeHookRegistry:
    calls: ClassVar[list[tuple[str, SglangHookHandler, HookType]]] = []

    @classmethod
    def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
        cls.calls.append((target, handler, hook_type))


def minimal_config() -> XpoolConfig:
    """Return a minimal xpool config for plugin install tests."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )


class FakeAdapter(SglangModelAdapter):
    name = "fake"

    def __init__(
        self,
        *,
        hooks: Sequence[SglangHook] = (),
        matches: bool = True,
        events: list[str] | None = None,
    ) -> None:
        self.hook_values = tuple(hooks)
        self.match_value = matches
        self.events = events

    def hooks(self) -> tuple[SglangHook, ...]:
        return self.hook_values

    def matches(self, model_runner: ModelRunner) -> bool:
        return self.match_value

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        if self.events is not None:
            self.events.append("validate_before_load")

    def bind_runtime(self, model_runner: ModelRunner) -> None:
        if self.events is not None:
            self.events.append("bind_runtime")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        if self.events is not None:
            self.events.append("validate_after_load")


class FailingAfterLoadAdapter(FakeAdapter):
    def validate_after_load(self, model_runner: ModelRunner) -> None:
        super().validate_after_load(model_runner)
        raise RuntimeError("validation failed")
