"""Install and reset the pinned SGLang plugin hook environment for tests."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry

import xpool.config
import xpool.integrations.sglang.plugin
from tests.harness.support.config import TEST_MODEL_ID
from xpool.config import XpoolConfig
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangInstanceRankBinding,
    SglangShimAdapter,
)
from xpool.integrations.sglang.topology import SglangAttentionKind, SglangModelMetadata
from xpool.native.ffn import LayerKind


@pytest.fixture
def reset_plugin_required_hook_targets(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    apply_hooks = HookRegistry.__dict__["apply_hooks"]
    guarded = HookRegistry.__dict__.get("xpool_apply_hooks_guarded")
    HookRegistry.reset()
    monkeypatch.setattr(xpool.integrations.sglang.plugin.bootstrap, "init", lambda cuda_device, role: None)
    monkeypatch.setattr(xpool.integrations.sglang.plugin.devkit, "install", lambda package=None: None)
    xpool.integrations.sglang.plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()
    yield
    HookRegistry.reset()
    setattr(HookRegistry, "apply_hooks", apply_hooks)
    if guarded is None:
        with contextlib.suppress(AttributeError):
            delattr(HookRegistry, "xpool_apply_hooks_guarded")
    else:
        setattr(HookRegistry, "xpool_apply_hooks_guarded", guarded)
    xpool.integrations.sglang.plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()


def configure_xpool_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    *,
    atn_cuda_devices: tuple[int, ...] = (0,),
    ffn_cuda_devices: tuple[int, ...] = (1,),
    atn_kind: SglangAttentionKind = SglangAttentionKind.GQA,
    num_key_value_heads: int = 2,
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
id = "{TEST_MODEL_ID}"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            model_id=model_id,
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=num_key_value_heads,
            atn_kind=atn_kind,
            raw_config_path=config_path,
        )

    monkeypatch.setattr(SglangModelMetadata, "load", fake_sglang_metadata)
    xpool.config.init_global_config()


def binding() -> SglangInstanceRankBinding:
    return SglangInstanceRankBinding(
        instance_id=TEST_MODEL_ID,
        model_path=Path("/tmp/xpool/fake-model"),
        instance_index=0,
        worker_rank=0,
        cuda_device=0,
        worker_world_size=1,
        sglang_base_gpu_id=0,
        sglang_gpu_id_step=1,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def ffn_profile(
    *,
    hidden_size: int = 2048,
    decode_payload_row_capacity: int = 4,
    prefill_payload_row_capacity: int = 8,
) -> InstanceFfnProfile:
    """Return one strict FFN profile suitable for SGLang plugin tests."""

    return InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.float16,
        hidden_size=hidden_size,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=decode_payload_row_capacity,
        prefill_payload_row_capacity=prefill_payload_row_capacity,
        group_sum_complete_admitted=False,
    )


def minimal_config() -> XpoolConfig:
    """Return a minimal xpool config for plugin install tests."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )


def hook_target_load() -> str:
    """Synthetic valid target for the plugin's required load hook."""

    return "load"


def hook_target_alloc_memory_pool() -> str:
    """Synthetic valid target for the plugin's pool-allocation hook."""

    return "memory_pool"


def hook_target_get_init_info() -> str:
    """Synthetic valid target for the plugin's scheduler-handshake hook."""

    return "initialize"


class FakeAdapter(SglangShimAdapter):
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
