"""Install and reset the pinned SGLang plugin hook environment for tests."""

from __future__ import annotations

import signal
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry

import xpool.config
import xpool.integrations.sglang.hooks.lifecycle
import xpool.integrations.sglang.hooks.shutdown
import xpool.integrations.sglang.plugin
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.integrations.sglang.adapter import (
    SglangInstanceRankBinding,
    SglangShimAdapter,
)
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.topology import SglangAttentionKind, SglangModelMetadata
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.utils.mps import MpsEndpoint
from xtest.harness.support.config import TEST_MODEL_ID, synthetic_config, write_minimal_config


@pytest.fixture
def reset_plugin_required_hook_targets(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    apply_hooks = HookRegistry.__dict__["apply_hooks"]
    signal_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    HookRegistry.reset()
    monkeypatch.setenv("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING", "false")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")
    monkeypatch.setenv("SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION", "true")
    visibility = tuple(f"GPU-00000000-0000-0000-0000-{index:012x}" for index in range(8))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(visibility))
    for name, value in MpsEndpoint(visibility[:1]).environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "visible_uuids", lambda: visibility)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle, "visible_uuids", lambda: visibility)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.shutdown, "visible_uuids", lambda: visibility)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.shutdown, "get_global_config", synthetic_config)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.bootstrap, "init", lambda device, role: None)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.devkit, "install", lambda package=None: None)

    class ConfigClient:
        def check_config(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(xpool.integrations.sglang.plugin, "XpoolClient", ConfigClient)
    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.MpsEndpoint, "require_client", lambda self: None)
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "discover_sglang_hooks", lambda: ())
    xpool.integrations.sglang.plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()
    yield
    for signum, handler in signal_handlers.items():
        signal.signal(signum, handler)
    HookRegistry.reset()
    setattr(HookRegistry, "apply_hooks", apply_hooks)
    xpool.integrations.sglang.plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()


def configure_xpool_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    *,
    atn_devices: tuple[int, ...] = (0,),
    ffn_devices: tuple[int, ...] = (1,),
    atn_kind: SglangAttentionKind = SglangAttentionKind.GQA,
    num_key_value_heads: int = 2,
    model_slo: tuple[int, int] | None = None,
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
    config_path = write_minimal_config(
        tmp_path / "xpool.toml",
        model_path=resolved_model_path,
        atn_devices=atn_devices,
        ffn_devices=ffn_devices,
        model_slo=model_slo,
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(config_path: Path, *, model_id: ModelId) -> SglangModelMetadata:
        return SglangModelMetadata(
            model_id=model_id,
            family=str(model_id),
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
        model_id=TEST_MODEL_ID,
        model_path=Path("/tmp/xpool/fake-model"),
        instance_index=0,
        worker_rank=0,
        device=0,
        worker_world_size=1,
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
        payload_dtype=torch.float16,
        hidden_size=hidden_size,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=decode_payload_row_capacity,
        prefill_payload_row_capacity=prefill_payload_row_capacity,
        group_sum_complete_admitted=False,
    )


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
