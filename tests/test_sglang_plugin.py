from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import cast

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import ServerArgs
from sglang_fakes import server_args
from transformers import PretrainedConfig

from xpool.config import MissingRequiredConfig
from xpool.integrations.sglang import plugin as sglang_plugin
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangHookHandler,
    SglangModelAdapter,
    XpoolModelBinding,
)
from xpool.integrations.sglang.server_args import SGLANG_SERVER_ARG_RULES, validate_sglang_server_args


def test_sglang_discovers_xpool_entry_point() -> None:
    discovered = {entry_point.name: entry_point.value for entry_point in entry_points(group="sglang.srt.plugins")}

    assert discovered["xpool"] == "xpool.integrations.sglang.plugin:install"


def test_sglang_plugin_whitelist_finds_xpool(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.plugins import GENERAL_PLUGINS_GROUP, load_plugins_by_group

    monkeypatch.setenv("SGLANG_PLUGINS", "xpool")

    plugins = load_plugins_by_group(GENERAL_PLUGINS_GROUP)

    assert "xpool" in plugins


def test_importing_global_plugin_does_not_import_model_adapters() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import xpool.integrations.sglang.plugin; "
            "raise SystemExit(1 if 'xpool.integrations.sglang.models.deepseek_v2' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_plugin_registers_adapter_owned_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeHookRegistry.calls = []

    def fake_hook() -> None:
        return None

    adapter = _FakeAdapter(hooks=(SglangHook("xpool.fake.Target", fake_hook, HookType.REPLACE),))

    monkeypatch.setattr(sglang_plugin, "sglang_model_adapters", lambda: (adapter,))
    monkeypatch.setattr(sglang_plugin, "HookRegistry", _FakeHookRegistry)

    sglang_plugin.install()

    assert ("xpool.fake.Target", fake_hook, HookType.REPLACE) in _FakeHookRegistry.calls
    assert any(
        target == sglang_plugin.MODEL_RUNNER_LOAD_MODEL and registered_type is HookType.AROUND
        for target, _handler, registered_type in _FakeHookRegistry.calls
    )


def test_model_runner_hook_delegates_to_matching_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=True, events=events)
    runner = _FakeModelRunner(model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        assert model_runner is as_model_runner(runner)
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    binding = runner.xpool_model_binding
    assert binding is not None
    assert binding.instance_id == "deepseek-v2-lite-chat"
    assert binding.sglang_tp_size == 1
    assert binding.sglang_dp_size == 1


def test_model_runner_hook_rejects_configured_model_without_matching_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=False, events=events)
    runner = _FakeModelRunner(model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(RuntimeError, match="no xpool adapter"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))

    assert events == []


def test_model_runner_hook_rejects_model_path_missing_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = _FakeModelRunner(model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, str(tmp_path / "other-model"))

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="no model entry"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))


def test_model_runner_hook_requires_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = _FakeModelRunner()
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(MissingRequiredConfig, match="XPOOL_CONFIG"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))


def test_model_runner_hook_requires_server_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = _FakeModelRunner(
        model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=None,
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="requires ModelRunner.server_args"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))


def test_model_runner_hook_rejects_sglang_tp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = _FakeModelRunner(
        model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=server_args(tp_size=1),
    )
    configure_xpool_model(
        tmp_path,
        monkeypatch,
        runner.model_config.model_path,
        attention_cuda_devices=(0, 1),
        ffn_cuda_devices=(2,),
    )

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="tp_size=2"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))


def test_model_runner_hook_rejects_sglang_dp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = _FakeModelRunner(
        model_config=_FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=server_args(dp_size=2),
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="dp_size=1"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, as_model_runner(runner))


def test_global_server_arg_gate_rejects_unsupported_sglang_features() -> None:
    args = server_args(enable_two_batch_overlap=True, cpu_offload_gb=1)

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_default_sglang_features() -> None:
    validate_sglang_server_args(server_args())


def test_global_server_arg_gate_allows_resolved_sglang_defaults() -> None:
    args = ServerArgs(model_path="dummy")

    validate_sglang_server_args(args)


def test_every_server_arg_rule_evaluates_without_attribute_error() -> None:
    args = ServerArgs(model_path="dummy")

    for rule in SGLANG_SERVER_ARG_RULES:
        rule.supported(args)


def test_global_server_arg_gate_rejects_lora_when_enabled() -> None:
    args = server_args(enable_lora=True)

    with pytest.raises(RuntimeError, match="LoRA"):
        validate_sglang_server_args(args)


@pytest.mark.parametrize(
    ("override", "label"),
    [
        ({"enable_mixed_chunk": True}, "Mixed Chunked Prefill"),
        ({"disaggregation_mode": "prefill"}, "PD Disaggregation"),
        ({"dllm_algorithm": "next_block"}, "Diffusion LLM"),
        ({"enable_pdmux": True}, "PD Multiplexing"),
    ],
)
def test_global_server_arg_gate_rejects_composite_forward_modes(
    override: dict[str, object],
    label: str,
) -> None:
    args = server_args(**override)

    with pytest.raises(RuntimeError, match=label):
        validate_sglang_server_args(args)


def test_global_server_arg_labels_are_title_case() -> None:
    labels = [rule.label for rule in SGLANG_SERVER_ARG_RULES]

    assert "Pipeline Parallelism" in labels
    assert "Two-Batch Overlap" in labels
    assert "SGLang CPU Offload" in labels
    assert "FlashInfer All-Reduce Fusion" in labels
    assert "Mixed Chunked Prefill" in labels
    assert "PD Multiplexing" in labels
    assert all(label[:1].isupper() for label in labels)


def as_model_runner(runner: "_FakeModelRunner") -> ModelRunner:
    return cast(ModelRunner, runner)


def configure_xpool_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    *,
    attention_cuda_devices: tuple[int, ...] = (0,),
    ffn_cuda_devices: tuple[int, ...] = (1,),
) -> None:
    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_model_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "xpool.toml"
    attention_devices = ", ".join(str(device) for device in attention_cuda_devices)
    ffn_devices = ", ".join(str(device) for device in ffn_cuda_devices)
    config_path.write_text(
        f"""
[daemon]
host = "127.0.0.1"
port = 9810

[scheduler]
attention_concurrency = 1
transport_concurrency = 1

[devices]
attention_cuda_devices = [{attention_devices}]
ffn_cuda_devices = [{ffn_devices}]

[[models]]
id = "deepseek-v2-lite-chat"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))


class _FakeHookRegistry:
    calls: list[tuple[str, SglangHookHandler, HookType]] = []

    @classmethod
    def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
        cls.calls.append((target, handler, hook_type))


@dataclass(slots=True)
class _FakeModelConfig:
    hf_config: PretrainedConfig | None = field(
        default_factory=lambda: PretrainedConfig(architectures=["FakeForCausalLM"])
    )
    model_path: str = "/tmp/xpool/fake-model"


@dataclass(slots=True)
class _FakeModelRunner:
    model_config: _FakeModelConfig = field(default_factory=_FakeModelConfig)
    server_args: ServerArgs | None = field(default_factory=lambda: server_args())
    xpool_model_binding: XpoolModelBinding | None = None


class _FakeAdapter(SglangModelAdapter):
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
        del model_runner
        return self.match_value

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        del model_runner
        if self.events is not None:
            self.events.append("validate_before_load")

    def bind_runtime(self, model_runner: ModelRunner) -> None:
        del model_runner
        if self.events is not None:
            self.events.append("bind_runtime")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        del model_runner
        if self.events is not None:
            self.events.append("validate_after_load")
