from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Sequence
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from helpers.sglang import FakeModelConfig, FakeModelRunner, server_args
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import ServerArgs

import xpool.config as config_module
from xpool.cext import NativeLoadError
from xpool.config import MissingRequiredConfig, TopologyError
from xpool.integrations.sglang import plugin as sglang_plugin
from xpool.integrations.sglang import topology as sglang_topology
from xpool.integrations.sglang.adapter import (
    SglangHook,
    SglangHookHandler,
    SglangModelAdapter,
)
from xpool.integrations.sglang.server_args import (
    SGLANG_SERVER_ARG_RULES,
    validate_sglang_server_args,
)
from xpool.integrations.sglang.topology import AttentionKind, SglangModelMetadata

UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME = {
    "MIXED": "Mixed Chunked Prefill",
    "IDLE": "DP Attention",
    "TARGET_VERIFY": "Speculative Decoding",
    "DRAFT_EXTEND": "Speculative Decoding",
    "DRAFT_EXTEND_V2": "Speculative Decoding",
    "PREBUILT": "PD Disaggregation",
    "SPLIT_PREFILL": "PD Multiplexing",
    "DLLM_EXTEND": "Diffusion LLM",
}


@pytest.fixture(autouse=True)
def reset_plugin_required_hook_targets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    sglang_plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()
    yield
    sglang_plugin.XPOOL_REQUIRED_HOOK_TARGETS.clear()
    monkeypatch.setattr(config_module, "_global_config", None)


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
    events: list[str] = []

    def fake_hook() -> None:
        return None

    adapter = _FakeAdapter(hooks=(SglangHook("xpool.fake.Target", fake_hook, HookType.REPLACE),))

    def fake_adapters() -> tuple[_FakeAdapter]:
        events.append("discover_adapters")
        return (adapter,)

    monkeypatch.setattr(sglang_plugin, "ensure_xpool_ops_loaded", lambda: events.append("load_cext"))
    monkeypatch.setattr(sglang_plugin, "init_global_config", lambda: events.append("init_config"))
    monkeypatch.setattr(sglang_plugin, "sglang_model_adapters", fake_adapters)
    monkeypatch.setattr(sglang_plugin, "HookRegistry", _FakeHookRegistry)

    sglang_plugin.install()

    assert events == ["load_cext", "init_config", "discover_adapters"]
    assert ("xpool.fake.Target", fake_hook, HookType.REPLACE) in _FakeHookRegistry.calls
    assert any(
        target == sglang_plugin.MODEL_RUNNER_LOAD_MODEL and registered_type is HookType.AROUND
        for target, _handler, registered_type in _FakeHookRegistry.calls
    )


def test_plugin_install_fails_closed_when_native_preflight_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_preflight() -> None:
        raise NativeLoadError("missing native op")

    monkeypatch.setattr(sglang_plugin, "ensure_xpool_ops_loaded", fail_preflight)

    with pytest.raises(SystemExit, match="missing native op"):
        sglang_plugin.install()


def test_plugin_apply_hooks_guard_fails_closed_when_required_target_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardedHookRegistry:
        calls: list[tuple[str, SglangHookHandler, HookType]] = []
        _patched: set[str] = {sglang_plugin.MODEL_RUNNER_LOAD_MODEL}

        @classmethod
        def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
            cls.calls.append((target, handler, hook_type))

        @classmethod
        def apply_hooks(cls) -> None:
            return None

    adapter = _FakeAdapter(hooks=(SglangHook("xpool.fake.Target", lambda: None, HookType.REPLACE),))
    monkeypatch.setattr(sglang_plugin, "ensure_xpool_ops_loaded", lambda: None)
    monkeypatch.setattr(sglang_plugin, "init_global_config", lambda: None)
    monkeypatch.setattr(sglang_plugin, "sglang_model_adapters", lambda: (adapter,))
    monkeypatch.setattr(sglang_plugin, "HookRegistry", GuardedHookRegistry)

    sglang_plugin.install()

    with pytest.raises(SystemExit, match="xpool.fake.Target"):
        GuardedHookRegistry.apply_hooks()


def test_plugin_apply_hooks_guard_allows_all_required_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    class GuardedHookRegistry:
        calls: list[tuple[str, SglangHookHandler, HookType]] = []
        _patched: set[str] = {"xpool.fake.Target", sglang_plugin.MODEL_RUNNER_LOAD_MODEL}

        @classmethod
        def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
            cls.calls.append((target, handler, hook_type))

        @classmethod
        def apply_hooks(cls) -> None:
            return None

    adapter = _FakeAdapter(hooks=(SglangHook("xpool.fake.Target", lambda: None, HookType.REPLACE),))
    monkeypatch.setattr(sglang_plugin, "ensure_xpool_ops_loaded", lambda: None)
    monkeypatch.setattr(sglang_plugin, "init_global_config", lambda: None)
    monkeypatch.setattr(sglang_plugin, "sglang_model_adapters", lambda: (adapter,))
    monkeypatch.setattr(sglang_plugin, "HookRegistry", GuardedHookRegistry)

    sglang_plugin.install()

    GuardedHookRegistry.apply_hooks()


def test_model_runner_hook_delegates_to_matching_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        assert model_runner is runner.as_model_runner()
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    binding = runner.xpool_model_binding
    assert binding is not None
    assert binding.instance_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert binding.sglang_tp_size == 1
    assert binding.sglang_dp_size == 1


def test_model_runner_hook_resolves_only_the_matching_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=True, events=events)
    model_path = (tmp_path / "served-model").resolve()
    unrelated_model_path = (tmp_path / "broken-unrelated-model").resolve()
    model_path.mkdir()
    (model_path / "config.json").write_text(
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
    config_path.write_text(
        f"""
[daemon]
host = "127.0.0.1"
port = 9810

[scheduler]
attention_concurrency = 1
transport_concurrency = 1

[devices]
attention_cuda_devices = [0]
ffn_cuda_devices = [1]

[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{model_path}"

[[models]]
id = "deepseek-ai/Broken-Unrelated-Model"
path = "{unrelated_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        assert config_path == model_path / "config.json"
        assert model_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
        return SglangModelMetadata(
            family=model_id,
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=2,
            attention_kind=AttentionKind.GQA,
            physical_kv_lanes=2,
        )

    monkeypatch.setattr(sglang_topology, "_load_sglang_model_metadata", fake_sglang_metadata)
    config_module.init_global_config()
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(model_path)))

    def original(_model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_model_binding is not None
    assert runner.xpool_model_binding.instance_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"


def test_model_runner_hook_rejects_configured_model_without_matching_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=False, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(RuntimeError, match="no xpool adapter"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == []
    assert runner.xpool_model_binding is None


def test_model_runner_hook_rejects_model_path_missing_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, str(tmp_path / "other-model"))

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="no model entry"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_requires_global_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner()
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(MissingRequiredConfig, match="global config"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_server_args_before_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner(server_args=server_args(enable_dp_attention=True))
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="DP Attention"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_requires_server_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=None,
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="requires ModelRunner.server_args"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_sglang_tp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
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
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_sglang_dp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=server_args(dp_size=2),
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(_model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="dp_size=1"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_ffn_tp_divisibility_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = _FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(
        tmp_path,
        monkeypatch,
        runner.model_config.model_path,
        ffn_cuda_devices=(2, 3, 4),
    )

    def original(_model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(TopologyError, match="FFN TP 3"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == []
    assert runner.xpool_model_binding is None


def test_global_server_arg_gate_rejects_unsupported_sglang_features() -> None:
    args = server_args(enable_two_batch_overlap=True, cpu_offload_gb=1)

    with pytest.raises(RuntimeError, match="Two-Batch Overlap"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_allows_default_sglang_features() -> None:
    validate_sglang_server_args(server_args())
    validate_sglang_server_args(server_args(piecewise_cuda_graph_compiler="eager"))


def test_global_server_arg_gate_allows_resolved_sglang_defaults() -> None:
    args = ServerArgs(model_path="dummy")

    validate_sglang_server_args(args)


def test_every_server_arg_rule_evaluates_without_attribute_error() -> None:
    args = ServerArgs(model_path="dummy")

    for rule in SGLANG_SERVER_ARG_RULES:
        rule.supported(args)


def test_every_unsupported_sglang_forward_mode_has_server_arg_gate_label() -> None:
    labels = {rule.label for rule in SGLANG_SERVER_ARG_RULES}
    unsupported_modes = set(ForwardMode.__members__) - {"DECODE", "EXTEND"}

    assert set(UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME) == unsupported_modes
    assert set(UNSUPPORTED_FORWARD_MODE_RULE_LABEL_BY_NAME.values()).issubset(labels)


def test_global_server_arg_gate_rejects_dp_attention() -> None:
    args = server_args(tp_size=2, dp_size=2, enable_dp_attention=True)

    with pytest.raises(RuntimeError, match="DP Attention"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_rejects_lora_when_enabled() -> None:
    args = server_args(enable_lora=True)

    with pytest.raises(RuntimeError, match="LoRA"):
        validate_sglang_server_args(args)


def test_global_server_arg_gate_rejects_compile_paths() -> None:
    with pytest.raises(RuntimeError, match="Torch Compile"):
        validate_sglang_server_args(server_args(enable_torch_compile=True))
    with pytest.raises(RuntimeError, match="Piecewise CUDA Graph Compiler"):
        validate_sglang_server_args(server_args(piecewise_cuda_graph_compiler="inductor"))


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
    assert "Torch Compile" in labels
    assert "Piecewise CUDA Graph Compiler" in labels
    assert all(label[:1].isupper() for label in labels)


def configure_xpool_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    *,
    attention_cuda_devices: tuple[int, ...] = (0,),
    ffn_cuda_devices: tuple[int, ...] = (1,),
    attention_kind: AttentionKind = AttentionKind.GQA,
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
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(_config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=model_id,
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=num_key_value_heads,
            attention_kind=attention_kind,
            physical_kv_lanes=physical_kv_lanes,
        )

    monkeypatch.setattr(sglang_topology, "_load_sglang_model_metadata", fake_sglang_metadata)
    config_module.init_global_config()


class _FakeHookRegistry:
    calls: list[tuple[str, SglangHookHandler, HookType]] = []

    @classmethod
    def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
        cls.calls.append((target, handler, hook_type))


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
