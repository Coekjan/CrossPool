from __future__ import annotations

from pathlib import Path

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner

import xpool.config
import xpool.integrations.sglang.plugin
from tests.harness.support.config import TEST_MODEL_ID, reset_global_config
from tests.harness.support.sglang.fakes import FakeModelConfig, FakeModelRunner
from tests.harness.support.sglang.plugin import (
    FailingAfterLoadAdapter,
    FakeAdapter,
    configure_xpool_model,
    ffn_workload,
    reset_plugin_required_hook_targets,
)
from xpool.config import MissingRequiredConfig
from xpool.integrations.sglang.topology import AtnKind, SglangModelMetadata
from xpool.runtime import RuntimeRole
from xpool.runtime.transport import InstanceTransportAttributes

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_plugin_required_hook_targets.__name__)


def test_model_runner_hook_delegates_to_matching_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        assert model_runner is runner.as_model_runner()
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), original, runner.as_model_runner()
    )

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_runtime is not None
    binding = runner.xpool_runtime.binding
    assert binding.instance_id == TEST_MODEL_ID
    assert binding.worker_world_size == 1
    assert binding.atn_dp_size == 1


def test_model_runner_hook_resolves_only_the_matching_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
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
atn_concurrency = 1
ffn_concurrency = 1

[devices]
atn_cuda_devices = [0]
ffn_cuda_devices = [1]

[[models]]
id = "test/model"
path = "{model_path}"

[[models]]
id = "organization/unrelated-model"
path = "{unrelated_model_path}"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    def fake_sglang_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        assert config_path == model_path / "config.json"
        assert model_id == "test/model"
        return SglangModelMetadata(
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=2,
            atn_kind=AtnKind.GQA,
        )

    monkeypatch.setattr(SglangModelMetadata, "load", fake_sglang_metadata)
    xpool.config.init_global_config()
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(model_path)))

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), original, runner.as_model_runner()
    )

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_runtime is not None
    assert runner.xpool_runtime.binding.instance_id == "test/model"


def test_model_runner_hook_installs_transport_runtime_for_production_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    installs: list[tuple[str, int]] = []
    registrations: list[tuple[str, int, int]] = []
    workloads: list[object] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin.bootstrap,
        "init",
        lambda cuda_device, role: events.append(f"init:{cuda_device}:{int(role)}"),
    )
    monkeypatch.setattr(xpool.integrations.sglang.plugin.devkit, "install", lambda: events.append("devkit"))

    class FakeInstanceRuntime:
        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return object()

        def attach_arena_from_daemon(self) -> None:
            events.append("attach_transport")

        def start_failure_monitor(self) -> None:
            events.append("start_failure_monitor")

        def publish_initialized(self) -> None:
            events.append("publish_initialized")

        def wait_for_ready(self) -> None:
            events.append("wait_for_ready")

    def fake_instance_init(
        *,
        instance_id: str,
        rank: int,
        transport: InstanceTransportAttributes,
        workload: object,
    ) -> FakeInstanceRuntime:
        registrations.append((instance_id, rank, transport.max_tokens))
        workloads.append(workload)
        installs.append((instance_id, rank))
        events.append("start_instance")
        return FakeInstanceRuntime()

    monkeypatch.setattr(xpool.integrations.sglang.plugin.Instance, "start", fake_instance_init)
    workload = ffn_workload()
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin, "derive_workload", lambda model_runner, binding, args: workload
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), original, runner.as_model_runner()
    )
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
    ]

    pool_result = xpool.integrations.sglang.plugin.after_model_runner_init_memory_pool(
        None, runner.as_model_runner(), 0
    )
    initialize_result = xpool.integrations.sglang.plugin.after_model_runner_initialize(
        None, runner.as_model_runner(), 0.0
    )

    assert result == "loaded"
    assert pool_result is None
    assert initialize_result is None
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
        "wait_for_fabric_executable",
        "attach_transport",
        "start_failure_monitor",
        "publish_initialized",
        "wait_for_ready",
    ]
    assert registrations == [(TEST_MODEL_ID, 0, 8)]
    assert installs == [(TEST_MODEL_ID, 0)]
    assert workloads == [workload]


def test_model_runner_hook_validates_before_daemon_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FailingAfterLoadAdapter(events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(RuntimeError, match="validation failed"):
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_runtime is None


def test_model_runner_hook_clears_binding_when_instance_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def fail_instance_init(*args: object, **kwargs: object) -> None:
        events.append("start_instance")
        raise RuntimeError("install failed")

    monkeypatch.setattr(xpool.integrations.sglang.plugin.Instance, "start", fail_instance_init)
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "derive_workload", lambda *args: ffn_workload())

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    assert (
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())
        == "loaded"
    )
    with pytest.raises(RuntimeError, match="install failed"):
        xpool.integrations.sglang.plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert events == [
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
    ]
    assert runner.xpool_runtime is None


def test_model_runner_hook_cleans_up_when_post_executable_transport_attach_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    class FakeInstanceRuntime:
        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return object()

        def attach_arena_from_daemon(self) -> None:
            events.append("attach_transport")
            raise RuntimeError("attach failed")

        def start_failure_monitor(self) -> None:
            pytest.fail("failure monitor must not start after attachment failure")

        def close(self) -> None:
            events.append("close_instance")

    def start_instance(*args: object, **kwargs: object) -> FakeInstanceRuntime:
        events.append("start_instance")
        return FakeInstanceRuntime()

    monkeypatch.setattr(xpool.integrations.sglang.plugin.Instance, "start", start_instance)
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "derive_workload", lambda *args: ffn_workload())

    xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), lambda model_runner: None, runner.as_model_runner()
    )
    with pytest.raises(RuntimeError, match="attach failed"):
        xpool.integrations.sglang.plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert events[-4:] == [
        "start_instance",
        "wait_for_fabric_executable",
        "attach_transport",
        "close_instance",
    ]
    assert runner.xpool_runtime is None


def test_model_runner_hook_skips_transport_runtime_for_instance_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    monkeypatch.setenv("XPOOL_DEBUG_LOOPBACK_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_LOOPBACK_SITE", "instance")
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin.Instance,
        "start",
        lambda *args, **kwargs: pytest.fail("instance loopback must not create instance runtime"),
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), original, runner.as_model_runner()
    )
    pool_result = xpool.integrations.sglang.plugin.after_model_runner_init_memory_pool(
        None, runner.as_model_runner(), 0
    )

    assert result == "loaded"
    assert pool_result is None
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]


def test_model_runner_hook_waits_for_executable_ffnagent_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FfnAgent loopback attaches transport and waits for joined participants."""

    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    monkeypatch.setenv("XPOOL_DEBUG_LOOPBACK_ENABLE", "1")
    monkeypatch.setenv("XPOOL_DEBUG_LOOPBACK_SITE", "ffnagent")
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    class FakeInstanceRuntime:
        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return object()

        def attach_arena_from_daemon(self) -> None:
            events.append("attach_transport")

        def start_failure_monitor(self) -> None:
            events.append("start_failure_monitor")

    def fake_instance_init(*args: object, **kwargs: object) -> FakeInstanceRuntime:
        events.append("register_instance_workload")
        return FakeInstanceRuntime()

    monkeypatch.setattr(xpool.integrations.sglang.plugin.Instance, "start", fake_instance_init)
    monkeypatch.setattr(xpool.integrations.sglang.plugin, "derive_workload", lambda *args: ffn_workload())

    xpool.integrations.sglang.plugin.around_model_runner_load_model(
        (adapter,), lambda model_runner: None, runner.as_model_runner()
    )

    result = xpool.integrations.sglang.plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert result is None
    assert runner.xpool_runtime is not None
    assert events[-4:] == [
        "register_instance_workload",
        "wait_for_fabric_executable",
        "attach_transport",
        "start_failure_monitor",
    ]


def test_model_runner_hook_rejects_configured_model_without_matching_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=False, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(RuntimeError, match="no xpool adapter"):
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == []
    assert runner.xpool_runtime is None


def test_model_runner_hook_rejects_model_path_missing_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, str(tmp_path / "other-model"))

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="no model entry"):
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_requires_global_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner()
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(MissingRequiredConfig, match="global config"):
        xpool.integrations.sglang.plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())
