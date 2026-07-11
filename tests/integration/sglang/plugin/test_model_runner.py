from __future__ import annotations

from tests.harness.sglang.fakes import FakeModelConfig, FakeModelRunner, server_args
from tests.harness.sglang.plugin import (
    AtnKind,
    FailingAfterLoadAdapter,
    FakeAdapter,
    ModelRunner,
    Path,
    SglangModelMetadata,
    config_module,
    configure_xpool_model,
    pytest,
    sglang_plugin,
    sglang_topology,
)
from xpool.abi import RuntimeRole
from xpool.config import MissingRequiredConfig, TopologyError
from xpool.runtime.transport import InstanceTransportAttributes


def test_model_runner_hook_delegates_to_matching_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    monkeypatch.setattr(sglang_plugin, "init_instance", lambda *args, **kwargs: None)

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
            num_atn_heads=16,
            num_key_value_heads=2,
            atn_kind=AtnKind.GQA,
            physical_kv_lanes=2,
        )

    monkeypatch.setattr(sglang_topology, "load_sglang_model_metadata", fake_sglang_metadata)
    config_module.init_global_config()
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(model_path)))

    monkeypatch.setattr(sglang_plugin, "init_instance", lambda *args, **kwargs: None)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert result == "loaded"
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_model_binding is not None
    assert runner.xpool_model_binding.instance_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"


def test_model_runner_hook_installs_transport_runtime_for_production_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    installs: list[tuple[str, int]] = []
    registrations: list[tuple[str, int, int]] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        sglang_plugin.bootstrap,
        "init",
        lambda cuda_device, role: events.append(f"init:{cuda_device}:{int(role)}"),
    )
    monkeypatch.setattr(sglang_plugin.devkit, "install", lambda: events.append("devkit"))

    def fake_instance_init(
        *,
        instance_id: str,
        rank: int,
        transport: InstanceTransportAttributes,
    ) -> None:
        registrations.append((instance_id, rank, transport.element_size))
        installs.append((instance_id, rank))
        events.append("start_instance")

    monkeypatch.setattr(sglang_plugin, "init_instance", fake_instance_init)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
    ]

    pool_result = sglang_plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert result == "loaded"
    assert pool_result is None
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
    ]
    assert registrations == [("deepseek-ai/DeepSeek-V2-Lite-Chat", 0, 2)]
    assert installs == [("deepseek-ai/DeepSeek-V2-Lite-Chat", 0)]


def test_model_runner_hook_validates_before_daemon_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FailingAfterLoadAdapter(events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        sglang_plugin,
        "init_instance",
        lambda *args, **kwargs: pytest.fail("load validation failure must not create instance runtime"),
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(RuntimeError, match="validation failed"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_model_binding is None


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

    monkeypatch.setattr(sglang_plugin, "init_instance", fail_instance_init)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    assert sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner()) == "loaded"
    with pytest.raises(RuntimeError, match="install failed"):
        sglang_plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert events == [
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
    ]
    assert runner.xpool_model_binding is None


def test_model_runner_hook_clears_binding_when_instance_start_fails_before_attach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def fail_instance_init(*args: object, **kwargs: object) -> None:
        events.append("start_instance")
        raise RuntimeError("heartbeat failed")

    monkeypatch.setattr(sglang_plugin, "init_instance", fail_instance_init)

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    assert sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner()) == "loaded"
    with pytest.raises(RuntimeError, match="heartbeat failed"):
        sglang_plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert events == [
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
    ]
    assert runner.xpool_model_binding is None


def test_model_runner_hook_skips_transport_runtime_for_direct_shim_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    monkeypatch.setenv("XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE", "1")
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        sglang_plugin,
        "init_instance",
        lambda *args, **kwargs: pytest.fail("direct shim loopback must not create instance runtime"),
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())
    pool_result = sglang_plugin.after_model_runner_init_memory_pool(None, runner.as_model_runner(), 0)

    assert result == "loaded"
    assert pool_result is None
    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]


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
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == []
    assert runner.xpool_model_binding is None


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
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_requires_global_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner()
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(MissingRequiredConfig, match="global config"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_sglang_tp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=server_args(tp_size=1),
    )
    configure_xpool_model(
        tmp_path,
        monkeypatch,
        runner.model_config.model_path,
        atn_cuda_devices=(0, 1),
        ffn_cuda_devices=(2,),
    )

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="tp_size=2"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_sglang_dp_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")),
        server_args=server_args(dp_size=2),
    )
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(RuntimeError, match="dp_size=1"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())


def test_model_runner_hook_rejects_ffn_tp_divisibility_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(
        tmp_path,
        monkeypatch,
        runner.model_config.model_path,
        ffn_cuda_devices=(2, 3, 4),
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    with pytest.raises(TopologyError, match="FFN TP 3"):
        sglang_plugin.around_model_runner_load_model((adapter,), original, runner.as_model_runner())

    assert events == []
    assert runner.xpool_model_binding is None
