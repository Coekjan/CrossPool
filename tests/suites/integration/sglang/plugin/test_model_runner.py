from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner

import xpool.config
import xpool.integrations.sglang.hooks.lifecycle
from tests.harness.support.config import TEST_MODEL_ID, reset_global_config
from tests.harness.support.kv import kv_capacity_profile
from tests.harness.support.sglang.fakes import FakeModelConfig, FakeModelRunner
from tests.harness.support.sglang.plugin import (
    FailingAfterLoadAdapter,
    FakeAdapter,
    binding,
    configure_xpool_model,
    ffn_profile,
    reset_plugin_required_hook_targets,
)
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.config import MissingRequiredConfig
from xpool.fabric import FabricGenerationId
from xpool.integrations.sglang.adapter import SglangInstanceRankRuntime
from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.integrations.sglang.topology import SglangAttentionKind, SglangModelMetadata
from xpool.native import RuntimeRole
from xpool.runtime.transport import InstanceRankTransportProfile
from xpool.service.wire import KvCapacityChannelRef, ServingListener

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__, reset_plugin_required_hook_targets.__name__, published_sglang_config.__name__
)


class FakeElasticPool(ElasticMHATokenToKVPool):
    """Concrete elastic-pool witness for lifecycle tests."""

    def __init__(self) -> None:
        self.backing = cast(KvVmmBacking, SimpleNamespace(partition_profile=kv_capacity_profile))

    def close(self) -> None:
        """Release no resources because this witness allocates none."""


def install_fake_elastic_kv(runner: FakeModelRunner) -> None:
    """Install concrete replacement witnesses without allocating KV storage."""

    runner.token_to_kv_pool = FakeElasticPool()
    runner.token_to_kv_pool_allocator = ElasticTokenToKVPoolAllocator.__new__(ElasticTokenToKVPoolAllocator)
    runner.req_to_token_pool = ReqToTokenPool.__new__(ReqToTokenPool)


def test_model_runner_hook_delegates_to_matching_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def original(model_runner: ModelRunner) -> str:
        assert model_runner is runner.as_model_runner()
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
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

[atn]
devices = [0]

[ffn]
devices = [1]

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
        assert config_path == model_path
        assert model_id == "test/model"
        return SglangModelMetadata(
            model_id=model_id,
            family=model_id,
            hidden_size=2048,
            num_atn_heads=16,
            num_key_value_heads=2,
            atn_kind=SglangAttentionKind.GQA,
            raw_config_path=config_path,
        )

    monkeypatch.setattr(SglangModelMetadata, "load", fake_sglang_metadata)
    xpool.config.init_global_config()
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(model_path)))

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
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
    profiles: list[object] = []
    listeners: list[ServingListener] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    install_fake_elastic_kv(runner)
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle.bootstrap,
        "init",
        lambda cuda_device, role: events.append(f"init:{cuda_device}:{int(role)}"),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle.devkit,
        "install",
        lambda package=None: events.append("devkit"),
    )
    generation = FabricGenerationId(high=1, low=2)

    class FakeClient:
        def kv_capacity_channel(self, candidate: FabricGenerationId) -> KvCapacityChannelRef:
            assert candidate == generation
            events.append("discover_capacity")
            return KvCapacityChannelRef(generation=generation, name="/xpool-kv-test")

    class FakeInstanceRuntime:
        fabric_plan = None
        client = FakeClient()

        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return SimpleNamespace(generation=generation)

        def attach_arena_from_daemon(self) -> None:
            events.append("attach_transport")

        def start_failure_monitor(self) -> None:
            events.append("start_failure_monitor")

        def publish_initialized(self, serving_listener: ServingListener) -> None:
            events.append("publish_initialized")
            listeners.append(serving_listener)

        def wait_for_ready(self) -> None:
            events.append("wait_for_ready")

        def close(self) -> None:
            events.append("close_instance")

    def fake_instance_init(
        *,
        instance_id: str,
        rank: int,
        transport: InstanceRankTransportProfile,
        ffn_profile: object,
        kv_capacity: object,
    ) -> FakeInstanceRuntime:
        registrations.append((instance_id, rank, transport.payload_row_capacity))
        profiles.append(ffn_profile)
        assert kv_capacity == kv_capacity_profile()
        installs.append((instance_id, rank))
        events.append("start_instance")
        return FakeInstanceRuntime()

    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.InstanceRankRuntime, "start", fake_instance_init)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle.CapacityReconciler,
        "attach",
        lambda **kwargs: events.append("attach_capacity") or SimpleNamespace(close=lambda: None),
    )
    profile = ffn_profile()
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle, "derive_instance_ffn_profile", lambda model_runner, binding: profile
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    result = xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
        (adapter,), original, runner.as_model_runner()
    )
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
    ]

    pool_result = xpool.integrations.sglang.hooks.lifecycle.after_model_runner_alloc_memory_pool(
        None, runner.as_model_runner()
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_worker = SimpleNamespace(model_runner=runner.as_model_runner())
    scheduler.server_args = runner.server_args
    init_info = {"status": "ready"}
    initialize_result = xpool.integrations.sglang.hooks.lifecycle.after_scheduler_get_init_info(init_info, scheduler)

    assert result == "loaded"
    assert pool_result is None
    assert initialize_result is init_info
    assert events == [
        f"init:0:{int(RuntimeRole.INSTANCE)}",
        "devkit",
        "devkit",
        "validate_before_load",
        "bind_runtime",
        "original",
        "validate_after_load",
        "start_instance",
        "wait_for_fabric_executable",
        "discover_capacity",
        "attach_capacity",
        "attach_transport",
        "start_failure_monitor",
        "publish_initialized",
        "wait_for_ready",
    ]
    assert registrations == [(TEST_MODEL_ID, 0, 8)]
    assert installs == [(TEST_MODEL_ID, 0)]
    assert profiles == [profile]
    assert listeners == [ServingListener(host=runner.server_args.host, port=runner.server_args.port)]


def test_scheduler_teardown_releases_xpool_resources_even_when_upstream_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model_runner = SimpleNamespace(device=torch.device("cuda", 0), xpool_runtime=None)
    runtime = SglangInstanceRankRuntime(binding=binding())
    model_runner.xpool_runtime = runtime
    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=model_runner))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(
        SglangInstanceRankRuntime,
        "detach",
        lambda self, runner: events.append("detach"),
    )

    def fail_release(candidate: object) -> None:
        events.append("release")
        raise RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        xpool.integrations.sglang.hooks.lifecycle.around_scheduler_release_host_resources(
            fail_release,
            cast(Scheduler, scheduler),
        )

    assert events == ["release", "synchronize", "detach"]


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
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )

    assert events == ["validate_before_load", "bind_runtime", "original", "validate_after_load"]
    assert runner.xpool_runtime is None


def test_model_runner_hook_clears_binding_when_instance_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    install_fake_elastic_kv(runner)
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)

    def fail_instance_init(*args: object, **kwargs: object) -> None:
        events.append("start_instance")
        raise RuntimeError("install failed")

    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.InstanceRankRuntime, "start", fail_instance_init)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle, "derive_instance_ffn_profile", lambda *args: ffn_profile()
    )

    def original(model_runner: ModelRunner) -> str:
        events.append("original")
        return "loaded"

    assert (
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )
        == "loaded"
    )
    with pytest.raises(RuntimeError, match="install failed"):
        xpool.integrations.sglang.hooks.lifecycle.after_model_runner_alloc_memory_pool(None, runner.as_model_runner())

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
    install_fake_elastic_kv(runner)
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    generation = FabricGenerationId(high=1, low=2)

    class FakeClient:
        def kv_capacity_channel(self, candidate: FabricGenerationId) -> KvCapacityChannelRef:
            assert candidate == generation
            return KvCapacityChannelRef(generation=generation, name="/xpool-kv-test")

    class FakeInstanceRuntime:
        client = FakeClient()

        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return SimpleNamespace(generation=generation)

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

    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.InstanceRankRuntime, "start", start_instance)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle.CapacityReconciler,
        "attach",
        lambda **kwargs: (
            events.append("attach_capacity") or SimpleNamespace(close=lambda: events.append("close_capacity"))
        ),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle, "derive_instance_ffn_profile", lambda *args: ffn_profile()
    )

    xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
        (adapter,), lambda model_runner: None, runner.as_model_runner()
    )
    with pytest.raises(RuntimeError, match="attach failed"):
        xpool.integrations.sglang.hooks.lifecycle.after_model_runner_alloc_memory_pool(None, runner.as_model_runner())

    assert events[-6:] == [
        "start_instance",
        "wait_for_fabric_executable",
        "attach_capacity",
        "attach_transport",
        "close_capacity",
        "close_instance",
    ]
    assert runner.xpool_runtime is None


def test_model_runner_hook_waits_for_executable_fabric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model runner attaches Transport only after Fabric is executable."""

    events: list[str] = []
    adapter = FakeAdapter(matches=True, events=events)
    runner = FakeModelRunner(model_config=FakeModelConfig(model_path=str(tmp_path / "fake-model")))
    install_fake_elastic_kv(runner)
    configure_xpool_model(tmp_path, monkeypatch, runner.model_config.model_path)
    generation = FabricGenerationId(high=1, low=2)

    class FakeClient:
        def kv_capacity_channel(self, candidate: FabricGenerationId) -> KvCapacityChannelRef:
            assert candidate == generation
            events.append("discover_capacity")
            return KvCapacityChannelRef(generation=generation, name="/xpool-kv-test")

    class FakeInstanceRuntime:
        client = FakeClient()

        def wait_for_fabric_executable(self) -> object:
            events.append("wait_for_fabric_executable")
            return SimpleNamespace(generation=generation)

        def attach_arena_from_daemon(self) -> None:
            events.append("attach_transport")

        def start_failure_monitor(self) -> None:
            events.append("start_failure_monitor")

        def close(self) -> None:
            events.append("close_instance")

    def fake_instance_init(*args: object, **kwargs: object) -> FakeInstanceRuntime:
        events.append("register_instance_profile")
        return FakeInstanceRuntime()

    monkeypatch.setattr(xpool.integrations.sglang.hooks.lifecycle.InstanceRankRuntime, "start", fake_instance_init)
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle.CapacityReconciler,
        "attach",
        lambda **kwargs: events.append("attach_capacity") or SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.hooks.lifecycle, "derive_instance_ffn_profile", lambda *args: ffn_profile()
    )

    xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
        (adapter,), lambda model_runner: None, runner.as_model_runner()
    )

    result = xpool.integrations.sglang.hooks.lifecycle.after_model_runner_alloc_memory_pool(
        None, runner.as_model_runner()
    )

    assert result is None
    assert runner.xpool_runtime is not None
    assert events[-6:] == [
        "register_instance_profile",
        "wait_for_fabric_executable",
        "discover_capacity",
        "attach_capacity",
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
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )

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
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )


def test_model_runner_hook_requires_global_xpool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter(matches=True)
    runner = FakeModelRunner()
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    def original(model_runner: ModelRunner) -> str:
        return "loaded"

    with pytest.raises(MissingRequiredConfig, match="global config"):
        xpool.integrations.sglang.hooks.lifecycle.around_model_runner_load_model(
            (adapter,), original, runner.as_model_runner()
        )
