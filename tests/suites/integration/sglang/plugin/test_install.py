from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from sglang.srt.arg_groups import overrides
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig, PhaseConfig
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs

import xpool.integrations.sglang.plugin
import xtest.harness.support.config
from xpool.integrations.sglang.hooks.lifecycle import around_runtime_context_publish
from xpool.integrations.sglang.hooks.registry import SglangHook, discover_sglang_hooks
from xtest.harness.support.config import reset_global_config
from xtest.harness.support.sglang.fakes import server_args as make_server_args
from xtest.harness.support.sglang.plugin import reset_plugin_required_hook_targets

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_plugin_required_hook_targets.__name__)


def test_sglang_discovers_xpool_entry_point() -> None:
    discovered = {entry_point.name: entry_point.value for entry_point in entry_points(group="sglang.srt.plugins")}

    assert discovered["xpool"] == "xpool.integrations.sglang.plugin:install"


def test_sglang_plugin_whitelist_finds_xpool(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.plugins import GENERAL_PLUGINS_GROUP, load_plugins_by_group

    monkeypatch.setenv("SGLANG_PLUGINS", "xpool")

    plugins = load_plugins_by_group(GENERAL_PLUGINS_GROUP)

    assert "xpool" in plugins


def test_hook_discovery_collects_kv_lifecycle_and_model_hooks() -> None:
    targets = {hook.target for hook in discover_sglang_hooks()}

    assert "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool" in targets
    assert "sglang.srt.model_executor.model_runner.ModelRunner.load_model" in targets
    assert "sglang.srt.models.qwen3.Qwen3MLP" in targets
    assert "sglang.srt.runtime_context.publish" in targets
    assert "sglang.srt.entrypoints.engine.Engine._launch_subprocesses" in targets
    assert "sglang.srt.utils.common.kill_process_tree" in targets
    assert "sglang.srt.entrypoints.engine.Engine._terminate_weight_cache_daemons" in targets


@pytest.mark.parametrize(
    ("configured", "backend", "expected"),
    [
        (None, Backend.FULL, 32),
        (64, Backend.FULL, 64),
        (None, Backend.DISABLED, None),
    ],
    ids=["graph-default", "configured", "eager"],
)
def test_runtime_context_publish_defaults_request_concurrency_to_decode_graph(
    monkeypatch: pytest.MonkeyPatch,
    configured: int | None,
    backend: str,
    expected: int | None,
) -> None:
    server_args = make_server_args(max_running_requests=configured)
    overrides.declare_resolution(
        server_args,
        "test Decode Graph configuration",
        cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend=backend, max_bs=32)),
    )
    monkeypatch.setattr(ServerArgs, "resolve_once", lambda self: None)

    def publish(args: object, *, role: str, hf_config: object | None) -> str:
        assert args is server_args
        assert role == "scheduler"
        assert hf_config is None
        return "published"

    result = around_runtime_context_publish(publish, server_args, role="scheduler")

    assert result == "published"
    assert overrides.resolution_result(server_args, "max_running_requests") == expected


def test_plugin_applies_discovered_hooks_with_pinned_sglang_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hook(monkeypatch, "xtest.harness.support.config.minimal_config")

    xpool.integrations.sglang.plugin.install()
    HookRegistry.apply_hooks()

    assert xtest.harness.support.config.minimal_config() == "patched"


def test_plugin_sets_required_engine_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hook(monkeypatch, "xtest.harness.support.config.minimal_config")

    xpool.integrations.sglang.plugin.install()

    assert envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.get() is True
    assert envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() is False
    assert envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get() is False


def test_plugin_rejects_incorrect_mps_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin, "init_global_config", xtest.harness.support.config.minimal_config
    )
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/external/pipe")
    with pytest.raises(SystemExit, match="CUDA_MPS_PIPE_DIRECTORY"):
        xpool.integrations.sglang.plugin.install()


def test_plugin_apply_hooks_guard_fails_closed_with_pinned_sglang_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hook(monkeypatch, "xtest.harness.support.sglang.plugin.missing_target")

    xpool.integrations.sglang.plugin.install()
    with pytest.raises(
        SystemExit,
        match=r"xtest\.harness\.support\.sglang\.plugin\.missing_target",
    ):
        HookRegistry.apply_hooks()


def test_plugin_fails_closed_when_hook_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_discovery() -> tuple[SglangHook, ...]:
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(
        xpool.integrations.sglang.plugin,
        "init_global_config",
        xtest.harness.support.config.minimal_config,
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin,
        "discover_sglang_hooks",
        fail_discovery,
    )

    with pytest.raises(SystemExit, match="discovery failed"):
        xpool.integrations.sglang.plugin.install()


def install_fake_hook(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    monkeypatch.setattr(
        xtest.harness.support.config,
        "minimal_config",
        xtest.harness.support.config.minimal_config,
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin, "init_global_config", xtest.harness.support.config.minimal_config
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin,
        "discover_sglang_hooks",
        lambda: (
            SglangHook(
                target,
                lambda: "patched",
                HookType.REPLACE,
            ),
        ),
    )
