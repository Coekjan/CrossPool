from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import tests.harness.support.sglang.plugin
import xpool.integrations.sglang.plugin
from tests.harness.support.config import reset_global_config
from tests.harness.support.sglang.plugin import reset_plugin_required_hook_targets
from xpool.integrations.sglang.adapter import SglangCudaPlacement
from xpool.integrations.sglang.hooks.registry import SglangHook, discover_sglang_hooks

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_plugin_required_hook_targets.__name__)


def test_sglang_discovers_xpool_entry_point() -> None:
    discovered = {entry_point.name: entry_point.value for entry_point in entry_points(group="sglang.srt.plugins")}

    assert discovered["xpool"] == "xpool.integrations.sglang.plugin:install"


def test_sglang_plugin_whitelist_finds_xpool(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.plugins import GENERAL_PLUGINS_GROUP, load_plugins_by_group

    monkeypatch.setenv("SGLANG_PLUGINS", "xpool")

    plugins = load_plugins_by_group(GENERAL_PLUGINS_GROUP)

    assert "xpool" in plugins


def test_sglang_cuda_placement_is_derived_inside_sglang_integration() -> None:
    placement = SglangCudaPlacement.derive([2, 4, 6])

    assert placement.base_gpu_id == 2
    assert placement.gpu_id_step == 2
    with pytest.raises(RuntimeError, match="base_gpu_id"):
        SglangCudaPlacement.derive([0, 2, 3])


def test_hook_discovery_collects_kv_lifecycle_and_model_hooks() -> None:
    targets = {hook.target for hook in discover_sglang_hooks()}

    assert "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool" in targets
    assert "sglang.srt.model_executor.model_runner.ModelRunner.load_model" in targets
    assert "sglang.srt.models.qwen3.Qwen3MLP" in targets


def test_plugin_applies_discovered_hooks_with_pinned_sglang_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hook(monkeypatch, "tests.harness.support.sglang.plugin.minimal_config")

    xpool.integrations.sglang.plugin.install()
    HookRegistry.apply_hooks()

    assert tests.harness.support.sglang.plugin.minimal_config() == "patched"


def test_plugin_apply_hooks_guard_fails_closed_with_pinned_sglang_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hook(monkeypatch, "tests.harness.support.sglang.plugin.missing_target")

    xpool.integrations.sglang.plugin.install()
    with pytest.raises(
        SystemExit,
        match=r"tests\.harness\.support\.sglang\.plugin\.missing_target",
    ):
        HookRegistry.apply_hooks()


def test_plugin_fails_closed_when_hook_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_discovery() -> tuple[SglangHook, ...]:
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(
        xpool.integrations.sglang.plugin,
        "init_global_config",
        tests.harness.support.sglang.plugin.minimal_config,
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
        tests.harness.support.sglang.plugin,
        "minimal_config",
        tests.harness.support.sglang.plugin.minimal_config,
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.plugin, "init_global_config", tests.harness.support.sglang.plugin.minimal_config
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
