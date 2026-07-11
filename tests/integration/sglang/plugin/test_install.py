from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points
from typing import ClassVar

from tests.harness.sglang.plugin import (
    FakeAdapter,
    FakeHookRegistry,
    HookType,
    SglangHook,
    SglangHookHandler,
    minimal_config,
    pytest,
    sglang_plugin,
)
from xpool.integrations.sglang.adapter import derive_sglang_cuda_placement


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


def test_sglang_cuda_placement_is_derived_inside_sglang_integration() -> None:
    placement = derive_sglang_cuda_placement([2, 4, 6])

    assert placement.base_gpu_id == 2
    assert placement.gpu_id_step == 2
    with pytest.raises(RuntimeError, match="base_gpu_id"):
        derive_sglang_cuda_placement([0, 2, 3])


def test_plugin_registers_adapter_owned_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHookRegistry.calls = []
    events: list[str] = []

    def fake_hook() -> None:
        return None

    adapter = FakeAdapter(hooks=(SglangHook("xpool.fake.Target", fake_hook, HookType.REPLACE),))

    def fake_adapters(package_name: str) -> tuple[FakeAdapter]:
        events.append("discover_adapters")
        return (adapter,)

    monkeypatch.setattr(sglang_plugin, "init_global_config", lambda: events.append("init_config") or minimal_config())
    monkeypatch.setattr(sglang_plugin, "discover_sglang_model_adapters", fake_adapters)
    monkeypatch.setattr(sglang_plugin, "HookRegistry", FakeHookRegistry)

    sglang_plugin.install()

    assert events == ["init_config", "discover_adapters"]
    assert ("xpool.fake.Target", fake_hook, HookType.REPLACE) in FakeHookRegistry.calls
    assert any(
        target == sglang_plugin.MODEL_RUNNER_LOAD_MODEL and registered_type is HookType.AROUND
        for target, handler, registered_type in FakeHookRegistry.calls
    )
    assert any(
        target == sglang_plugin.MODEL_RUNNER_INIT_MEMORY_POOL and registered_type is HookType.AFTER
        for target, handler, registered_type in FakeHookRegistry.calls
    )


def test_plugin_apply_hooks_guard_fails_closed_when_required_target_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardedHookRegistry:
        calls: ClassVar[list[tuple[str, SglangHookHandler, HookType]]] = []
        _patched: ClassVar[set[str]] = {
            sglang_plugin.MODEL_RUNNER_LOAD_MODEL,
            sglang_plugin.MODEL_RUNNER_INIT_MEMORY_POOL,
        }

        @classmethod
        def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
            cls.calls.append((target, handler, hook_type))

        @classmethod
        def apply_hooks(cls) -> None:
            return None

    adapter = FakeAdapter(hooks=(SglangHook("xpool.fake.Target", lambda: None, HookType.REPLACE),))
    monkeypatch.setattr(sglang_plugin, "init_global_config", minimal_config)
    monkeypatch.setattr(sglang_plugin, "discover_sglang_model_adapters", lambda package_name: (adapter,))
    monkeypatch.setattr(sglang_plugin, "HookRegistry", GuardedHookRegistry)

    sglang_plugin.install()

    with pytest.raises(SystemExit, match=r"xpool\.fake\.Target"):
        GuardedHookRegistry.apply_hooks()


def test_plugin_apply_hooks_guard_allows_all_required_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    class GuardedHookRegistry:
        calls: ClassVar[list[tuple[str, SglangHookHandler, HookType]]] = []
        _patched: ClassVar[set[str]] = {
            "xpool.fake.Target",
            sglang_plugin.MODEL_RUNNER_LOAD_MODEL,
            sglang_plugin.MODEL_RUNNER_INIT_MEMORY_POOL,
        }

        @classmethod
        def register(cls, target: str, handler: SglangHookHandler, hook_type: HookType) -> None:
            cls.calls.append((target, handler, hook_type))

        @classmethod
        def apply_hooks(cls) -> None:
            return None

    adapter = FakeAdapter(hooks=(SglangHook("xpool.fake.Target", lambda: None, HookType.REPLACE),))
    monkeypatch.setattr(sglang_plugin, "init_global_config", minimal_config)
    monkeypatch.setattr(sglang_plugin, "discover_sglang_model_adapters", lambda package_name: (adapter,))
    monkeypatch.setattr(sglang_plugin, "HookRegistry", GuardedHookRegistry)

    sglang_plugin.install()

    GuardedHookRegistry.apply_hooks()
