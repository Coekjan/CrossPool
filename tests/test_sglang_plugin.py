import pkgutil
from importlib.metadata import entry_points

import pytest

from xpool.sglang_plugin import FFN_FORWARD_TARGETS, _around_ffn_forward
from xpool.shim import ShimUnavailableError


def test_sglang_discovers_xpool_entry_point() -> None:
    discovered = {entry_point.name: entry_point.value for entry_point in entry_points(group="sglang.srt.plugins")}

    assert discovered["xpool"] == "xpool.sglang_plugin:install"


def test_sglang_plugin_whitelist_finds_xpool(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.srt.plugins import GENERAL_PLUGINS_GROUP, load_plugins_by_group

    monkeypatch.setenv("SGLANG_PLUGINS", "xpool")

    plugins = load_plugins_by_group(GENERAL_PLUGINS_GROUP)

    assert "xpool" in plugins


def test_plugin_declares_initial_ffn_targets() -> None:
    assert "sglang.srt.models.deepseek_v2.DeepseekV2MLP.forward" in FFN_FORWARD_TARGETS
    assert "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward" in FFN_FORWARD_TARGETS


def test_plugin_ffn_targets_resolve_against_installed_sglang() -> None:
    for target in FFN_FORWARD_TARGETS:
        object_path, attr_name = target.rsplit(".", 1)
        target_object = pkgutil.resolve_name(object_path)

        assert hasattr(target_object, attr_name)


def test_hook_fails_closed_after_plugin_loads() -> None:
    def original(module: object, value: int) -> int:
        return value + 1

    with pytest.raises(ShimUnavailableError):
        _around_ffn_forward(original, object(), 2)
