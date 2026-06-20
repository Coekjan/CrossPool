from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

import xpool.config as config_module
import xpool.devkit.sglang.plugins as devkit_plugins
import xpool.devkit.sglang.plugins.graph_observer as graph_observer
from xpool.config import XpoolConfig, init_global_config


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    yield
    monkeypatch.setattr(config_module, "_global_config", None)


def test_devkit_sglang_plugins_install_enabled_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Path | None] = []
    config = graph_observer_enabled_config(tmp_path)

    def fake_graph_observer_install() -> None:
        events.append(config_module.get_global_config().debug.graph_observer.outdir)

    monkeypatch.setattr(graph_observer, "install", fake_graph_observer_install)
    init_global_config(config=config)

    devkit_plugins.install()

    assert events == [tmp_path.resolve()]


def test_devkit_sglang_plugins_skip_disabled_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_graph_observer_install() -> None:
        raise AssertionError("disabled graph observer should not be installed")

    monkeypatch.setattr(graph_observer, "install", fail_graph_observer_install)
    init_global_config(config=minimal_config())

    devkit_plugins.install()


def test_devkit_sglang_plugins_discover_nested_enabled_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "xpool_devkit_plugin_probe"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "graph_observer.py").write_text(
        """
INSTALLED = False


def install():
    global INSTALLED
    INSTALLED = True
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    plugins = devkit_plugins.discover_sglang_devkit_plugins("xpool_devkit_plugin_probe")

    assert [name for name, _installer in plugins] == ["graph_observer"]
    plugins[0][1]()
    module = importlib.import_module("xpool_devkit_plugin_probe.nested.graph_observer")
    assert module.INSTALLED is True


def test_devkit_sglang_plugins_do_not_import_disabled_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "xpool_devkit_plugin_disabled_probe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph_observer.py").write_text("raise AssertionError('disabled plugin imported')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    init_global_config(config=minimal_config())

    assert devkit_plugins.discover_sglang_devkit_plugins("xpool_devkit_plugin_disabled_probe") == ()


def test_devkit_sglang_plugins_strict_discovery_rejects_broken_enabled_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "xpool_devkit_plugin_strict_probe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph_observer.py").write_text("raise ImportError('missing graph dependency')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    with pytest.raises(RuntimeError, match="xpool_devkit_plugin_strict_probe.graph_observer"):
        devkit_plugins.discover_sglang_devkit_plugins("xpool_devkit_plugin_strict_probe")


def test_devkit_sglang_plugins_reject_missing_install_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "xpool_devkit_plugin_missing_install_probe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph_observer.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    with pytest.raises(RuntimeError, match="does not expose install"):
        devkit_plugins.discover_sglang_devkit_plugins("xpool_devkit_plugin_missing_install_probe")


def graph_observer_enabled_config(outdir: Path) -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )


def minimal_config() -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
