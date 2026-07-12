from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import xpool.bootstrap as bootstrap_module
import xpool.devkit.registry as devkit_registry
import xpool.devkit.sglang.graph_observer as graph_observer
from xpool.abi import RuntimeRole
from xpool.config import XpoolConfig, init_global_config


def test_registry_installs_enabled_observer_for_runtime_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled observers install when they support the bootstrapped role."""

    events: list[Path | None] = []
    config = graph_observer_enabled_config(tmp_path)
    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.INSTANCE)
    monkeypatch.setattr(
        graph_observer,
        "install",
        lambda: events.append(config.debug.graph_observer.outdir),
    )
    init_global_config(config=config)

    devkit_registry.install()

    assert events == [tmp_path.resolve()]


def test_registry_skips_observer_for_different_runtime_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled observers do not install in an unsupported process role."""

    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.ATNAGENT)
    monkeypatch.setattr(
        graph_observer,
        "install",
        lambda: pytest.fail("graph observer must not install in a AtnAgent"),
    )
    init_global_config(config=graph_observer_enabled_config(tmp_path))

    devkit_registry.install()


@pytest.mark.parametrize(
    ("runtime_role", "expected"),
    [
        (RuntimeRole.ATNAGENT, ("xpool.devkit.common.transport_observer",)),
        (RuntimeRole.INSTANCE, ()),
    ],
)
def test_registry_filters_common_transport_observer_by_runtime_role(
    runtime_role: RuntimeRole,
    expected: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common transport observation is discoverable only for AtnAgents."""

    monkeypatch.setattr(bootstrap_module, "runtime_role", runtime_role)
    init_global_config(config=transport_observer_enabled_config(tmp_path))

    observers = devkit_registry.discover_devkit_observers(devkit_registry.DEVKIT_PACKAGE)

    assert tuple(name for name, installer in observers) == expected


def test_registry_does_not_import_disabled_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled observer modules are filtered before import."""

    package = create_probe_package(
        tmp_path,
        "disabled_probe",
        {"graph_observer.py": "raise AssertionError('disabled observer imported')\n"},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.INSTANCE)
    init_global_config(config=minimal_config())

    assert devkit_registry.discover_devkit_observers(package) == ()


def test_registry_discovers_nested_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery supports observer modules nested under integration packages."""

    package = create_probe_package(
        tmp_path,
        "nested_probe",
        {
            "nested/__init__.py": "",
            "nested/graph_observer.py": (
                "from xpool.abi import RuntimeRole\n"
                "runtime_roles = frozenset({RuntimeRole.INSTANCE})\n"
                "installed = False\n"
                "def install():\n"
                "    global installed\n"
                "    installed = True\n"
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.INSTANCE)
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    observers = devkit_registry.discover_devkit_observers(package)
    observers[0][1]()
    module = importlib.import_module(f"{package}.nested.graph_observer")

    assert [name for name, installer in observers] == [f"{package}.nested.graph_observer"]
    assert module.installed is True


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def install():\n    return None\n", "runtime_roles"),
        (
            "from xpool.abi import RuntimeRole\nruntime_roles = frozenset({RuntimeRole.INSTANCE})\n",
            "does not expose install",
        ),
    ],
)
def test_registry_rejects_invalid_observer_contract(
    source: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled observers must provide role metadata and an installer."""

    package_name = "invalid_role_probe" if message == "runtime_roles" else "missing_install_probe"
    package = create_probe_package(tmp_path, package_name, {"graph_observer.py": source})
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.INSTANCE)
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    with pytest.raises(RuntimeError, match=message):
        devkit_registry.discover_devkit_observers(package)


def test_registry_rejects_duplicate_config_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two modules cannot claim one debug config field."""

    observer_source = (
        "from xpool.abi import RuntimeRole\n"
        "runtime_roles = frozenset({RuntimeRole.INSTANCE})\n"
        "def install():\n"
        "    return None\n"
    )
    package = create_probe_package(
        tmp_path,
        "duplicate_probe",
        {
            "first/__init__.py": "",
            "first/graph_observer.py": observer_source,
            "second/__init__.py": "",
            "second/graph_observer.py": observer_source,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap_module, "runtime_role", RuntimeRole.INSTANCE)
    init_global_config(config=graph_observer_enabled_config(tmp_path / "events"))

    with pytest.raises(RuntimeError, match="ambiguous"):
        devkit_registry.discover_devkit_observers(package)


def create_probe_package(tmp_path: Path, name: str, files: dict[str, str]) -> str:
    """Create an importable package tree for registry behavior tests."""

    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for relative_path, source in files.items():
        path = package / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return name


def graph_observer_enabled_config(outdir: Path) -> XpoolConfig:
    """Return a config with graph observation enabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )


def transport_observer_enabled_config(outdir: Path) -> XpoolConfig:
    """Return a config with transport observation enabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(outdir),
        },
    )


def minimal_config() -> XpoolConfig:
    """Return a config with every observer disabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
