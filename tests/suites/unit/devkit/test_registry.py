from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import xpool.bootstrap
import xpool.devkit.registry
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import create_observer_package, observer_enabled_config, observers_disabled_config
from xpool.runtime import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_registry_does_not_import_disabled_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled observer modules are filtered before import."""

    package = create_observer_package(
        tmp_path,
        "disabled_probe",
        {"graph_observer.py": "raise AssertionError('disabled observer imported')\n"},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.INSTANCE)
    install_test_config(config=observers_disabled_config())

    assert xpool.devkit.registry.discover_devkit_observers(package) == ()


def test_registry_discovers_nested_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery supports observer modules nested under integration packages."""

    package = create_observer_package(
        tmp_path,
        "nested_probe",
        {
            "nested/__init__.py": "",
            "nested/graph_observer.py": (
                "from xpool.runtime import RuntimeRole\n"
                "runtime_roles = frozenset({RuntimeRole.INSTANCE})\n"
                "installed = False\n"
                "def install():\n"
                "    global installed\n"
                "    installed = True\n"
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.INSTANCE)
    install_test_config(config=observer_enabled_config("graph_observer", tmp_path / "events"))

    observers = xpool.devkit.registry.discover_devkit_observers(package)
    observers[0][1]()
    module = importlib.import_module(f"{package}.nested.graph_observer")

    assert [name for name, installer in observers] == [f"{package}.nested.graph_observer"]
    assert module.installed is True


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def install():\n    return None\n", "runtime_roles"),
        (
            "from xpool.runtime import RuntimeRole\nruntime_roles = frozenset({RuntimeRole.INSTANCE})\n",
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
    package = create_observer_package(tmp_path, package_name, {"graph_observer.py": source})
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.INSTANCE)
    install_test_config(config=observer_enabled_config("graph_observer", tmp_path / "events"))

    with pytest.raises(RuntimeError, match=message):
        xpool.devkit.registry.discover_devkit_observers(package)


def test_registry_rejects_duplicate_config_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two modules cannot claim one debug config field."""

    observer_source = (
        "from xpool.runtime import RuntimeRole\n"
        "runtime_roles = frozenset({RuntimeRole.INSTANCE})\n"
        "def install():\n"
        "    return None\n"
    )
    package = create_observer_package(
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
    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.INSTANCE)
    install_test_config(config=observer_enabled_config("graph_observer", tmp_path / "events"))

    with pytest.raises(RuntimeError, match="ambiguous"):
        xpool.devkit.registry.discover_devkit_observers(package)
