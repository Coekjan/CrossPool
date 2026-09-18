from __future__ import annotations

from pathlib import Path

import pytest

import xpool.devkit.fabric_observer
import xpool.devkit.registry
import xpool.devkit.transport_observer
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.native import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.mark.parametrize(
    ("runtime_role", "observer_name", "expected_install_count"),
    [
        (RuntimeRole.ATNAGENT, "transport_observer", 1),
        (RuntimeRole.FFNAGENT, "fabric_observer", 1),
        (RuntimeRole.INSTANCE, "transport_observer", 1),
    ],
)
def test_registry_filters_core_observers_by_runtime_role(
    runtime_role: RuntimeRole,
    observer_name: str,
    expected_install_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core observers expose the intended participant-role boundary."""

    monkeypatch.setattr(xpool.devkit.registry, "get_runtime_role", lambda: runtime_role)
    install_test_config(config=observer_enabled_config(observer_name, tmp_path))
    events: list[str] = []
    module = xpool.devkit.transport_observer if observer_name == "transport_observer" else xpool.devkit.fabric_observer
    monkeypatch.setattr(module, "install", lambda: events.append(observer_name))

    xpool.devkit.registry.install()

    assert len(events) == expected_install_count
