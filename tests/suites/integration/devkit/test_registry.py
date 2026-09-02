from __future__ import annotations

from pathlib import Path

import pytest

import xpool.devkit.fabric_observer
import xpool.devkit.registry
import xpool.devkit.transport_observer
import xpool.integrations.sglang.devkit.graph_observer
import xpool.integrations.sglang.devkit.prefill_logit_observer
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.native import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_registry_installs_enabled_graph_observer_for_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Graph observer is discovered and installed in an Instance-rank runtime."""

    events: list[Path | None] = []
    config = observer_enabled_config("graph_observer", tmp_path)
    monkeypatch.setattr(xpool.devkit.registry, "get_runtime_role", lambda: RuntimeRole.INSTANCE)
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "install",
        lambda: events.append(config.debug.graph_observer.outdir),
    )
    install_test_config(config=config)

    xpool.devkit.registry.install("xpool.integrations.sglang.devkit")

    assert events == [tmp_path.resolve()]


def test_registry_skips_graph_observer_for_atnagent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Graph observer does not install in an unsupported role."""

    monkeypatch.setattr(xpool.devkit.registry, "get_runtime_role", lambda: RuntimeRole.ATNAGENT)
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "install",
        lambda: pytest.fail("graph observer must not install in an AtnAgent"),
    )
    install_test_config(config=observer_enabled_config("graph_observer", tmp_path))

    xpool.devkit.registry.install("xpool.integrations.sglang.devkit")


def test_registry_installs_enabled_prefill_logit_observer_for_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Path | None] = []
    config = observer_enabled_config("prefill_logit_observer", tmp_path)
    monkeypatch.setattr(xpool.devkit.registry, "get_runtime_role", lambda: RuntimeRole.INSTANCE)
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.prefill_logit_observer,
        "install",
        lambda: events.append(config.debug.prefill_logit_observer.outdir),
    )
    install_test_config(config=config)

    xpool.devkit.registry.install("xpool.integrations.sglang.devkit")

    assert events == [tmp_path.resolve()]


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
