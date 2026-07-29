from __future__ import annotations

from pathlib import Path

import pytest

import xpool.bootstrap
import xpool.devkit.registry
import xpool.devkit.sglang.graph_observer
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.runtime import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_registry_installs_enabled_graph_observer_for_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Graph observer is discovered and installed in an Instance."""

    events: list[Path | None] = []
    config = observer_enabled_config("graph_observer", tmp_path)
    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.INSTANCE)
    monkeypatch.setattr(
        xpool.devkit.sglang.graph_observer,
        "install",
        lambda: events.append(config.debug.graph_observer.outdir),
    )
    install_test_config(config=config)

    xpool.devkit.registry.install()

    assert events == [tmp_path.resolve()]


def test_registry_skips_graph_observer_for_atnagent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real Graph observer does not install in an unsupported role."""

    monkeypatch.setattr(xpool.bootstrap, "runtime_role", RuntimeRole.ATNAGENT)
    monkeypatch.setattr(
        xpool.devkit.sglang.graph_observer,
        "install",
        lambda: pytest.fail("graph observer must not install in an AtnAgent"),
    )
    install_test_config(config=observer_enabled_config("graph_observer", tmp_path))

    xpool.devkit.registry.install()


@pytest.mark.parametrize(
    ("runtime_role", "observer_name", "expected"),
    [
        (RuntimeRole.ATNAGENT, "transport_observer", ("xpool.devkit.common.transport_observer",)),
        (RuntimeRole.FFNAGENT, "fabric_observer", ("xpool.devkit.common.fabric_observer",)),
        (RuntimeRole.INSTANCE, "transport_observer", ()),
    ],
)
def test_registry_filters_real_common_observers_by_runtime_role(
    runtime_role: RuntimeRole,
    observer_name: str,
    expected: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real common observers expose the intended participant-role boundary."""

    monkeypatch.setattr(xpool.bootstrap, "runtime_role", runtime_role)
    install_test_config(config=observer_enabled_config(observer_name, tmp_path))

    observers = xpool.devkit.registry.discover_devkit_observers(xpool.devkit.registry.DEVKIT_PACKAGE)

    assert tuple(name for name, _ in observers) == expected
