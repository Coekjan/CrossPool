from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import xpool.bootstrap
from tests.harness.support.config import reset_global_config
from xpool.config import DebugConfig
from xpool.native import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def disable_runtime_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep bootstrap tests focused on native initialization ordering."""

    monkeypatch.setattr(xpool.bootstrap.xpool.logging, "configure", lambda role: None)


def test_bootstrap_initializes_native_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap delegates identity ownership before Python-side setup."""

    events: list[tuple[str | RuntimeRole | int | None, ...]] = []
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: events.append(("load",)))
    monkeypatch.setattr(
        xpool.bootstrap.xpool.logging,
        "configure",
        lambda role: events.append(("logging", role)),
    )
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", lambda title: events.append(("title", title)))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("device", device)))
    monkeypatch.setattr(
        xpool.bootstrap.xpool.native,
        "initialize",
        lambda role, cuda_device, debug_options: events.append(("init", role, cuda_device, debug_options)),
    )
    monkeypatch.setattr(xpool.bootstrap.xpool.native, "runtime_role", lambda: RuntimeRole.INSTANCE)

    xpool.bootstrap.init(2, RuntimeRole.INSTANCE)

    assert xpool.bootstrap.get_runtime_role() is RuntimeRole.INSTANCE
    assert events[0] == ("load",)
    assert events[1] == ("logging", RuntimeRole.INSTANCE)
    assert events[2][:3] == ("init", RuntimeRole.INSTANCE, 2)
    debug_options = events[2][3]
    assert isinstance(debug_options, xpool.bootstrap.xpool.native.debug.Options)
    assert debug_options.transport_observer.record_capacity == 8192
    assert debug_options.fabric_observer.record_capacity == 8192
    assert debug_options.graph_observer.enable is False
    assert debug_options.ffn_routing_observer.record_capacity == 8
    assert events[3] == ("device", 2)
    assert len(events) == 4


@pytest.mark.parametrize(
    ("cuda_device", "role", "title"),
    [
        (None, RuntimeRole.DAEMON, "xpool::daemon"),
        (0, RuntimeRole.ATNAGENT, "xpool::atnagent"),
        (1, RuntimeRole.FFNAGENT, "xpool::ffnagent"),
    ],
)
def test_bootstrap_sets_resident_process_title_after_native_initialization(
    cuda_device: int | None,
    role: RuntimeRole,
    title: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful resident-role bootstrap installs its stable process title."""

    events: list[tuple[str | RuntimeRole | int | None, ...]] = []
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: events.append(("load",)))
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    monkeypatch.setattr(
        xpool.bootstrap.xpool.native,
        "initialize",
        lambda native_role, native_device=None, debug_options=None: events.append(
            ("init", native_role, native_device, debug_options)
        ),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("device", device)))
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", lambda value: events.append(("title", value)))
    monkeypatch.setattr(xpool.bootstrap.xpool.native, "runtime_role", lambda: role)

    xpool.bootstrap.init(cuda_device, role)

    assert events[0] == ("load",)
    assert events[1][0:3] == ("init", role, cuda_device)
    assert isinstance(events[1][3], xpool.bootstrap.xpool.native.debug.Options)
    if cuda_device is None:
        assert events[2] == ("title", title)
    else:
        assert events[2:] == [("device", cuda_device), ("title", title)]
    assert xpool.bootstrap.get_runtime_role() is role


def test_bootstrap_stops_after_failed_native_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed native initialization prevents Python-side setup."""

    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: None)
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    process_titles: list[str] = []
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", process_titles.append)

    def fail_init(
        role: RuntimeRole,
        cuda_device: int,
        debug_options: xpool.bootstrap.xpool.native.debug.Options,
    ) -> None:
        raise RuntimeError("native init failed")

    monkeypatch.setattr(xpool.bootstrap.xpool.native, "initialize", fail_init)

    with pytest.raises(RuntimeError, match="native init failed"):
        xpool.bootstrap.init(0, RuntimeRole.ATNAGENT)
    assert process_titles == []


def test_daemon_bootstrap_rejects_cuda_device_before_native_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host-only daemon delegates CUDA-device rejection to native."""

    native_loads: list[None] = []
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: native_loads.append(None))
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))

    def reject_daemon_device(
        role: RuntimeRole,
        cuda_device: int | None,
        debug_options: xpool.bootstrap.xpool.native.debug.Options,
    ) -> None:
        assert (role, cuda_device) == (RuntimeRole.DAEMON, 0)
        raise RuntimeError("must not own a CUDA device")

    monkeypatch.setattr(xpool.bootstrap.xpool.native, "initialize", reject_daemon_device)

    with pytest.raises(RuntimeError, match="must not own a CUDA device"):
        xpool.bootstrap.init(0, RuntimeRole.DAEMON)

    assert native_loads == [None]


@pytest.mark.parametrize("role", [RuntimeRole.INSTANCE, RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT])
def test_gpu_runtime_bootstrap_requires_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    role: RuntimeRole,
) -> None:
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: None)
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))

    def reject_missing_device(
        native_role: RuntimeRole,
        cuda_device: int | None,
        debug_options: xpool.bootstrap.xpool.native.debug.Options,
    ) -> None:
        assert (native_role, cuda_device) == (role, None)
        raise RuntimeError("requires a CUDA device")

    monkeypatch.setattr(xpool.bootstrap.xpool.native, "initialize", reject_missing_device)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        xpool.bootstrap.init(None, role)


def test_get_runtime_role_requires_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime role access fails clearly before process bootstrap."""

    def fail_role() -> RuntimeRole:
        raise RuntimeError("before bootstrap")

    monkeypatch.setattr(xpool.bootstrap.xpool.native, "runtime_role", fail_role)
    with pytest.raises(RuntimeError, match="before bootstrap"):
        xpool.bootstrap.get_runtime_role()
