from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import xpool.bootstrap
from tests.harness.pytest_plugin import reset_global_config
from xpool.config import DebugConfig
from xpool.runtime import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_bootstrap_initializes_native_runtime_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical bootstrap calls share one successful native initialization."""

    events: list[tuple[str | RuntimeRole | int | None, ...]] = []
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: events.append(("load",)))
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", lambda title: events.append(("title", title)))
    monkeypatch.setattr(
        xpool.bootstrap.xpool.native,
        "initialize",
        lambda role, cuda_device, debug_options: events.append(("init", role, cuda_device, debug_options)),
    )

    xpool.bootstrap.init(2, RuntimeRole.INSTANCE)
    xpool.bootstrap.init(2, RuntimeRole.INSTANCE)

    assert xpool.bootstrap.get_runtime_role() is RuntimeRole.INSTANCE
    assert events[0] == ("load",)
    assert events[1][:3] == ("init", RuntimeRole.INSTANCE, 2)
    assert json.loads(str(events[1][3])) == {
        "loopback": {"enable": False, "site": None},
        "transport_observer": {"enable": False, "trace_capacity": 8192},
        "fabric_observer": {"enable": False, "trace_capacity": 8192},
    }
    assert len(events) == 2


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
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", lambda value: events.append(("title", value)))

    xpool.bootstrap.init(cuda_device, role)

    assert events[0] == ("load",)
    assert events[1][0:3] == ("init", role, cuda_device)
    assert events[2] == ("title", title)
    assert xpool.bootstrap.get_runtime_role() is role


@pytest.mark.parametrize(
    ("cuda_device", "role"),
    [(3, RuntimeRole.INSTANCE), (2, RuntimeRole.ATNAGENT)],
)
def test_bootstrap_rejects_conflicting_reinitialization(
    cuda_device: int,
    role: RuntimeRole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process cannot change CUDA device or runtime role after bootstrap."""

    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: None)
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    monkeypatch.setattr(xpool.bootstrap.xpool.native, "initialize", lambda role, cuda_device, debug_options: None)
    xpool.bootstrap.init(2, RuntimeRole.INSTANCE)

    with pytest.raises(RuntimeError, match="already initialized"):
        xpool.bootstrap.init(cuda_device, role)


def test_bootstrap_does_not_commit_failed_native_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed native initialization leaves Python bootstrap retryable."""

    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: None)
    monkeypatch.setattr(xpool.bootstrap, "get_global_config", lambda: SimpleNamespace(debug=DebugConfig()))
    process_titles: list[str] = []
    monkeypatch.setattr(xpool.bootstrap, "set_process_title", process_titles.append)

    def fail_init(role: RuntimeRole, cuda_device: int, debug_options: str) -> None:
        raise RuntimeError("native init failed")

    monkeypatch.setattr(xpool.bootstrap.xpool.native, "initialize", fail_init)

    with pytest.raises(RuntimeError, match="native init failed"):
        xpool.bootstrap.init(0, RuntimeRole.ATNAGENT)
    with pytest.raises(RuntimeError, match="before bootstrap"):
        xpool.bootstrap.get_runtime_role()
    assert process_titles == []


def test_daemon_bootstrap_rejects_cuda_device_before_native_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host-only daemon cannot bootstrap with a CUDA device."""

    native_loads: list[None] = []
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: native_loads.append(None))

    with pytest.raises(RuntimeError, match="must not own a CUDA device"):
        xpool.bootstrap.init(0, RuntimeRole.DAEMON)

    assert native_loads == []


@pytest.mark.parametrize("role", [RuntimeRole.INSTANCE, RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT])
def test_gpu_runtime_bootstrap_requires_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    role: RuntimeRole,
) -> None:
    monkeypatch.setattr(xpool.bootstrap.xpool.cext, "ensure_native_loaded", lambda: None)
    monkeypatch.setattr(
        xpool.bootstrap.xpool.native,
        "initialize",
        lambda *args: pytest.fail("native initialization must not run without a CUDA device"),
    )

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        xpool.bootstrap.init(None, role)


def test_get_runtime_role_requires_bootstrap() -> None:
    """Runtime role access fails clearly before process bootstrap."""

    with pytest.raises(RuntimeError, match="before bootstrap"):
        xpool.bootstrap.get_runtime_role()
