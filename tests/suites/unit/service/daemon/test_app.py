from __future__ import annotations

import pytest

import xpool.service.daemon.app
from xpool.native import RuntimeRole
from xpool.service.daemon.control import ControlPlane
from xtest.harness.support.config import reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_daemon_failure_retains_first_exception_and_retirement_budget() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    control = ControlPlane()
    failure = xpool.service.daemon.app.DaemonFailure(control)

    failure.record(first)
    deadline = control.cleanup_deadline
    failure.record(second)

    assert failure.failed
    assert failure.exception is first
    assert control.admission_closed
    assert deadline is not None
    assert control.cleanup_deadline == deadline


def test_create_daemon_initializes_native_daemon_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daemon construction initializes the process-wide native role."""

    calls: list[tuple[int | None, RuntimeRole]] = []
    monkeypatch.setattr(
        xpool.service.daemon.app.bootstrap,
        "init",
        lambda device, role: calls.append((device, role)),
    )

    app = xpool.service.daemon.app.create_daemon()

    assert calls == [(None, RuntimeRole.DAEMON)]
    assert isinstance(app.state.control_plane, xpool.service.daemon.app.ControlPlane)
    assert isinstance(app.state.daemon_failure, xpool.service.daemon.app.DaemonFailure)
    assert not app.state.daemon_failure.failed


def test_daemon_openapi_operations_have_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every supported HTTP operation exposes useful generated API documentation."""

    monkeypatch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda device, role: None)

    schema = xpool.service.daemon.app.create_daemon().openapi()

    missing = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation.get("description")
    ]
    assert missing == []
