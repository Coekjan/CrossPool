from __future__ import annotations

import pytest

import xpool.bootstrap as bootstrap
from xpool.abi import RuntimeRole


def test_bootstrap_initializes_native_runtime_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical bootstrap calls share one successful native initialization."""

    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(bootstrap, "ensure_xpool_ops_loaded", lambda: events.append(("load",)))
    monkeypatch.setattr(
        bootstrap.xpool.ops,
        "init",
        lambda cuda_device, role: events.append(("init", cuda_device, role)),
    )

    bootstrap.init(2, RuntimeRole.INSTANCE)
    bootstrap.init(2, RuntimeRole.INSTANCE)

    assert bootstrap.get_runtime_role() is RuntimeRole.INSTANCE
    assert events == [("load",), ("init", 2, RuntimeRole.INSTANCE)]


@pytest.mark.parametrize(
    ("cuda_device", "role"),
    [(3, RuntimeRole.INSTANCE), (2, RuntimeRole.DEVAGENT)],
)
def test_bootstrap_rejects_conflicting_reinitialization(
    cuda_device: int,
    role: RuntimeRole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process cannot change CUDA device or runtime role after bootstrap."""

    monkeypatch.setattr(bootstrap, "ensure_xpool_ops_loaded", lambda: None)
    monkeypatch.setattr(bootstrap.xpool.ops, "init", lambda cuda_device, role: None)
    bootstrap.init(2, RuntimeRole.INSTANCE)

    with pytest.raises(RuntimeError, match="already initialized"):
        bootstrap.init(cuda_device, role)


def test_bootstrap_does_not_commit_failed_native_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed native initialization leaves Python bootstrap retryable."""

    monkeypatch.setattr(bootstrap, "ensure_xpool_ops_loaded", lambda: None)

    def fail_init(cuda_device: int, role: RuntimeRole) -> None:
        raise RuntimeError("native init failed")

    monkeypatch.setattr(bootstrap.xpool.ops, "init", fail_init)

    with pytest.raises(RuntimeError, match="native init failed"):
        bootstrap.init(0, RuntimeRole.DEVAGENT)
    with pytest.raises(RuntimeError, match="before bootstrap"):
        bootstrap.get_runtime_role()


def test_get_runtime_role_requires_bootstrap() -> None:
    """Runtime role access fails clearly before process bootstrap."""

    with pytest.raises(RuntimeError, match="before bootstrap"):
        bootstrap.get_runtime_role()
