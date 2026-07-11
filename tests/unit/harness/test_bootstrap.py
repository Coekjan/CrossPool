"""Tests for mandatory pytest native bootstrap."""

import pytest

import tests.harness.bootstrap as bootstrap
from xpool.cext import NativeLoadError


def test_ensure_test_native_ops_delegates_to_native_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []
    monkeypatch.setattr(bootstrap, "ensure_xpool_ops_loaded", lambda: calls.append(None))

    bootstrap.ensure_test_native_ops()

    assert calls == [None]


def test_ensure_test_native_ops_wraps_native_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load() -> None:
        raise NativeLoadError("missing library")

    monkeypatch.setattr(bootstrap, "ensure_xpool_ops_loaded", fail_load)

    with pytest.raises(bootstrap.TestBootstrapError, match="native operator preflight failed: missing library"):
        bootstrap.ensure_test_native_ops()
