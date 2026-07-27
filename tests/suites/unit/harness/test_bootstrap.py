"""Tests for mandatory pytest native bootstrap."""

import pytest

import tests.harness.bootstrap
from xpool.cext import NativeLoadError


def test_ensure_test_native_wraps_native_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load() -> None:
        raise NativeLoadError("missing library")

    monkeypatch.setattr(tests.harness.bootstrap, "ensure_native_loaded", fail_load)

    with pytest.raises(tests.harness.bootstrap.TestBootstrapError, match="native preflight failed: missing library"):
        tests.harness.bootstrap.ensure_test_native()
