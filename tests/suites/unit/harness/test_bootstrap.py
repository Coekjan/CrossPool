"""Tests for mandatory pytest native bootstrap."""

import pytest

import tests.harness.runner.bootstrap
from xpool.cext import NativeLoadError


def test_ensure_test_native_wraps_native_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load() -> None:
        raise NativeLoadError("missing library")

    monkeypatch.setattr(tests.harness.runner.bootstrap, "ensure_native_loaded", fail_load)

    with pytest.raises(
        tests.harness.runner.bootstrap.TestBootstrapError, match="native preflight failed: missing library"
    ):
        tests.harness.runner.bootstrap.ensure_test_native()
