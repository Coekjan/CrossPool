import pytest

import xtest.harness.runner.bootstrap
from xpool.cext import NativeLoadError


def test_ensure_test_native_wraps_native_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load() -> None:
        raise NativeLoadError("missing library")

    monkeypatch.setattr(xtest.harness.runner.bootstrap, "ensure_native_loaded", fail_load)

    with pytest.raises(
        xtest.harness.runner.bootstrap.TestBootstrapError, match="native preflight failed: missing library"
    ):
        xtest.harness.runner.bootstrap.ensure_test_native()
