"""Mandatory native-extension preflight for Python test sessions."""

from importlib import import_module

from xpool.cext import NativeLoadError, ensure_native_loaded


class TestBootstrapError(RuntimeError):
    """Raised when the mandatory test-process bootstrap cannot complete."""


def ensure_test_native() -> None:
    """Validate the native ABI and register the Tensor data-path operator.

    Raises:
        TestBootstrapError: If the extension is missing, cannot load, or has an
            incompatible ABI.
    """

    try:
        ensure_native_loaded()
        import_module("xpool.ops")
    except (NativeLoadError, ImportError, RuntimeError) as exc:
        raise TestBootstrapError(f"xpool native preflight failed: {exc}") from exc
