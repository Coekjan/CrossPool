"""Mandatory native-operator preflight for Python test sessions."""

from xpool.cext import NativeLoadError, ensure_xpool_ops_loaded


class TestBootstrapError(RuntimeError):
    """Raised when the mandatory test-process bootstrap cannot complete."""


def ensure_test_native_ops() -> None:
    """Load and validate xpool native operators for the current test process.

    Raises:
        TestBootstrapError: If the extension is missing, cannot load, or has an
            incompatible ABI.
    """

    try:
        ensure_xpool_ops_loaded()
    except NativeLoadError as exc:
        raise TestBootstrapError(f"xpool native operator preflight failed: {exc}") from exc
