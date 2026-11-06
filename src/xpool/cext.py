"""ABI preflight for the importable CrossPool native extension."""

from __future__ import annotations

import xpool.native

__all__ = ["NativeLoadError", "ensure_native_loaded"]

EXPECTED_NATIVE_ABI_VERSION = 93


class NativeLoadError(RuntimeError):
    """Raised when the CrossPool native ABI is incompatible with Python."""


def ensure_native_loaded() -> None:
    """Validate the imported native extension's ABI metadata.

    Raises:
        NativeLoadError: If the installed extension has a different ABI.

    Side Effects:
        Reads the imported extension's native ABI version on every invocation.
    """

    native_version = int(xpool.native.ABI_VERSION)
    if native_version != EXPECTED_NATIVE_ABI_VERSION:
        raise NativeLoadError(
            f"xpool native ABI version {native_version} does not match expected ABI {EXPECTED_NATIVE_ABI_VERSION}"
        )
