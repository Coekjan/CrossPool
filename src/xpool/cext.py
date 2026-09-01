"""ABI preflight for the importable xpool native extension."""

from __future__ import annotations

from threading import Lock

import xpool.native

__all__ = ["NativeLoadError", "ensure_native_loaded"]

EXPECTED_NATIVE_ABI_VERSION = 79


class NativeLoadError(RuntimeError):
    """Raised when the xpool native ABI is incompatible with Python."""


native_loaded = False
native_load_lock = Lock()


def ensure_native_loaded() -> None:
    """Validate the imported native extension once per process.

    Raises:
        NativeLoadError: If the installed extension has a different ABI.

    Side Effects:
        Reads native ABI metadata on the first successful invocation. Later
        calls return without crossing the extension boundary.
    """

    global native_loaded
    if native_loaded:
        return
    with native_load_lock:
        if native_loaded:
            return
        native_version = int(xpool.native.ABI_VERSION)
        if native_version != EXPECTED_NATIVE_ABI_VERSION:
            raise NativeLoadError(
                f"xpool native ABI version {native_version} does not match expected ABI {EXPECTED_NATIVE_ABI_VERSION}"
            )
        native_loaded = True
