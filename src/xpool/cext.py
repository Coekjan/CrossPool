"""ABI preflight for the importable xpool native extension."""

from __future__ import annotations

from threading import Lock

import xpool.native
from xpool.abi import ABI_VERSION

__all__ = ["NativeLoadError", "ensure_native_loaded"]


class NativeLoadError(RuntimeError):
    """Raised when the xpool native ABI is incompatible with Python."""


native_loaded = False
native_load_lock = Lock()


def ensure_native_loaded() -> None:
    """Validate the imported native extension once per process.

    Raises:
        NativeLoadError: If ``xpool.native`` reports an ABI version different
            from :data:`xpool.abi.ABI_VERSION`.

    Side Effects:
        Calls the native ABI probe on the first successful invocation. Later
        calls return without crossing the extension boundary.
    """

    global native_loaded
    if native_loaded:
        return
    with native_load_lock:
        if native_loaded:
            return
        native_version = int(xpool.native.abi_version())
        if native_version != ABI_VERSION:
            raise NativeLoadError(
                f"xpool native ABI version {native_version} does not match python-side ABI {ABI_VERSION}"
            )
        native_loaded = True
