"""Runtime loader for the build-time xpool C extension."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import as_file, files
from pathlib import Path
from threading import Lock

import torch

from xpool.abi import ABI_VERSION

__all__ = ["NativeLoadError", "ensure_xpool_ops_loaded"]


class NativeLoadError(RuntimeError):
    """Raised when the xpool native library or ABI contract is unavailable."""


native_ops_loaded = False
native_load_lock = Lock()


def ensure_xpool_ops_loaded() -> None:
    """Load and preflight the installed xpool Torch operators once per process.

    Raises:
        NativeLoadError: If ``libxpool_cext.so`` cannot be located, the
            dynamic loader rejects it, the native ABI probe is missing, or the
            native ABI version does not match Python. Also raised if Python
            graph-wrapper op registration via ``xpool.ops`` fails after the
            native ABI preflight succeeds.

    Side Effects:
        Calls ``torch.ops.load_library`` and imports ``xpool.ops`` on first
        success. Later calls return without touching the dispatcher.
    """

    global native_ops_loaded
    if native_ops_loaded:
        return
    with native_load_lock:
        if native_ops_loaded:
            return

        resource = files("xpool").joinpath("libxpool_cext.so")
        with as_file(resource) as package_path:
            if package_path.is_file():
                load_library_and_prewarm(package_path)
                native_ops_loaded = True
                return
        try:
            distribution_path = Path(str(distribution("xpool").locate_file("xpool/libxpool_cext.so")))
        except PackageNotFoundError as exc:
            raise NativeLoadError("xpool is not installed; cannot locate libxpool_cext.so") from exc
        if distribution_path.is_file():
            load_library_and_prewarm(distribution_path)
            native_ops_loaded = True
            return
        raise NativeLoadError(
            f"xpool C extension is not installed at {package_path} or {distribution_path}; "
            "reinstall xpool so CMake builds and installs libxpool_cext.so"
        )


def load_library_and_prewarm(path: Path) -> None:
    """Load one native library candidate and preflight its Python bindings."""

    try:
        torch.ops.load_library(str(path))
    except OSError as exc:
        raise NativeLoadError(f"failed to load xpool C extension from {path}") from exc
    check_native_abi_version()
    prewarm_python_ops()


def check_native_abi_version() -> None:
    """Verify that the loaded native extension matches python-side ABI version.

    Raises:
        NativeLoadError: If the native extension does not expose
            ``torch.ops.xpool.abi_version`` or returns a version different from
            ``xpool.abi.ABI_VERSION``.

    Side Effects:
        Calls the native ABI-version Torch operator.
    """

    try:
        native_version = int(torch.ops.xpool.abi_version())
    except AttributeError as exc:
        raise NativeLoadError("xpool abi_version op is unavailable after loading libxpool_cext.so") from exc
    if native_version != ABI_VERSION:
        raise NativeLoadError(f"xpool native ABI version {native_version} does not match python-side ABI {ABI_VERSION}")


def prewarm_python_ops() -> None:
    """Import Python graph wrappers after native op preflight succeeds.

    Raises:
        NativeLoadError: If the Python custom-op wrapper module cannot be
            imported and registered.

    Side Effects:
        Imports ``xpool.ops`` so callers can use ``xpool.ops.*`` without a
        separate warmup import.
    """

    try:
        import_module("xpool.ops")
    except Exception as exc:
        raise NativeLoadError("xpool Python graph wrapper ops are unavailable after loading libxpool_cext.so") from exc
