"""Runtime loader for the build-time xpool C extension."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import as_file, files
from pathlib import Path
from threading import Lock

import torch

from xpool.abi import ABI_VERSION


class NativeLoadError(RuntimeError):
    """Raised when the xpool native library or ABI contract is unavailable."""


_loaded = False
_load_lock = Lock()


def ensure_xpool_ops_loaded() -> None:
    """Load and preflight the installed xpool Torch operators once per process.

    Raises:
        NativeLoadError: If ``libxpool_cext.so`` cannot be located, the
            dynamic loader rejects it, the native ABI probe is missing, or the
            native ABI version does not match Python.

    Side Effects:
        Calls ``torch.ops.load_library`` on first success. Later calls return
        without touching the dispatcher.
    """

    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        resource = files("xpool").joinpath("libxpool_cext.so")
        with as_file(resource) as package_path:
            if package_path.is_file():
                try:
                    torch.ops.load_library(str(package_path))
                except OSError as exc:
                    raise NativeLoadError(f"failed to load xpool C extension from {package_path}") from exc
                _check_native_abi_version()
                _loaded = True
                return
        try:
            distribution_path = Path(str(distribution("xpool").locate_file("xpool/libxpool_cext.so")))
        except PackageNotFoundError as exc:
            raise NativeLoadError("xpool is not installed; cannot locate libxpool_cext.so") from exc
        if distribution_path.is_file():
            try:
                torch.ops.load_library(str(distribution_path))
            except OSError as exc:
                raise NativeLoadError(f"failed to load xpool C extension from {distribution_path}") from exc
            _check_native_abi_version()
            _loaded = True
            return
        raise NativeLoadError(
            f"xpool C extension is not installed at {package_path} or {distribution_path}; "
            "reinstall xpool so CMake builds and installs libxpool_cext.so"
        )


def _check_native_abi_version() -> None:
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
