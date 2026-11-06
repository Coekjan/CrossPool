from __future__ import annotations

import pytest

import xpool.cext
from xpool.cext import NativeLoadError, ensure_native_loaded
from xpool.native import ABI_VERSION


def test_native_preflight_accepts_matching_version() -> None:
    ensure_native_loaded()
    ensure_native_loaded()


def test_native_preflight_rejects_abi_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python refuses an extension built for another ABI revision."""

    monkeypatch.setattr(xpool.cext.xpool.native, "ABI_VERSION", ABI_VERSION + 1)

    with pytest.raises(NativeLoadError, match="does not match"):
        ensure_native_loaded()
