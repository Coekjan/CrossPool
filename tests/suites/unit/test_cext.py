from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

import xpool.cext
from xpool.cext import NativeLoadError, ensure_native_loaded
from xpool.native import ABI_VERSION


@pytest.fixture
def reset_cext_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-local ABI preflight flag."""

    monkeypatch.setattr(xpool.cext, "native_loaded", False)


pytestmark = pytest.mark.usefixtures(reset_cext_loader.__name__)


def test_native_preflight_is_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent callers cross the native ABI boundary exactly once."""

    events: list[str] = []
    events_lock = Lock()

    class AbiVersion:
        def __int__(self) -> int:
            with events_lock:
                events.append("preflight")
            return ABI_VERSION

    monkeypatch.setattr(xpool.cext.xpool.native, "ABI_VERSION", AbiVersion())

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(lambda index: ensure_native_loaded(), range(32)))

    assert events == ["preflight"]


def test_native_preflight_rejects_abi_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python refuses an extension built for another ABI revision."""

    monkeypatch.setattr(xpool.cext.xpool.native, "ABI_VERSION", ABI_VERSION + 1)

    with pytest.raises(NativeLoadError, match="does not match"):
        ensure_native_loaded()

    assert xpool.cext.native_loaded is False
