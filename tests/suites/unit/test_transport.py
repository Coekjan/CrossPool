from __future__ import annotations

import pytest

from xpool.transport import TransportArenaHandle


def test_transport_arena_handle_validates_canonical_hex() -> None:
    handle = TransportArenaHandle(handle="ab" * 64)

    assert handle.handle == "ab" * 64
    with pytest.raises(ValueError, match="128 lowercase hex"):
        TransportArenaHandle(handle="ab")
    with pytest.raises(ValueError, match="only lowercase hex"):
        TransportArenaHandle(handle="AB" * 64)
