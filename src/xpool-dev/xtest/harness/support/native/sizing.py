"""Unit-test replacement for pure native allocation-size queries."""

from __future__ import annotations

import pytest

from xpool.runtime.ffnagent import device_memory


def install_native_allocation_sizing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arena_bytes: int = 3328,
    control_bytes: int = 128,
    execution_state_bytes: int = 312,
    fabric_observer_bytes: int = 0,
    routing_record_bytes: int = 0,
) -> None:
    """Replace native sizing calls with deterministic Unit-test values."""

    monkeypatch.setattr(device_memory.xpool.native.fabric, "arena_allocation_bytes", lambda *args: arena_bytes)
    monkeypatch.setattr(
        device_memory.xpool.native.fabric,
        "ffnagent_control_allocation_bytes",
        lambda *args: control_bytes,
    )
    monkeypatch.setattr(
        device_memory.xpool.native.ffnagent,
        "execution_state_allocation_bytes",
        lambda *args: execution_state_bytes,
    )
    monkeypatch.setattr(
        device_memory.xpool.native.devkit.fabric_observer,
        "allocation_bytes",
        lambda *args: fabric_observer_bytes,
    )
    monkeypatch.setattr(
        device_memory.xpool.native.devkit.ffn_routing_observer,
        "allocation_bytes",
        lambda *args: routing_record_bytes,
    )
