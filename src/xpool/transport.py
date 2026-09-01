"""Typed host lifecycle contracts for native CUDA IPC transport."""

from __future__ import annotations

from dataclasses import dataclass

from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement

__all__ = ["TRANSPORT_ARENA_HANDLE_HEX_LENGTH", "FfnRequestMetadata", "TransportArenaHandle"]

TRANSPORT_ARENA_HANDLE_HEX_LENGTH = 128
"""Character count of a lowercase hexadecimal CUDA IPC memory handle."""


@dataclass(frozen=True, slots=True)
class FfnRequestMetadata:
    """Transport wire metadata for one public FFN shim call.

    Attributes:
        layer_ordinal: Zero-based FFN position in the Fabric workload.
        forward_mode: Prefill, decode, or idle ABI value.
        output_requirement: Mathematical output requirement.
        dp_row_layout: Physical data-parallel row layout selected by the producer.
    """

    layer_ordinal: int
    forward_mode: ForwardMode
    output_requirement: OutputRequirement
    dp_row_layout: DpRowLayout

    def __post_init__(self) -> None:
        """Reject untyped request metadata before dispatcher entry."""

        if isinstance(self.layer_ordinal, bool) or not isinstance(self.layer_ordinal, int) or self.layer_ordinal < 0:
            raise ValueError("xpool FFN layer ordinal must be a non-negative integer")
        if not isinstance(self.forward_mode, ForwardMode):
            raise TypeError("xpool FFN forward mode must be ForwardMode")
        if not isinstance(self.output_requirement, OutputRequirement):
            raise TypeError("xpool FFN output requirement must be OutputRequirement")
        if not isinstance(self.dp_row_layout, DpRowLayout):
            raise TypeError("xpool FFN DP row layout must be DpRowLayout")


@dataclass(frozen=True, slots=True)
class TransportArenaHandle:
    """CUDA IPC transport arena handle brokered by the daemon.

    Attributes:
        handle: Lowercase hex CUDA IPC handle used as control-plane identity.
    """

    handle: str

    def __post_init__(self) -> None:
        """Validate the handle at Python transport boundaries."""

        if len(self.handle) != TRANSPORT_ARENA_HANDLE_HEX_LENGTH:
            raise ValueError(
                f"transport arena handle must contain {TRANSPORT_ARENA_HANDLE_HEX_LENGTH} lowercase hex characters"
            )
        if any(character not in "0123456789abcdef" for character in self.handle):
            raise ValueError("transport arena handle must contain only lowercase hex characters")
