"""Typed host lifecycle contracts for native CUDA IPC transport."""

from __future__ import annotations

from dataclasses import dataclass

from xpool.abi import DpPaddingMode, FfnResultHandoff, XPoolForwardMode

__all__ = ["TRANSPORT_ARENA_HANDLE_HEX_LENGTH", "FfnRequestMetadata", "TransportArenaHandle"]

TRANSPORT_ARENA_HANDLE_HEX_LENGTH = 128
"""Character count of a lowercase hexadecimal CUDA IPC memory handle."""


@dataclass(frozen=True, slots=True)
class FfnRequestMetadata:
    """Transport wire metadata for one public FFN shim call.

    Attributes:
        layer_ordinal: Zero-based FFN position in the Fabric workload.
        forward_mode: Decode, extend, or idle ABI value.
        result_handoff: SGLang-facing result handoff contract.
        dp_padding_mode: Data-parallel padding ABI value selected by the producer.
    """

    layer_ordinal: int
    forward_mode: XPoolForwardMode
    result_handoff: FfnResultHandoff
    dp_padding_mode: DpPaddingMode

    def __post_init__(self) -> None:
        """Reject untyped request metadata before dispatcher entry."""

        if isinstance(self.layer_ordinal, bool) or not isinstance(self.layer_ordinal, int) or self.layer_ordinal < 0:
            raise ValueError("xpool FFN layer ordinal must be a non-negative integer")
        if not isinstance(self.forward_mode, XPoolForwardMode):
            raise TypeError("xpool FFN forward mode must be XPoolForwardMode")
        if not isinstance(self.result_handoff, FfnResultHandoff):
            raise TypeError("xpool FFN result handoff must be FfnResultHandoff")
        if not isinstance(self.dp_padding_mode, DpPaddingMode):
            raise TypeError("xpool FFN DP padding mode must be DpPaddingMode")


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
