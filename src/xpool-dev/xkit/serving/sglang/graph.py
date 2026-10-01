"""SGLang graph settings shared by test and benchmark launches."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = ["SglangGraphBackend", "SglangGraphMode", "SglangGraphSettings"]

type SglangGraphBackend = Literal["disabled", "full", "breakable"]


@dataclass(frozen=True, slots=True)
class SglangGraphSettings:
    """Decode and prefill CUDA graph backends for one serving process."""

    decode_backend: SglangGraphBackend
    prefill_backend: SglangGraphBackend

    def id(self) -> str:
        """Return the stable artifact suffix for these backend settings."""

        return f"decode-{self.decode_backend}-prefill-{self.prefill_backend}"


class SglangGraphMode(StrEnum):
    """Supported serving modes, independent of test verdicts and observers."""

    EAGER = "eager"
    DECODE_FULL = "decode-full"
    DECODE_FULL_PREFILL_BREAKABLE = "decode-full-prefill-breakable"

    def settings(self) -> SglangGraphSettings:
        """Resolve the pinned engine's decode/prefill backend pair."""

        match self:
            case SglangGraphMode.EAGER:
                return SglangGraphSettings("disabled", "disabled")
            case SglangGraphMode.DECODE_FULL:
                return SglangGraphSettings("full", "disabled")
            case SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE:
                return SglangGraphSettings("full", "breakable")
