"""Stable values shared by Python and native xpool data planes."""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "ABI_VERSION",
    "DpPaddingMode",
    "FfnResultCode",
    "FfnResultHandoff",
    "TensorDType",
    "XPoolForwardMode",
]

ABI_VERSION = 54


class XPoolForwardMode(IntEnum):
    """Runtime forward modes accepted by the xpool FFN ABI.

    Attributes:
        EXTEND: Prefill request carrying contiguous prompt-token rows.
        DECODE: One scheduler decode step.
        IDLE: Data-parallel rank with no live FFN work.
    """

    EXTEND = 1
    DECODE = 2
    IDLE = 4


class FfnResultHandoff(IntEnum):
    """SGLang-facing handoff contract for one FFN result contribution.

    Attributes:
        REPLICATED_FULL: Every result-group recipient receives the complete contribution.
        REDUCE_SCATTER_INPUT: Rank zero receives the complete additive input for
            SGLang's reduce-scatter path.
    """

    REPLICATED_FULL = 1
    REDUCE_SCATTER_INPUT = 2


class DpPaddingMode(IntEnum):
    """Data-parallel padding mode carried by FFN requests.

    Attributes:
        NONE: Request does not use data-parallel synchronization.
        MAX_LEN: Every rank is padded to the maximum rank-local row count.
        SUM_LEN: Rank-local rows are packed into one summed buffer.
    """

    NONE = 0
    MAX_LEN = 1
    SUM_LEN = 2


class TensorDType(IntEnum):
    """Stable hidden-state dtype values shared by native protocols.

    Attributes:
        BF16: Bfloat16 hidden states.
        FP16: IEEE float16 hidden states.
        FP32: IEEE float32 hidden states.
    """

    BF16 = 1
    FP16 = 2
    FP32 = 3


class FfnResultCode(IntEnum):
    """FFN completion result codes shared by Transport and Fabric.

    Attributes:
        OK: Request completed successfully.
        SHUTDOWN: Shutdown failed an in-flight request.
        PROTOCOL_MISMATCH: Device-visible protocol validation failed.
        TIMEOUT: A bounded device protocol phase exceeded its deadline.
        NOT_IMPLEMENTED: A valid requested execution path is unsupported.
    """

    OK = 0
    SHUTDOWN = 1
    PROTOCOL_MISMATCH = 2
    TIMEOUT = 3
    NOT_IMPLEMENTED = 4
