"""Stable Python facade for xpool native ABI values and descriptor models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag

ABI_VERSION = 22
TRANSPORT_ARENA_HANDLE_HEX_LENGTH = 128


@dataclass(frozen=True, slots=True)
class TransportTraceRecord:
    """One native transport request's device timestamps.

    All timestamps are CUDA ``globaltimer`` readings in nanoseconds. A zero
    ``trace_id`` marks an unused observer ring row, while zero timestamps mark
    phases that had not completed when the arena was drained.
    """

    trace_id: int
    slot: int
    num_tokens: int
    request_begin: int
    slot_claimed: int
    input_staged: int
    request_published: int
    devagent_dequeued: int
    descriptor_granted: int
    executor_begin: int
    executor_end: int
    result_published: int
    result_observed: int
    output_copied: int
    slot_recycled: int

    @classmethod
    def from_raw(cls, values: list[int]) -> TransportTraceRecord:
        """Decode one raw dispatcher row in canonical ABI field order.

        Args:
            values: Integer values returned by the native dispatcher.

        Returns:
            Validated trace record.

        Raises:
            RuntimeError: If the native row width does not match this ABI.
        """

        if len(values) != len(TRANSPORT_TRACE_FIELDS):
            raise RuntimeError(
                "xpool native transport trace row width does not match the Python ABI: "
                f"expected {len(TRANSPORT_TRACE_FIELDS)}, got {len(values)}"
            )
        return cls(*values)


TRANSPORT_TRACE_FIELDS = tuple(field.name for field in fields(TransportTraceRecord))


@dataclass(frozen=True, slots=True)
class TransportTraceSnapshot:
    """Host-owned snapshot of one native transport observer ring.

    Attributes:
        sequence: Number of trace rows allocated since arena creation.
        dropped: Number of traces that overwrote older ring rows.
        records: Ring storage copied during arena destruction. This is empty
            when transport observation was disabled.
    """

    sequence: int
    dropped: int
    records: tuple[TransportTraceRecord, ...]

    @classmethod
    def from_raw(cls, sequence: int, dropped: int, records: list[list[int]]) -> TransportTraceSnapshot:
        """Decode the raw native dispatcher representation.

        Args:
            sequence: Native observer allocation sequence.
            dropped: Native observer overwrite count.
            records: Native trace rows in ring storage order.

        Returns:
            Immutable ABI snapshot.

        Raises:
            RuntimeError: If any record has an ABI-incompatible width.
        """

        return cls(
            sequence=sequence,
            dropped=dropped,
            records=tuple(TransportTraceRecord.from_raw(record) for record in records),
        )


TRANSPORT_PHASES = (
    ("slot_wait", "request_begin", "slot_claimed"),
    ("input_staging", "slot_claimed", "input_staged"),
    ("request_publish", "input_staged", "request_published"),
    ("publish_to_dequeue", "request_published", "devagent_dequeued"),
    ("descriptor_grant", "devagent_dequeued", "descriptor_granted"),
    ("executor", "executor_begin", "executor_end"),
    ("result_notify", "result_published", "result_observed"),
    ("output_copy", "result_observed", "output_copied"),
    ("slot_recycle", "output_copied", "slot_recycled"),
    ("total", "request_begin", "slot_recycled"),
)


class RuntimeRole(IntEnum):
    """Native runtime process role selected during xpool initialization.

    Attributes:
        INSTANCE: Process that attaches transport arenas and runs FFN shim calls.
        DEVAGENT: Process that owns transport arenas and persistent kernels.
    """

    INSTANCE = 1
    DEVAGENT = 2


class DebugOption(IntFlag):
    """Native process-wide debug option bits passed during xpool initialization.

    Attributes:
        SHIM_LOOPBACK: Route FFN shim calls through direct debug loopback.
        TRANSPORT_LOOPBACK: Route daemon-brokered transport requests through the
            debug persistent transport loopback executor.
        TRANSPORT_OBSERVER: Record native transport device-phase timings.
    """

    SHIM_LOOPBACK = 1 << 0
    TRANSPORT_LOOPBACK = 1 << 1
    TRANSPORT_OBSERVER = 1 << 2


class XPoolForwardMode(IntEnum):
    """Runtime forward modes accepted by the xpool FFN descriptor ABI.

    Attributes:
        EXTEND: Extend/prefill-mode FFN request for a contiguous prompt-token batch.
        DECODE: Decode-mode FFN request for one scheduler decode step.
        IDLE: Data-parallel idle-rank request with no live FFN work.
    """

    EXTEND = 1
    DECODE = 2
    IDLE = 4


class FfnCollectivePolicy(IntEnum):
    """FFN output collective contract represented in request descriptors.

    Attributes:
        FULL_REDUCED: FFN output is fully reduced and may be consumed directly.
        ATN_TP_PARTIAL: FFN output is an additive partial for this attention TP rank.
    """

    FULL_REDUCED = 1
    ATN_TP_PARTIAL = 2


class DpPaddingMode(IntEnum):
    """Data-parallel padding mode represented in FFN request descriptors.

    Attributes:
        NONE: Request is not running through data-parallel synchronization.
        MAX_LEN: Producer padded each DP rank to the maximum rank-local token count.
        SUM_LEN: Producer packed all DP rank token counts into one summed buffer.
    """

    NONE = 0
    MAX_LEN = 1
    SUM_LEN = 2


class DescriptorStatus(IntEnum):
    """Lifecycle states shared by FFN request and result descriptors.

    Attributes:
        EMPTY: Descriptor arena is available for a new request.
        PUBLISHED: Attention-side shim has published a request for polling.
        GRANTED: Devagent has granted the communication slot.
        DONE: FFN execution completed and the output descriptor is valid.
        FAILED: FFN execution failed and the result descriptor carries an error code.
    """

    EMPTY = 0
    PUBLISHED = 1
    GRANTED = 2
    DONE = 3
    FAILED = 4


class FfnResultErrorCode(IntEnum):
    """FFN result descriptor error codes written by native executors.

    Attributes:
        OK: Result completed successfully.
        SHUTDOWN: Arena shutdown failed an in-flight request slot.
        NOT_IMPLEMENTED: The requested executor path has not been implemented.
    """

    OK = 0
    SHUTDOWN = 1
    NOT_IMPLEMENTED = 2


@dataclass(frozen=True, slots=True)
class FfnRequestMetadata:
    """Request metadata for one public FFN shim call.

    Attributes:
        instance_index: Integer instance index from xpool config declaration order.
        layer_id: Decoder layer id whose FFN implementation should consume this request.
        forward_mode: Decode, extend, or idle mode represented in the native ABI.
        collective_policy: Whether the FFN result must be fully reduced or an
            attention TP partial.
        dp_padding_mode: Data-parallel padding mode represented in the FFN
            request descriptor.
        global_dp_buffer_len: Token rows in the producer's global DP buffer
            after padding or packing.
        atn_tp_rank: Attention tensor-parallel rank.
        atn_tp_size: Attention tensor-parallel world size.
        atn_dp_rank: Attention data-parallel rank.
        atn_dp_size: Attention data-parallel world size.
    """

    instance_index: int
    layer_id: int
    forward_mode: int
    collective_policy: int
    dp_padding_mode: int
    global_dp_buffer_len: int
    atn_tp_rank: int
    atn_tp_size: int
    atn_dp_rank: int
    atn_dp_size: int


@dataclass(frozen=True, slots=True)
class TransportArenaHandle:
    """CUDA IPC transport arena handle brokered by the daemon.

    Attributes:
        handle: Lowercase hex CUDA IPC handle used as control-plane identity.
    """

    handle: str

    def __post_init__(self) -> None:
        """Validate the handle at Python API boundaries."""

        if len(self.handle) != TRANSPORT_ARENA_HANDLE_HEX_LENGTH:
            raise ValueError(
                f"transport arena handle must contain {TRANSPORT_ARENA_HANDLE_HEX_LENGTH} lowercase hex characters"
            )
        if any(char not in "0123456789abcdef" for char in self.handle):
            raise ValueError("transport arena handle must contain only lowercase hex characters")
