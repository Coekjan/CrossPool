from __future__ import annotations

import pytest

from xpool.abi import (
    ABI_VERSION,
    TRANSPORT_TRACE_FIELDS,
    DebugLoopbackSite,
    DebugOption,
    DebugOptions,
    DescriptorStatus,
    DpPaddingMode,
    FfnCollectivePolicy,
    FfnResultErrorCode,
    RuntimeRole,
    TransportTraceRecord,
    TransportTraceSnapshot,
    XPoolForwardMode,
)


def test_python_abi_enum_values_match_native_contract() -> None:
    """Python ABI enum integers match the on-wire native contract."""

    assert int(XPoolForwardMode.EXTEND) == 1
    assert int(XPoolForwardMode.DECODE) == 2
    assert int(XPoolForwardMode.IDLE) == 4
    assert int(RuntimeRole.INSTANCE) == 1
    assert int(RuntimeRole.ATNAGENT) == 2
    assert int(RuntimeRole.FFNAGENT) == 3
    assert int(DebugOption.LOOPBACK) == 1 << 32
    assert int(DebugOption.TRANSPORT_OBSERVER) == 1 << 33
    assert int(DebugLoopbackSite.NONE) == 0
    assert int(DebugLoopbackSite.INSTANCE) == 1
    assert int(DebugLoopbackSite.ATNAGENT) == 2
    assert int(DebugLoopbackSite.FFNAGENT) == 3
    assert int(FfnCollectivePolicy.FULL_REDUCED) == 1
    assert int(FfnCollectivePolicy.ATN_TP_PARTIAL) == 2
    assert int(DpPaddingMode.NONE) == 0
    assert int(DpPaddingMode.MAX_LEN) == 1
    assert int(DpPaddingMode.SUM_LEN) == 2
    assert int(DescriptorStatus.EMPTY) == 0
    assert int(DescriptorStatus.PUBLISHED) == 1
    assert int(DescriptorStatus.GRANTED) == 2
    assert int(DescriptorStatus.DONE) == 3
    assert int(DescriptorStatus.FAILED) == 4
    assert int(FfnResultErrorCode.OK) == 0
    assert int(FfnResultErrorCode.SHUTDOWN) == 1
    assert int(FfnResultErrorCode.NOT_IMPLEMENTED) == 2


def test_transport_trace_snapshot_decodes_canonical_wire_order() -> None:
    """Structured transport traces preserve the native ABI field order."""

    assert ABI_VERSION == 24
    assert TRANSPORT_TRACE_FIELDS == (
        "trace_id",
        "slot",
        "num_tokens",
        "request_begin",
        "slot_claimed",
        "input_staged",
        "request_published",
        "atnagent_dequeued",
        "descriptor_granted",
        "executor_begin",
        "executor_end",
        "result_published",
        "result_observed",
        "output_copied",
        "slot_recycled",
    )
    snapshot = TransportTraceSnapshot.from_raw(7, 2, [list(range(1, 16))])

    assert snapshot.sequence == 7
    assert snapshot.dropped == 2
    assert snapshot.records == (TransportTraceRecord(*range(1, 16)),)


def test_transport_trace_snapshot_rejects_incompatible_record_width() -> None:
    """Malformed dispatcher rows fail at the Python ABI boundary."""

    with pytest.raises(RuntimeError, match="expected 15, got 2"):
        TransportTraceSnapshot.from_raw(1, 0, [[1, 2]])


def test_debug_options_encode_orthogonal_features() -> None:
    """Loopback fields and observer feature occupy their assigned ABI bits."""

    options = DebugOptions.create(
        loopback_site=DebugLoopbackSite.ATNAGENT,
        transport_observer=True,
    )

    assert options.raw == (1 << 32) | (1 << 33) | 2
    assert options.enabled(DebugOption.LOOPBACK)
    assert options.enabled(DebugOption.TRANSPORT_OBSERVER)
    assert options.loopback_site is DebugLoopbackSite.ATNAGENT


@pytest.mark.parametrize("raw", [-1, 1 << 34, 1 << 2, 1 << 32, 1])
def test_debug_options_reject_invalid_encodings(raw: int) -> None:
    """Unknown, reserved, and inconsistent fields fail at the Python ABI boundary."""

    with pytest.raises(ValueError, match=r"debug options|loopback option"):
        DebugOptions(raw)
