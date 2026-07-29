"""Shared semantic assertions for native Fabric topology reports."""

from __future__ import annotations

import pytest

from tests.harness.native.fabric.protocol import (
    FabricAtnAgentTrace,
    FabricCoordinatorTrace,
    FabricExecutionTrace,
    FabricTopologyReport,
)
from xpool.abi import FfnResultCode


def assert_fabric_report(
    report: FabricTopologyReport,
    *,
    expected_repetitions: int | None,
) -> None:
    """Verify numerical results, complete traces, and topology identity."""

    assert not report.activation_rejections
    assert len(report.instances) == report.atnagent_count * len(report.forward_modes)
    assert len(report.participants) == report.atnagent_count + report.ffnagent_count
    for result in report.instances:
        assert result.forward_mode is report.forward_modes[result.instance_index]
        assert result.repetitions_completed > 0
        assert result.result_code is FfnResultCode.OK
        if expected_repetitions is not None:
            assert result.repetitions_completed == expected_repetitions
        assert result.actual == pytest.approx(result.expected, abs=1e-2, rel=1e-2)

    for participant in report.participants:
        assert participant.dropped == 0
        assert participant.sequence == len(participant.records)
        assert participant.records
        assert participant.failure is None
        for record in participant.records:
            match record:
                case FabricAtnAgentTrace():
                    assert record.prepared_ns > 0
                    assert record.acknowledged_ns >= record.prepared_ns
                case FabricCoordinatorTrace():
                    assert record.executor_index is not None
                    assert 0 <= record.executor_index < report.executor_count
                    assert record.enqueued_ns > 0
                    assert record.scheduled_ns >= record.enqueued_ns
                    assert record.released_ns >= record.scheduled_ns
                case FabricExecutionTrace():
                    assert record.observed_ns > 0
                    assert record.started_ns >= record.observed_ns
                    assert record.completed_ns >= record.started_ns
                    assert record.published_ns >= record.completed_ns


def coordinator_records(report: FabricTopologyReport) -> tuple[FabricCoordinatorTrace, ...]:
    """Collect every Coordinator trace from an aggregate topology report."""

    return tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricCoordinatorTrace)
    )
