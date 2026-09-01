"""Shared semantic assertions for native Fabric topology reports."""

from __future__ import annotations

import pytest

from tests.harness.native.fabric.protocol import (
    FabricAtnAgentTrace,
    FabricCoordinatorTrace,
    FabricFfnAgentTrace,
    FabricTopologyReport,
)
from xpool.native import RuntimeRole
from xpool.native.ffn import ResultCode


def assert_fabric_report(
    report: FabricTopologyReport,
    *,
    expected_repetitions: int | None,
) -> None:
    """Verify numerical results, complete traces, and topology identity."""

    assert len(report.instances) == report.atnagent_count * len(report.forward_modes)
    assert len(report.participants) == report.atnagent_count + report.ffnagent_count
    for result in report.instances:
        assert result.forward_mode is report.forward_modes[result.instance_index]
        assert result.repetitions_completed > 0
        assert result.result_code is ResultCode.OK
        if expected_repetitions is not None:
            assert result.repetitions_completed == expected_repetitions
        if result.expected is not None:
            assert result.actual == pytest.approx(result.expected, abs=1e-2, rel=1e-2)

    for participant in report.participants:
        assert participant.dropped == 0
        assert participant.sequence == len(participant.records)
        assert participant.failure is None
        if participant.routing is not None:
            assert participant.routing.dropped == 0
            assert participant.routing.sequence == len(participant.routing.records)
        if not participant.records:
            snapshot = participant.graph_snapshot
            assert participant.role is RuntimeRole.FFNAGENT
            assert snapshot is not None
            assert snapshot.primary_graph_binding_site_counts == ()
            continue
        for record in participant.records:
            match record:
                case FabricAtnAgentTrace():
                    assert record.submission_prepared_ns > 0
                    assert record.output_acknowledgement_published_ns >= record.submission_prepared_ns
                case FabricCoordinatorTrace():
                    assert record.executor_lane_index is not None
                    assert 0 <= record.executor_lane_index < report.executor_lane_count
                    assert record.executor_lease_sequence is not None
                    assert record.enqueued_ns > 0
                    assert record.scheduled_ns >= record.enqueued_ns
                    assert record.lane_released_ns >= record.scheduled_ns
                case FabricFfnAgentTrace():
                    assert record.executor_lease_sequence is not None
                    assert record.payload_row_capacity is not None
                    assert record.delivery is not None
                    assert record.lane_execution_observed_ns > 0
                    assert record.compute_started_ns >= record.lane_execution_observed_ns
                    assert record.compute_completed_ns >= record.compute_started_ns
                    assert record.completion_published_ns >= record.compute_completed_ns


def coordinator_records(report: FabricTopologyReport) -> tuple[FabricCoordinatorTrace, ...]:
    """Collect every Coordinator trace from an aggregate topology report."""

    return tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricCoordinatorTrace)
    )
