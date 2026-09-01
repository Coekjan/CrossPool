"""Native Fabric topology, concurrency, and quiesce behavior."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tests.harness.native.fabric.bootstrap import fabric_bootstrap
from tests.harness.native.fabric.protocol import FabricCoordinatorTrace
from tests.harness.native.fabric.topology import run_fabric_topology
from tests.harness.support.native.fabric import assert_fabric_report, coordinator_records
from xpool.native.ffn import ForwardMode, LayerKind

pytestmark = [
    pytest.mark.requires_cuda(min_devices=4),
    pytest.mark.requires_mps,
    pytest.mark.timeout(180),
]


def assert_exclusive_executor_leases(records: tuple[FabricCoordinatorTrace, ...]) -> None:
    """Verify that one Executor never owns overlapping Invocation leases."""

    executor_records: dict[int, list[FabricCoordinatorTrace]] = {}
    for record in records:
        assert record.executor_lane_index is not None
        executor_records.setdefault(record.executor_lane_index, []).append(record)
    for leases in executor_records.values():
        leases.sort(key=lambda record: record.scheduled_ns)
        for previous, current in pairwise(leases):
            assert previous.lane_released_ns <= current.scheduled_ns


def test_fabric_topology_executes_mixed_requests_with_exclusive_two_lane_leases(tmp_path: Path) -> None:
    """Exercise concurrent mixed requests in the largest component topology."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=2,
            ffnagent_count=2,
            executor_lane_count=2,
            forward_modes=(ForwardMode.DECODE, ForwardMode.PREFILL),
            layer_kind=LayerKind.DENSE,
        )
    assert_fabric_report(report, expected_repetitions=3)

    coordinator = coordinator_records(report)
    assert coordinator
    assert_exclusive_executor_leases(coordinator)


def test_fabric_topology_quiesces_mixed_two_lane_requests_before_native_drain(tmp_path: Path) -> None:
    """Finish admitted mixed requests before native Fabric drain."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=2,
            ffnagent_count=2,
            executor_lane_count=2,
            forward_modes=(ForwardMode.DECODE, ForwardMode.PREFILL),
            layer_kind=LayerKind.DENSE,
            quiesce_after_first_request=True,
        )
    assert_fabric_report(report, expected_repetitions=1)
