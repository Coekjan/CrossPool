"""Native Fabric topology, concurrency, and quiesce behavior."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tests.harness.native.fabric.bootstrap import fabric_bootstrap
from tests.harness.native.fabric.protocol import FabricCoordinatorTrace
from tests.harness.native.fabric.topology import run_fabric_topology
from tests.harness.support.native.fabric import assert_fabric_report, coordinator_records
from xpool.abi import TensorDType, XPoolForwardMode

pytestmark = [pytest.mark.requires_mps, pytest.mark.timeout(180)]


def assert_exclusive_executor_leases(records: tuple[FabricCoordinatorTrace, ...]) -> None:
    """Verify that one Executor never owns overlapping Invocation leases."""

    executor_records: dict[int, list[FabricCoordinatorTrace]] = {}
    for record in records:
        assert record.executor_index is not None
        executor_records.setdefault(record.executor_index, []).append(record)
    for leases in executor_records.values():
        leases.sort(key=lambda record: record.scheduled_ns)
        for previous, current in pairwise(leases):
            assert previous.released_ns <= current.scheduled_ns


topology_cases = (
    pytest.param(1, 1, marks=pytest.mark.requires_cuda(min_devices=2), id="1-atn-1-ffn"),
    pytest.param(1, 2, marks=pytest.mark.requires_cuda(min_devices=3), id="1-atn-2-ffn"),
    pytest.param(2, 1, marks=pytest.mark.requires_cuda(min_devices=3), id="2-atn-1-ffn"),
    pytest.param(2, 2, marks=pytest.mark.requires_cuda(min_devices=4), id="2-atn-2-ffn"),
)

mode_cases = (
    pytest.param((XPoolForwardMode.DECODE, XPoolForwardMode.DECODE), id="decode"),
    pytest.param((XPoolForwardMode.EXTEND, XPoolForwardMode.EXTEND), id="prefill"),
    pytest.param((XPoolForwardMode.DECODE, XPoolForwardMode.EXTEND), id="mixed"),
)


@pytest.mark.parametrize(("atnagent_count", "ffnagent_count"), topology_cases)
@pytest.mark.parametrize("executor_count", [1, 2])
@pytest.mark.parametrize("forward_modes", mode_cases)
def test_fabric_topology_executes_requests_with_exclusive_executor_leases(
    atnagent_count: int,
    ffnagent_count: int,
    executor_count: int,
    forward_modes: tuple[XPoolForwardMode, ...],
    tmp_path: Path,
) -> None:
    """Exercise every accepted topology with traced concurrent requests."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_count=executor_count,
            forward_modes=forward_modes,
            dtype=TensorDType.FP32,
        )
    assert_fabric_report(report, expected_repetitions=3)

    coordinator = coordinator_records(report)
    assert coordinator
    assert_exclusive_executor_leases(coordinator)


@pytest.mark.parametrize(("atnagent_count", "ffnagent_count"), topology_cases)
@pytest.mark.parametrize("executor_count", [1, 2])
def test_fabric_topology_quiesces_before_native_drain(
    atnagent_count: int,
    ffnagent_count: int,
    executor_count: int,
    tmp_path: Path,
) -> None:
    """Stop production and finish every invocation before native Fabric drain."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=atnagent_count,
            ffnagent_count=ffnagent_count,
            executor_count=executor_count,
            forward_modes=(XPoolForwardMode.DECODE, XPoolForwardMode.EXTEND),
            dtype=TensorDType.FP32,
            quiesce_after_first_request=True,
        )
    assert_fabric_report(report, expected_repetitions=1)
