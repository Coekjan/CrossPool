"""Native Fabric topology, concurrency, and quiesce behavior."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import xtest
from xkit.child import PythonChildProcess
from xpool.fabric import FABRIC_UID_HEX_LENGTH, FabricUid
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind
from xtest.harness.native.fabric import topology
from xtest.harness.native.fabric.bootstrap import fabric_bootstrap
from xtest.harness.native.fabric.protocol import (
    FabricCoordinatorTrace,
    FabricParticipantCommand,
    FabricParticipantSpec,
    FabricParticipantStopped,
)
from xtest.harness.native.fabric.topology import run_fabric_topology
from xtest.harness.support.config import TEST_EXECUTION_CONTRACT
from xtest.harness.support.native.fabric import assert_fabric_report

pytestmark = pytest.mark.timeout(180)


def setup_peer(connection: Connection, spec: FabricParticipantSpec, *, directory: Path, failure: str) -> None:
    try:
        if spec.role is RuntimeRole.FFNAGENT:
            if failure == "source" and spec.pe == spec.atnagent_count:
                raise RuntimeError("source preparation failed")
            connection.send(TEST_EXECUTION_CONTRACT)
        assert connection.recv() is FabricParticipantCommand.STOP
        if failure == "cleanup" and spec.pe == spec.atnagent_count:
            raise RuntimeError("local cleanup failed")
        connection.send(FabricParticipantStopped())
    finally:
        (directory / f"exited-{spec.pe}").write_text(str(os.getpid()))


@pytest.mark.parametrize("failure", ("source", "cleanup"))
def test_setup_failure_reaps_all_children_before_stopping_mps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    children: list[PythonChildProcess] = []
    stopped = []

    def spawn(
        name: str,
        target: Callable[[Connection, FabricParticipantSpec], None],
        spec: FabricParticipantSpec,
        *,
        log_path: Path,
    ) -> PythonChildProcess:
        child = PythonChildProcess(
            name,
            partial(setup_peer, directory=tmp_path, failure=failure),
            spec,
            log_path=log_path,
            import_paths=(Path(__file__).resolve().parents[4],),
        )
        children.append(child)
        return child

    def stop(*, deadline: float) -> None:
        assert len(children) == 3 and all(child.closed for child in children)
        for pe in range(3):
            assert not psutil.pid_exists(int((tmp_path / f"exited-{pe}").read_text()))
        stopped.append(True)

    def fail_plan(**arguments: object) -> None:
        raise RuntimeError("Plan construction failed")

    monkeypatch.setattr(topology, "get_task_root", lambda: None)
    monkeypatch.setattr(
        topology, "visible_uuids", lambda: tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(3))
    )
    monkeypatch.setattr(
        topology, "MpsScope", lambda endpoint: SimpleNamespace(endpoint=endpoint, start=lambda: None, stop=stop)
    )
    monkeypatch.setattr(topology, "PythonChildProcess", spawn)
    monkeypatch.setattr(topology, "execution_fabric_plan", fail_plan)
    try:
        with pytest.raises(
            RuntimeError, match="source preparation failed" if failure == "source" else "Plan construction failed"
        ):
            run_fabric_topology(
                FabricUid("a" * FABRIC_UID_HEX_LENGTH),
                workdir=tmp_path / "topology",
                atnagent_count=1,
                ffnagent_count=2,
                executor_lane_count=1,
                forward_modes=(ForwardMode.DECODE,),
                layer_kind=LayerKind.DENSE,
            )
        assert stopped == [True]
        if failure == "cleanup":
            assert "local cleanup failed" in caplog.text
    finally:
        live = tuple(child for child in children if not child.closed)
        PythonChildProcess.terminate_all(live)
        for child in live:
            child.close()


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


@xtest.requirements(device_count=4)
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

    coordinator = tuple(
        record
        for participant in report.participants
        for record in participant.records
        if isinstance(record, FabricCoordinatorTrace)
    )
    assert coordinator
    assert_exclusive_executor_leases(coordinator)


@xtest.requirements(device_count=4)
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
