from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, overload

import pytest
from pydantic import ValidationError

from xpool.devkit.fabric_observer import (
    FabricAtnAgentRecordJson,
    FabricCoordinatorRecordJson,
    FabricFfnAgentRecordJson,
    FabricRecordFieldsJson,
    FabricRecordJson,
    FabricSnapshotJson,
)
from xpool.devkit.transport_observer import TransportRecordJson, TransportSnapshotJson
from xtest.harness.support.config import TEST_MODEL_ID
from xtest.harness.support.native.observer import (
    assert_atnagent_records,
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
    read_fabric_observer_snapshots,
)


@pytest.mark.parametrize("pid", [1, "invalid"])
def test_transport_observer_reads_declared_mailbox_records(tmp_path: Path, pid: int | str) -> None:
    """Validate external fields before checking complete mailbox traces."""

    payload: TransportSnapshotJson = {
        "pid": 1,
        "site": "instance",
        "model_id": str(TEST_MODEL_ID),
        "rank": 0,
        "arena_handle_suffix": "00" * 8,
        "sequence": 2,
        "dropped": 0,
        "phase_summary": {},
        "record_counts": {"retained": 2, "completed": 2, "closed": 0, "incomplete": 0},
        "records": [transport_record(1), transport_record(2)],
    }
    write_json(tmp_path / "xpool.transport-observer.1.instance.model.0.json", {**payload, "pid": pid})

    if isinstance(pid, str):
        with pytest.raises(ValidationError, match="pid"):
            assert_transport_observer_snapshots(tmp_path, expected_count=1, site="instance")
    else:
        assert_transport_observer_snapshots(tmp_path, expected_count=1, site="instance")


def test_fabric_observer_matches_all_trace_alternatives_and_modes(tmp_path: Path) -> None:
    """Accept matching AtnAgent, Coordinator, and per-FfnAgent Execution records."""

    modes = ("decode", "prefill")
    atnagent_records = [
        fabric_record("atnagent", trace_id=index, invocation_sequence=index, execution_mode=mode)
        for index, mode in enumerate(modes, start=1)
    ]
    coordinator_records = [
        record
        for index, mode in enumerate(modes, start=1)
        for record in (
            fabric_record("coordinator", trace_id=index * 2 - 1, invocation_sequence=index, execution_mode=mode),
            fabric_record("ffnagent", trace_id=index * 2, invocation_sequence=index, execution_mode=mode),
        )
    ]
    execution_records = [
        fabric_record("ffnagent", trace_id=index, invocation_sequence=index, execution_mode=mode)
        for index, mode in enumerate(modes, start=1)
    ]
    for pe, records in enumerate((atnagent_records, coordinator_records, execution_records)):
        write_fabric_snapshot(tmp_path, pe=pe, records=records)

    assert_fabric_observer_snapshots(
        tmp_path,
        atnagent_count=1,
        ffnagent_count=2,
        expected_topologies=((1, 1),),
    )


def test_atnagent_evidence_accepts_a_non_input_source() -> None:
    """Only an actual Input Source records InputReadyPublished."""

    publisher = fabric_record("atnagent", trace_id=1, invocation_sequence=1, execution_mode="prefill")
    publisher_facts = publisher["facts"]
    publisher_events = publisher["events_ns"]
    publisher_facts["forward_mode"] = "idle"
    publisher_events["input_ready_published"] = 110

    follower = fabric_record("atnagent", trace_id=2, invocation_sequence=1, execution_mode="prefill")
    follower_facts = follower["facts"]
    follower_events = follower["events_ns"]
    follower_facts["forward_mode"] = "idle"
    follower_events["input_ready_published"] = 0

    assert_atnagent_records([(0, publisher), (1, follower)], atnagent_count=2)


def test_two_model_overlap_accepts_distinct_executors_and_intersecting_intervals(tmp_path: Path) -> None:
    """Accept temporal overlap only when two models hold different Executors."""

    left = fabric_record(
        "coordinator",
        trace_id=1,
        invocation_sequence=1,
        execution_mode="decode",
        instance_index=0,
        executor_lane_index=0,
        active_interval=(100, 300),
    )
    right = fabric_record(
        "coordinator",
        trace_id=2,
        invocation_sequence=1,
        execution_mode="decode",
        instance_index=1,
        executor_lane_index=1,
        active_interval=(200, 400),
    )
    write_fabric_snapshot(tmp_path, pe=1, records=[left, right])

    assert_two_model_executor_overlap(tmp_path)


def test_dp_attention_evidence_accepts_padding_handoff_and_rank_paths(tmp_path: Path) -> None:
    records = [
        fabric_record("atnagent", trace_id=1, invocation_sequence=1, execution_mode="prefill"),
        fabric_record("atnagent", trace_id=2, invocation_sequence=2, execution_mode="decode"),
        fabric_record("atnagent", trace_id=3, invocation_sequence=2, execution_mode="decode"),
    ]
    records[0]["dp_row_layout"] = "packed_by_rank"
    records[1]["dp_row_layout"] = "uniform_by_rank"
    records[1]["output_requirement"] = "group_sum_complete"
    records[2]["dp_row_layout"] = "uniform_by_rank"
    records[2]["output_requirement"] = "group_sum_complete"
    idle_facts = records[2]["facts"]
    idle_facts["forward_mode"] = "idle"
    write_fabric_snapshot(tmp_path, pe=0, records=records)

    assert_dp_attention_paths(tmp_path)


def test_fabric_reader_retains_extensions_and_incomplete_facts(tmp_path: Path) -> None:
    """Incomplete diagnostics stay readable, including producer extensions."""

    record = fabric_record("atnagent", trace_id=1, invocation_sequence=1, execution_mode="decode")
    record["facts"]["executor_lane_index"] = None
    record["facts"]["executor_lease_sequence"] = None
    record["events_ns"].update(
        admission_observed=0,
        input_ready_published=0,
        output_commit_observed=0,
        output_acknowledgement_published=0,
    )
    write_fabric_snapshot(tmp_path, pe=0, records=[record])
    path = tmp_path / "xpool.fabric-observer.generation.0.json"
    # Exercise unknown fields at the serialized external boundary.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extension"] = {"enabled": True}
    payload["records"][0]["extension"] = 7
    payload["records"][0]["facts"]["extension"] = "retained"
    write_json(path, payload)

    assert read_fabric_observer_snapshots(tmp_path) == [payload]


@pytest.mark.parametrize("missing", [False, True])
def test_fabric_reader_requires_nullable_facts_with_their_declared_type(tmp_path: Path, missing: bool) -> None:
    """A nullable field still requires a key and rejects a wrong scalar type."""

    record = fabric_record("coordinator", trace_id=1, invocation_sequence=1, execution_mode="decode")
    write_fabric_snapshot(tmp_path, pe=1, records=[record])
    path = tmp_path / "xpool.fabric-observer.generation.1.json"
    # Deliberately corrupt an external field instead of an internal typed value.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        del payload["records"][0]["facts"]["executor_lane_index"]
    else:
        payload["records"][0]["facts"]["executor_lane_index"] = "invalid"
    write_json(path, payload)

    with pytest.raises(ValidationError, match="executor_lane_index"):
        read_fabric_observer_snapshots(tmp_path)


@overload
def fabric_record(
    kind: Literal["atnagent"],
    *,
    trace_id: int,
    invocation_sequence: int,
    execution_mode: str,
    instance_index: int = 0,
    executor_lane_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> FabricAtnAgentRecordJson: ...


@overload
def fabric_record(
    kind: Literal["coordinator"],
    *,
    trace_id: int,
    invocation_sequence: int,
    execution_mode: str,
    instance_index: int = 0,
    executor_lane_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> FabricCoordinatorRecordJson: ...


@overload
def fabric_record(
    kind: Literal["ffnagent"],
    *,
    trace_id: int,
    invocation_sequence: int,
    execution_mode: str,
    instance_index: int = 0,
    executor_lane_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> FabricFfnAgentRecordJson: ...


def fabric_record(
    kind: str,
    *,
    trace_id: int,
    invocation_sequence: int,
    execution_mode: str,
    instance_index: int = 0,
    executor_lane_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> FabricRecordJson:
    """Build one complete current Fabric observer alternative."""

    common: FabricRecordFieldsJson = {
        "local_trace_id": trace_id,
        "instance_index": instance_index,
        "invocation_sequence": invocation_sequence,
        "layer_ordinal": 1,
        "payload_rows": 8,
        "output_requirement": "per_rank_complete",
    }
    if kind == "atnagent":
        prefill = execution_mode == "prefill"
        return {
            **common,
            "kind": "atnagent",
            "dp_row_layout": "none",
            "facts": {
                "dp_rank_payload_rows": 8,
                "forward_mode": "prefill" if prefill else "decode",
                "executor_lane_index": executor_lane_index,
                "executor_lease_sequence": invocation_sequence,
            },
            "events_ns": {
                "submission_prepared": 100,
                "submission_published": 120,
                "admission_observed": 130,
                "input_ready_published": 150,
                "output_commit_observed": 200,
                "output_acknowledgement_published": 230,
            },
            "durations_ns": {},
        }
    elif kind == "coordinator":
        return {
            **common,
            "kind": "coordinator",
            "dp_row_layout": None,
            "facts": {
                "executor_lane_index": executor_lane_index,
                "executor_lease_sequence": invocation_sequence,
                "scheduler": {"policy": "fifo", "ready_ticket": invocation_sequence},
            },
            "events_ns": {
                "enqueued": 100,
                "scheduled": active_interval[0],
                "admission_published": 120,
                "lane_execution_published": 130,
                "ffnagent_completions_observed": 200,
                "output_commit_published": 210,
                "output_acknowledgements_observed": 220,
                "lane_released": active_interval[1],
            },
            "durations_ns": {},
        }
    elif kind == "ffnagent":
        return {
            **common,
            "kind": "ffnagent",
            "dp_row_layout": None,
            "facts": {
                "executor_lane_index": executor_lane_index,
                "executor_lease_sequence": invocation_sequence,
                "payload_row_capacity": 8,
                "delivery": "replicated_complete",
            },
            "events_ns": {
                "lane_execution_observed": 130,
                "input_ready_observed": 150,
                "routing_metadata_published": 0,
                "routing_metadata_observed": 0,
                "compute_started": 160,
                "compute_completed": 190,
                "partial_ready_published": 195,
                "peer_partials_ready_observed": 196,
                "peer_partials_validated": 0,
                "completion_published": 200,
            },
            "durations_ns": {},
        }
    else:
        raise ValueError(f"unexpected Fabric trace kind {kind}")


def write_fabric_snapshot(tmp_path: Path, *, pe: int, records: Sequence[FabricRecordJson]) -> None:
    """Write one current per-PE Fabric observer snapshot."""

    snapshot: FabricSnapshotJson = {
        "generation": "generation",
        "pe": pe,
        "atnagent_count": 1,
        "ffnagent_count": 2,
        "model_topologies": [{"atn_tp_size": 1, "atn_dp_size": 1}],
        "sequence": len(records),
        "dropped": 0,
        "records": list(records),
    }
    write_json(tmp_path / f"xpool.fabric-observer.generation.{pe}.json", snapshot)


def transport_record(trace_id: int) -> TransportRecordJson:
    """Build one complete acknowledged mailbox trace."""

    return {
        "trace_id": trace_id,
        "payload_rows": 1,
        "layer_ordinal": 0,
        "forward_mode": "decode",
        "output_requirement": "per_rank_complete",
        "dp_row_layout": "none",
        "result_code": "ok",
        "request_staging_started": 1,
        "request_staging_completed": 2,
        "request_published": 3,
        "request_observed": 4,
        "execution_started": 5,
        "execution_completed": 6,
        "result_published": 7,
        "result_observed": 8,
        "output_copied": 9,
        "result_acknowledged": 10,
        "closed": 0,
        "durations_ns": {},
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one deterministic JSON test fixture."""

    path.write_text(json.dumps(payload), encoding="utf-8")
