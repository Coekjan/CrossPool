from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tests.harness.support.native.observer import (
    assert_atnagent_records,
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_no_fabric_invocations,
    assert_no_transport_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)


def test_transport_observer_requires_completed_success_records(tmp_path: Path) -> None:
    """Accept one complete current Transport mailbox snapshot."""

    write_json(
        tmp_path / "xpool.transport-observer.1.instance.model.0.json",
        {
            "site": "instance",
            "sequence": 2,
            "dropped": 0,
            "record_counts": {"retained": 2, "completed": 2, "closed": 0, "incomplete": 0},
            "records": [
                {"trace_id": 1, "result_code": "ok", "result_acknowledged": 10, "closed": 0},
                {"trace_id": 2, "result_code": "ok", "result_acknowledged": 20, "closed": 0},
            ],
        },
    )

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
    publisher_facts = cast(dict[str, object], publisher["facts"])
    publisher_events = cast(dict[str, object], publisher["events_ns"])
    publisher_facts["forward_mode"] = "idle"
    publisher_events["input_ready_published"] = 110

    follower = fabric_record("atnagent", trace_id=2, invocation_sequence=1, execution_mode="prefill")
    follower_facts = cast(dict[str, object], follower["facts"])
    follower_events = cast(dict[str, object], follower["events_ns"])
    follower_facts["forward_mode"] = "idle"
    follower_events["input_ready_published"] = 0

    assert_atnagent_records([(0, publisher), (1, follower)], atnagent_count=2)


def test_two_model_overlap_requires_distinct_executors_and_intersecting_intervals(tmp_path: Path) -> None:
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


def test_dp_attention_evidence_requires_both_padding_handoff_and_rank_paths(tmp_path: Path) -> None:
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
    idle_facts = cast(dict[str, object], records[2]["facts"])
    idle_facts["forward_mode"] = "idle"
    write_fabric_snapshot(tmp_path, pe=0, records=records)

    assert_dp_attention_paths(tmp_path)


def test_no_invocation_evidence_accepts_absent_transport_and_empty_fabric(tmp_path: Path) -> None:
    write_fabric_snapshot(tmp_path, pe=0, records=[])

    assert_no_transport_observer_snapshots(tmp_path)
    assert_no_fabric_invocations(tmp_path)


def fabric_record(
    kind: str,
    *,
    trace_id: int,
    invocation_sequence: int,
    execution_mode: str,
    instance_index: int = 0,
    executor_lane_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> dict[str, object]:
    """Build one complete current Fabric observer alternative."""

    common: dict[str, object] = {
        "local_trace_id": trace_id,
        "kind": kind,
        "instance_index": instance_index,
        "invocation_sequence": invocation_sequence,
        "layer_ordinal": 1,
        "payload_rows": 8,
        "output_requirement": "per_rank_complete",
        "dp_row_layout": "none" if kind == "atnagent" else None,
    }
    if kind == "atnagent":
        prefill = execution_mode == "prefill"
        common["facts"] = {
            "submission_payload_rows": 8,
            "dp_rank_payload_rows": 8,
            "forward_mode": "prefill" if prefill else "decode",
            "executor_lane_index": executor_lane_index,
            "executor_lease_sequence": invocation_sequence,
        }
        common["events_ns"] = {
            "submission_prepared": 100,
            "submission_published": 120,
            "admission_observed": 130,
            "input_ready_published": 150,
            "output_commit_observed": 200,
            "output_acknowledgement_published": 230,
        }
    elif kind == "coordinator":
        common["facts"] = {
            "executor_lane_index": executor_lane_index,
            "executor_lease_sequence": invocation_sequence,
            "scheduler": {"policy": "fifo", "ready_ticket": invocation_sequence},
        }
        common["events_ns"] = {
            "enqueued": 100,
            "scheduled": active_interval[0],
            "admission_published": 120,
            "lane_execution_published": 130,
            "ffnagent_completions_observed": 200,
            "output_commit_published": 210,
            "output_acknowledgements_observed": 220,
            "lane_released": active_interval[1],
        }
    elif kind == "ffnagent":
        common["facts"] = {
            "executor_lane_index": executor_lane_index,
            "executor_lease_sequence": invocation_sequence,
            "payload_row_capacity": 8,
            "delivery": "replicated_complete",
        }
        common["events_ns"] = {
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
        }
    else:
        raise ValueError(f"unexpected Fabric trace kind {kind}")
    common["durations_ns"] = {}
    return common


def write_fabric_snapshot(tmp_path: Path, *, pe: int, records: list[dict[str, object]]) -> None:
    """Write one current per-PE Fabric observer snapshot."""

    write_json(
        tmp_path / f"xpool.fabric-observer.generation.{pe}.json",
        {
            "generation": "generation",
            "pe": pe,
            "atnagent_count": 1,
            "ffnagent_count": 2,
            "model_topologies": [{"atn_tp_size": 1, "atn_dp_size": 1}],
            "sequence": len(records),
            "dropped": 0,
            "records": records,
        },
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic JSON test fixture."""

    path.write_text(json.dumps(payload), encoding="utf-8")
