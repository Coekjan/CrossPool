"""Behavior tests for SGLang E2E observer evidence assertions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tests.harness.support.sglang.observer import (
    assert_atnagent_records,
    assert_dp_attention_paths,
    assert_fabric_observer_snapshots,
    assert_no_fabric_invocations,
    assert_no_transport_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)


def test_transport_observer_requires_acknowledged_success_records(tmp_path: Path) -> None:
    """Accept one complete current Transport mailbox snapshot."""

    write_json(
        tmp_path / "xpool.transport-observer.1.model.0.json",
        {
            "sequence": 2,
            "dropped": 0,
            "record_counts": {"retained": 2, "acknowledged": 2, "closed": 0, "incomplete": 0},
            "records": [
                {"trace_id": 1, "result_code": "ok", "acknowledged": 10, "closed": 0},
                {"trace_id": 2, "result_code": "ok", "acknowledged": 20, "closed": 0},
            ],
        },
    )

    assert_transport_observer_snapshots(tmp_path, expected_count=1)


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
            fabric_record("execution", trace_id=index * 2, invocation_sequence=index, execution_mode=mode),
        )
    ]
    execution_records = [
        fabric_record("execution", trace_id=index, invocation_sequence=index, execution_mode=mode)
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


def test_atnagent_evidence_accepts_idle_publisher_prefill_transition() -> None:
    """Idle PE zero may stage Decode before a peer makes the invocation Prefill."""

    publisher = fabric_record("atnagent", trace_id=1, invocation_sequence=1, execution_mode="prefill")
    publisher_facts = cast(dict[str, object], publisher["facts"])
    publisher_events = cast(dict[str, object], publisher["events_ns"])
    publisher_facts["forward_mode"] = "idle"
    publisher_events["decode_input_staged"] = 110

    follower = fabric_record("atnagent", trace_id=2, invocation_sequence=1, execution_mode="prefill")
    follower_facts = cast(dict[str, object], follower["facts"])
    follower_events = cast(dict[str, object], follower["events_ns"])
    follower_facts["forward_mode"] = "idle"
    follower_events["prefill_input_staged"] = 0
    follower_events["prefill_input_published"] = 0

    assert_atnagent_records([(0, publisher), (1, follower)], atnagent_count=2)


def test_two_model_overlap_requires_distinct_executors_and_intersecting_intervals(tmp_path: Path) -> None:
    """Accept temporal overlap only when two models hold different Executors."""

    left = fabric_record(
        "coordinator",
        trace_id=1,
        invocation_sequence=1,
        execution_mode="decode",
        model_index=0,
        executor_index=0,
        active_interval=(100, 300),
    )
    right = fabric_record(
        "coordinator",
        trace_id=2,
        invocation_sequence=1,
        execution_mode="decode",
        model_index=1,
        executor_index=1,
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
    records[0]["dp_padding_mode"] = "sum_len"
    records[1]["dp_padding_mode"] = "max_len"
    records[1]["result_handoff"] = "reduce_scatter_input"
    records[2]["dp_padding_mode"] = "max_len"
    records[2]["result_handoff"] = "reduce_scatter_input"
    idle_facts = cast(dict[str, object], records[2]["facts"])
    idle_facts["forward_mode"] = "idle"
    idle_facts["result_contribution"] = "zero"
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
    model_index: int = 0,
    executor_index: int = 0,
    active_interval: tuple[int, int] = (110, 230),
) -> dict[str, object]:
    """Build one complete current Fabric observer alternative."""

    common: dict[str, object] = {
        "local_trace_id": trace_id,
        "kind": kind,
        "model_index": model_index,
        "invocation_sequence": invocation_sequence,
        "layer_ordinal": 1,
        "layer_id": 2,
        "result_handoff": "replicated_full",
        "dp_padding_mode": "none",
    }
    if kind == "atnagent":
        prefill = execution_mode == "prefill"
        common["facts"] = {
            "submission_payload_rows": 8,
            "local_token_count": 8,
            "forward_mode": "extend" if prefill else "decode",
            "executor_index": executor_index,
            "execution_mode": execution_mode,
            "result_contribution": "full",
        }
        common["events_ns"] = {
            "submission_prepared": 100,
            "decode_input_staged": 0 if prefill else 110,
            "submission_published": 120,
            "admission_observed": 130,
            "prefill_input_staged": 140 if prefill else 0,
            "prefill_input_published": 150 if prefill else 0,
            "result_observed": 200,
            "output_prepared": 210,
            "transport_evaluated_published": 220,
            "acknowledgement_published": 230,
        }
    elif kind == "coordinator":
        common["facts"] = {
            "invocation_payload_rows": 8,
            "input_pe": 0,
            "execution_mode": execution_mode,
            "executor_index": executor_index,
            "scheduler": {"policy": "fifo", "ready_ticket": invocation_sequence},
        }
        common["events_ns"] = {
            "enqueued": 100,
            "scheduled": active_interval[0],
            "admissions_published": 120,
            "invocations_published": 130,
            "completions_observed": 200,
            "results_published": 210,
            "acknowledgements_observed": 220,
            "scheduler_released": active_interval[1],
        }
    elif kind == "execution":
        prefill = execution_mode == "prefill"
        common["facts"] = {
            "invocation_payload_rows": 8,
            "input_pe": 0,
            "execution_mode": execution_mode,
            "executor_index": executor_index,
        }
        common["events_ns"] = {
            "invocation_observed": 130,
            "decode_input_pull_started": 140 if not prefill else 0,
            "decode_input_pull_completed": 150 if not prefill else 0,
            "prefill_input_ready_observed": 150 if prefill else 0,
            "execution_started": 160,
            "execution_completed": 190,
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
