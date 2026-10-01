"""Assertions for structured native observer evidence in integration tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from xpool.devkit.fabric_observer import (
    FabricAtnAgentRecordJson,
    FabricCoordinatorRecordJson,
    FabricFfnAgentRecordJson,
    FabricRecordJson,
    FabricSnapshotJson,
)
from xpool.devkit.transport_observer import TransportSnapshotJson


def assert_transport_observer_snapshots(
    outdir: Path,
    *,
    expected_count: int,
    site: Literal["instance", "atnagent"] | None = None,
) -> None:
    """Require complete, successful, gap-free Transport mailbox traces."""

    pattern = "xpool.transport-observer.*.json" if site is None else f"xpool.transport-observer.*.{site}.*.json"
    paths = sorted(outdir.glob(pattern))
    assert len(paths) == expected_count
    adapter = TypeAdapter(TransportSnapshotJson)
    for path in paths:
        snapshot = adapter.validate_json(path.read_text(encoding="utf-8"), strict=True, extra="allow")
        records = snapshot["records"]
        record_counts = snapshot["record_counts"]
        snapshot_site = snapshot["site"]
        assert site is None or snapshot_site == site
        assert snapshot["sequence"] > 0
        assert snapshot["dropped"] == 0
        assert record_counts["incomplete"] == 0
        assert record_counts["closed"] == 0
        assert record_counts["completed"] == len(records)
        assert records
        trace_ids = [record["trace_id"] for record in records]
        assert trace_ids == list(range(trace_ids[0], trace_ids[0] + len(trace_ids)))
        assert trace_ids[-1] == snapshot["sequence"]
        assert all(record["result_code"] == "ok" for record in records)
        completion_field = "result_acknowledged" if snapshot_site == "instance" else "result_published"
        assert all(record[completion_field] != 0 and record["closed"] == 0 for record in records)


def assert_fabric_observer_snapshots(
    outdir: Path,
    *,
    atnagent_count: int,
    ffnagent_count: int,
    expected_topologies: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Require complete AtnAgent, Coordinator, and FfnAgent trace alternatives."""

    snapshots = read_fabric_observer_snapshots(outdir)
    expected_count = atnagent_count + ffnagent_count
    assert len(snapshots) == expected_count
    assert len({snapshot["generation"] for snapshot in snapshots}) == 1
    assert {snapshot["pe"] for snapshot in snapshots} == set(range(expected_count))
    assert all(snapshot["atnagent_count"] == atnagent_count for snapshot in snapshots)
    assert all(snapshot["ffnagent_count"] == ffnagent_count for snapshot in snapshots)
    if expected_topologies is not None:
        expected = [
            {"atn_tp_size": atn_tp_size, "atn_dp_size": atn_dp_size} for atn_tp_size, atn_dp_size in expected_topologies
        ]
        assert all(snapshot["model_topologies"] == expected for snapshot in snapshots)

    atnagent_records: list[tuple[int, FabricAtnAgentRecordJson]] = []
    coordinator_records: list[tuple[int, FabricCoordinatorRecordJson]] = []
    ffnagent_records: list[tuple[int, FabricFfnAgentRecordJson]] = []
    for snapshot in snapshots:
        records = snapshot["records"]
        assert snapshot["sequence"] > 0
        assert snapshot["dropped"] == 0
        assert records
        trace_ids = [record["local_trace_id"] for record in records]
        assert trace_ids == list(range(trace_ids[0], trace_ids[0] + len(trace_ids)))
        assert trace_ids[-1] == snapshot["sequence"]
        pe = snapshot["pe"]
        for record in records:
            match record["kind"]:
                case "atnagent":
                    atnagent_records.append((pe, record))
                case "coordinator":
                    coordinator_records.append((pe, record))
                case "ffnagent":
                    ffnagent_records.append((pe, record))
            assert record["output_requirement"] in {"per_rank_complete", "group_sum_complete"}
            assert record["dp_row_layout"] in {None, "none", "uniform_by_rank", "packed_by_rank"}

    assert_atnagent_records(atnagent_records, atnagent_count=atnagent_count)
    assert_coordinator_records(coordinator_records, coordinator_pe=atnagent_count)
    assert_ffnagent_records(
        ffnagent_records,
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
    )
    assert_invocation_correspondence(
        {"atnagent": atnagent_records, "coordinator": coordinator_records, "ffnagent": ffnagent_records},
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
    )


def assert_two_model_executor_overlap(outdir: Path) -> None:
    """Require model-zero and model-one Active intervals to overlap on distinct Executors."""

    snapshots = read_fabric_observer_snapshots(outdir)
    coordinator_records = [
        record for snapshot in snapshots for record in snapshot["records"] if record["kind"] == "coordinator"
    ]
    intervals_by_model: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for record in coordinator_records:
        instance_index = record["instance_index"]
        if instance_index not in intervals_by_model:
            continue
        facts = record["facts"]
        events = record["events_ns"]
        executor_lane_index = facts["executor_lane_index"]
        scheduled = events["scheduled"]
        released = events["lane_released"]
        if scheduled > 0 and released >= scheduled:
            assert executor_lane_index is not None
            intervals_by_model[instance_index].append((executor_lane_index, scheduled, released))

    overlap_durations = [
        max(0, min(left_end, right_end) - max(left_start, right_start))
        for left_executor, left_start, left_end in intervals_by_model[0]
        for right_executor, right_start, right_end in intervals_by_model[1]
        if left_executor != right_executor
    ]
    maximum_overlap_ns = max(overlap_durations, default=0)
    executors_by_model = {
        instance_index: sorted({executor for executor, _, _ in intervals})
        for instance_index, intervals in intervals_by_model.items()
    }
    assert maximum_overlap_ns > 0, (
        "two-model E2E did not observe overlapping Active intervals on distinct Executors: "
        f"interval_counts={{0: {len(intervals_by_model[0])}, 1: {len(intervals_by_model[1])}}}, "
        f"executors={executors_by_model}, candidate_pairs={len(overlap_durations)}, "
        f"maximum_overlap_ns={maximum_overlap_ns}"
    )


def assert_dp_attention_paths(outdir: Path) -> None:
    """Require active/Idle ranks and both accepted DP layout/output paths."""

    records = [
        record
        for snapshot in read_fabric_observer_snapshots(outdir)
        for record in snapshot["records"]
        if record["kind"] == "atnagent"
    ]
    assert records
    facts = [record["facts"] for record in records]
    assert {"prefill", "decode", "idle"} <= {fact["forward_mode"] for fact in facts}
    assert {"packed_by_rank", "uniform_by_rank"} <= {record["dp_row_layout"] for record in records}
    assert {"per_rank_complete", "group_sum_complete"} <= {record["output_requirement"] for record in records}


def read_fabric_observer_snapshots(outdir: Path) -> list[FabricSnapshotJson]:
    """Read every per-PE Fabric observer document in deterministic PE order."""

    paths = sorted(outdir.glob("xpool.fabric-observer.*.json"))
    adapter = TypeAdapter(FabricSnapshotJson)
    return [adapter.validate_json(path.read_text(encoding="utf-8"), strict=True, extra="allow") for path in paths]


def assert_atnagent_records(records: list[tuple[int, FabricAtnAgentRecordJson]], *, atnagent_count: int) -> None:
    """Validate AtnAgent mode-sensitive publication and result events."""

    assert records
    for pe, record in records:
        assert pe < atnagent_count
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["forward_mode"] in {"prefill", "decode", "idle"}
        assert record["payload_rows"] > 0
        assert facts["dp_rank_payload_rows"] >= 0
        lease_sequence = facts["executor_lease_sequence"]
        assert lease_sequence is not None and lease_sequence > 0
        require_positive_events(
            events,
            (
                "submission_prepared",
                "submission_published",
                "admission_observed",
                "output_commit_observed",
                "output_acknowledgement_published",
            ),
        )


def assert_coordinator_records(records: list[tuple[int, FabricCoordinatorRecordJson]], *, coordinator_pe: int) -> None:
    """Validate complete Coordinator scheduling and publication events."""

    assert records
    for pe, record in records:
        assert pe == coordinator_pe
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["scheduler"] is not None
        lease_sequence = facts["executor_lease_sequence"]
        assert lease_sequence is not None and lease_sequence > 0
        require_positive_events(
            events,
            (
                "enqueued",
                "scheduled",
                "admission_published",
                "lane_execution_published",
                "ffnagent_completions_observed",
                "output_commit_published",
                "output_acknowledgements_observed",
                "lane_released",
            ),
        )


def assert_ffnagent_records(
    records: list[tuple[int, FabricFfnAgentRecordJson]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Validate every selected physical FfnAgent contribution."""

    assert records
    for pe, record in records:
        assert atnagent_count <= pe < atnagent_count + ffnagent_count
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["executor_lease_sequence"] > 0
        assert facts["payload_row_capacity"] >= record["payload_rows"]
        assert facts["delivery"] in {"direct_partial", "single_complete", "replicated_complete"}
        require_positive_events(
            events,
            (
                "lane_execution_observed",
                "input_ready_observed",
                "compute_started",
                "compute_completed",
                "completion_published",
            ),
        )


def assert_invocation_correspondence(
    records_by_kind: Mapping[str, Sequence[tuple[int, FabricRecordJson]]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Match each invocation across AtnAgent, Coordinator, and selected FfnAgent records."""

    grouped: dict[str, dict[tuple[int, int, int, int | None], list[tuple[int, FabricRecordJson]]]] = {}
    for kind, records in records_by_kind.items():
        by_key: dict[tuple[int, int, int, int | None], list[tuple[int, FabricRecordJson]]] = {}
        for pe, record in records:
            by_key.setdefault(invocation_key(record), []).append((pe, record))
        grouped[kind] = by_key
    assert grouped["atnagent"].keys() == grouped["coordinator"].keys() == grouped["ffnagent"].keys()
    assert all(len(records) == atnagent_count for records in grouped["atnagent"].values())
    assert all(
        any(record["events_ns"]["input_ready_published"] > 0 for _, record in records)
        for records in grouped["atnagent"].values()
    )
    assert all(len(records) == 1 for records in grouped["coordinator"].values())
    assert all(1 <= len(records) <= ffnagent_count for records in grouped["ffnagent"].values())


def invocation_key(record: FabricRecordJson) -> tuple[int, int, int, int | None]:
    """Return the semantic identity shared by all trace alternatives."""

    facts = record["facts"]
    return (
        record["instance_index"],
        record["invocation_sequence"],
        record["layer_ordinal"],
        facts["executor_lane_index"],
    )


def require_positive_events(events: dict[str, int], names: tuple[str, ...]) -> None:
    """Require every named event timestamp to be present."""

    assert all(events[name] > 0 for name in names)
