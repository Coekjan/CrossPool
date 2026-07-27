"""Assertions for structured native observer evidence in SGLang E2E tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast


def assert_transport_observer_snapshots(outdir: Path, *, expected_count: int) -> None:
    """Require complete, successful, gap-free Transport mailbox traces."""

    paths = sorted(outdir.glob("xpool.transport-observer.*.json"))
    assert len(paths) == expected_count
    for path in paths:
        snapshot = read_json_object(path)
        records = object_list(snapshot, "records")
        record_counts = object_dict(snapshot, "record_counts")
        assert cast(int, snapshot["sequence"]) > 0
        assert snapshot["dropped"] == 0
        assert record_counts["incomplete"] == 0
        assert record_counts["closed"] == 0
        assert record_counts["acknowledged"] == len(records)
        assert records
        trace_ids = [record["trace_id"] for record in records]
        assert trace_ids == list(range(cast(int, trace_ids[0]), cast(int, trace_ids[0]) + len(trace_ids)))
        assert trace_ids[-1] == snapshot["sequence"]
        assert all(record["result_code"] == "ok" for record in records)
        assert all(record["acknowledged"] != 0 and record["closed"] == 0 for record in records)


def assert_no_transport_observer_snapshots(outdir: Path) -> None:
    """Require a loopback path to bypass Transport arena publication entirely."""

    assert sorted(outdir.glob("xpool.transport-observer.*.json")) == []


def assert_fabric_observer_snapshots(
    outdir: Path,
    *,
    atnagent_count: int,
    ffnagent_count: int,
    expected_topologies: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Require complete all-AtnAgent Coordinator and Execution trace alternatives."""

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

    records_by_kind: dict[str, list[tuple[int, dict[str, object]]]] = {
        "atnagent": [],
        "coordinator": [],
        "execution": [],
    }
    for snapshot in snapshots:
        records = object_list(snapshot, "records")
        assert cast(int, snapshot["sequence"]) > 0
        assert snapshot["dropped"] == 0
        assert records
        trace_ids = [record["local_trace_id"] for record in records]
        assert trace_ids == list(range(cast(int, trace_ids[0]), cast(int, trace_ids[0]) + len(trace_ids)))
        assert trace_ids[-1] == snapshot["sequence"]
        pe = cast(int, snapshot["pe"])
        for record in records:
            kind = cast(str, record["kind"])
            assert kind in records_by_kind
            records_by_kind[kind].append((pe, record))
            assert record["result_handoff"] in {"replicated_full", "reduce_scatter_input"}
            assert record["dp_padding_mode"] in {"none", "max_len", "sum_len"}

    assert_atnagent_records(records_by_kind["atnagent"], atnagent_count=atnagent_count)
    assert_coordinator_records(records_by_kind["coordinator"], coordinator_pe=atnagent_count)
    assert_execution_records(
        records_by_kind["execution"],
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
    )
    assert_invocation_correspondence(records_by_kind, atnagent_count=atnagent_count, ffnagent_count=ffnagent_count)


def assert_two_model_executor_overlap(outdir: Path) -> None:
    """Require model-zero and model-one Active intervals to overlap on distinct Executors."""

    snapshots = read_fabric_observer_snapshots(outdir)
    coordinator_records = [
        record
        for snapshot in snapshots
        for record in object_list(snapshot, "records")
        if record["kind"] == "coordinator"
    ]
    intervals_by_model: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for record in coordinator_records:
        model_index = cast(int, record["model_index"])
        if model_index not in intervals_by_model:
            continue
        facts = object_dict(record, "facts")
        events = object_dict(record, "events_ns")
        executor_index = cast(int, facts["executor_index"])
        scheduled = cast(int, events["scheduled"])
        released = cast(int, events["scheduler_released"])
        if scheduled > 0 and released >= scheduled:
            intervals_by_model[model_index].append((executor_index, scheduled, released))
    assert intervals_by_model[0]
    assert intervals_by_model[1]
    assert any(
        left_executor != right_executor and left_start < right_end and right_start < left_end
        for left_executor, left_start, left_end in intervals_by_model[0]
        for right_executor, right_start, right_end in intervals_by_model[1]
    )


def assert_dp_attention_paths(outdir: Path) -> None:
    """Require active/Idle ranks and both accepted DP padding/handoff paths."""

    records = [
        record
        for snapshot in read_fabric_observer_snapshots(outdir)
        for record in object_list(snapshot, "records")
        if record["kind"] == "atnagent"
    ]
    assert records
    facts = [object_dict(record, "facts") for record in records]
    assert {"extend", "decode", "idle"} <= {fact["forward_mode"] for fact in facts}
    assert {"full", "zero"} <= {fact["result_contribution"] for fact in facts}
    assert {"sum_len", "max_len"} <= {record["dp_padding_mode"] for record in records}
    assert {"replicated_full", "reduce_scatter_input"} <= {record["result_handoff"] for record in records}


def assert_no_fabric_invocations(outdir: Path) -> None:
    """Require lifecycle-only Fabric snapshots to contain no data-plane records."""

    for snapshot in read_fabric_observer_snapshots(outdir):
        assert snapshot["sequence"] == 0
        assert snapshot["dropped"] == 0
        assert object_list(snapshot, "records") == []


def read_fabric_observer_snapshots(outdir: Path) -> list[dict[str, object]]:
    """Read every per-PE Fabric observer document in deterministic PE order."""

    paths = sorted(outdir.glob("xpool.fabric-observer.*.json"))
    return [read_json_object(path) for path in paths]


def assert_atnagent_records(records: list[tuple[int, dict[str, object]]], *, atnagent_count: int) -> None:
    """Validate AtnAgent mode-sensitive publication and result events."""

    assert records
    for pe, record in records:
        assert pe < atnagent_count
        facts = object_dict(record, "facts")
        events = object_dict(record, "events_ns")
        assert facts["forward_mode"] in {"extend", "decode", "idle"}
        assert facts["execution_mode"] in {"prefill", "decode"}
        assert facts["result_contribution"] in {"full", "zero"}
        require_positive_events(
            events,
            (
                "submission_prepared",
                "submission_published",
                "admission_observed",
                "result_observed",
                "output_prepared",
                "transport_evaluated_published",
                "acknowledgement_published",
            ),
        )
        input_publisher = pe == 0
        if input_publisher and facts["forward_mode"] != "extend":
            require_positive_events(events, ("decode_input_staged",))
        else:
            require_zero_events(events, ("decode_input_staged",))
        if input_publisher and facts["execution_mode"] == "prefill":
            require_positive_events(events, ("prefill_input_staged", "prefill_input_published"))
        else:
            require_zero_events(events, ("prefill_input_staged", "prefill_input_published"))


def assert_coordinator_records(records: list[tuple[int, dict[str, object]]], *, coordinator_pe: int) -> None:
    """Validate complete Coordinator scheduling and publication events."""

    assert records
    for pe, record in records:
        assert pe == coordinator_pe
        facts = object_dict(record, "facts")
        events = object_dict(record, "events_ns")
        assert facts["execution_mode"] in {"prefill", "decode"}
        assert facts["scheduler"] is not None
        require_positive_events(
            events,
            (
                "enqueued",
                "scheduled",
                "admissions_published",
                "invocations_published",
                "completions_observed",
                "results_published",
                "acknowledgements_observed",
                "scheduler_released",
            ),
        )


def assert_execution_records(
    records: list[tuple[int, dict[str, object]]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Validate every physical FfnAgent contribution for each distributed Execution."""

    assert records
    for pe, record in records:
        assert atnagent_count <= pe < atnagent_count + ffnagent_count
        facts = object_dict(record, "facts")
        events = object_dict(record, "events_ns")
        require_positive_events(
            events,
            ("invocation_observed", "execution_started", "execution_completed", "completion_published"),
        )
        if facts["execution_mode"] == "decode":
            require_positive_events(events, ("decode_input_pull_started", "decode_input_pull_completed"))
            require_zero_events(events, ("prefill_input_ready_observed",))
        else:
            assert facts["execution_mode"] == "prefill"
            require_positive_events(events, ("prefill_input_ready_observed",))
            require_zero_events(events, ("decode_input_pull_started", "decode_input_pull_completed"))


def assert_invocation_correspondence(
    records_by_kind: dict[str, list[tuple[int, dict[str, object]]]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Match one Coordinator record to every AtnAgent and FfnAgent alternative."""

    grouped: dict[str, dict[tuple[object, ...], list[tuple[int, dict[str, object]]]]] = {}
    for kind, records in records_by_kind.items():
        by_key: dict[tuple[object, ...], list[tuple[int, dict[str, object]]]] = {}
        for pe, record in records:
            by_key.setdefault(invocation_key(record), []).append((pe, record))
        grouped[kind] = by_key
    assert grouped["atnagent"].keys() == grouped["coordinator"].keys() == grouped["execution"].keys()
    assert all(len(records) == atnagent_count for records in grouped["atnagent"].values())
    assert all(len(records) == 1 for records in grouped["coordinator"].values())
    assert all(len(records) == ffnagent_count for records in grouped["execution"].values())


def invocation_key(record: dict[str, object]) -> tuple[object, ...]:
    """Return the semantic identity shared by all trace alternatives."""

    facts = object_dict(record, "facts")
    return (
        record["model_index"],
        record["invocation_sequence"],
        record["layer_ordinal"],
        facts["executor_index"],
    )


def require_positive_events(events: dict[str, object], names: tuple[str, ...]) -> None:
    """Require every named event timestamp to be present."""

    assert all(cast(int, events[name]) > 0 for name in names)


def require_zero_events(events: dict[str, object], names: tuple[str, ...]) -> None:
    """Require every named mode-inapplicable event timestamp to be absent."""

    assert all(events[name] == 0 for name in names)


def read_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object and reject a non-object root."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def object_dict(container: dict[str, object], key: str) -> dict[str, object]:
    """Return one required object-valued field."""

    value = container[key]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def object_list(container: dict[str, object], key: str) -> list[dict[str, object]]:
    """Return one required list of object-valued fields."""

    value = container[key]
    assert isinstance(value, list) and all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)
