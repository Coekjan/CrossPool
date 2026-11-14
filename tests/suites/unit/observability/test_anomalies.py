"""Anomalous Transport facts survive loss without inventing protocol progress."""

from dataclasses import replace

import pytest
from tests.support.observability import events_for, fixture, fixture_root, scenario

from xpool.observability import Event, ObservationError, QualityCheckpoint, parse_chunk, reconstruct
from xpool.observability.wire import EventKind, EventRole, PresenceBit, ResultCode


def opening() -> Event:
    vector = next(v for v in fixture("encoding")["vectors"] if v["id"] == "anomaly_forward_mode")
    return parse_chunk(bytes.fromhex(vector["hex"]), (fixture_root / vector["metadata"]).read_bytes()).events[0]


def changed(event: Event, **fields: int) -> Event:
    record = event.record.model_copy(update=fields)
    source = replace(event.source, identity=replace(event.source.identity, event_sequence=record.event_sequence))
    return replace(event, record=record, source=source)


def terminal(event: Event, *, result: ResultCode = ResultCode.PROTOCOL_MISMATCH, time: int = 200) -> Event:
    return changed(
        event,
        event_sequence=event.record.event_sequence + 1,
        data0=EventKind.RESULT_PUBLISHED,
        raw_time_ns=time,
        presence=event.record.presence | (1 << PresenceBit.RESULT_CODE),
        result_code=result,
    )


@pytest.mark.parametrize("retained", ["opening", "terminal", "both"])
def test_loss_preserves_raw_request_and_only_retained_boundaries(retained: str) -> None:
    start = opening()
    end = terminal(start)
    events = {"opening": (start,), "terminal": (end,), "both": (start, end)}[retained]
    result = reconstruct(events)
    (operation,) = result.of_kind("operation")
    (observation,) = operation.observations.values()
    assert observation.facts["raw_forward_mode"] == 99
    assert observation.facts["invalid_fields"] == 1
    assert observation.facts["layer_ordinal"] == start.record.data3
    assert observation.facts["payload_rows"] == start.record.data4
    assert "forward_mode" not in observation.facts
    assert set(observation.events) == {event.record.data0 for event in events}
    if retained == "opening":
        assert "result_code" not in observation.facts
        assert "unknown_result" in result.quality.codes
    else:
        assert observation.facts["result_code"] == ResultCode.PROTOCOL_MISMATCH
    processing = next(interval for interval in result.intervals if interval.name == "request_processing")
    assert processing.status == ("complete" if retained == "both" else "partial")
    assert (processing.duration_ns is not None) == (retained == "both")
    assert "invalid_request" in result.quality.codes
    assert "legacy_fact_missing" not in result.quality.codes
    assert result.quality.anomalous_operation_count_lower_bound == 1
    assert result.quality.anomalous_operation_count_exact is False


def test_shutdown_and_recorded_zero_do_not_become_rejection_or_missing_time() -> None:
    end = terminal(opening(), result=ResultCode.SHUTDOWN, time=0)
    result = reconstruct((end,))
    (observation,) = result.of_kind("operation")[0].observations.values()
    assert observation.facts["result_code"] == ResultCode.SHUTDOWN
    assert observation.events[EventKind.RESULT_PUBLISHED].record.raw_time_ns == 0
    assert "unknown_result" not in result.quality.codes
    assert all(interval.duration_ns is None for interval in result.intervals)


def test_distinct_anomaly_boundaries_are_not_semantic_duplicates() -> None:
    start = opening()
    result = reconstruct((start, terminal(start)))
    assert "duplicate_event" not in result.quality.codes
    (observation,) = result.of_kind("operation")[0].observations.values()
    assert set(observation.events) == {EventKind.REQUEST_OBSERVED, EventKind.RESULT_PUBLISHED}


def test_semantic_duplicate_counts_one_boundary_and_operation() -> None:
    start = opening()
    duplicate = changed(start, event_sequence=start.record.event_sequence + 1)
    result = reconstruct((start, duplicate))
    (observation,) = result.of_kind("operation")[0].observations.values()
    assert len(observation.events) == 1
    assert "duplicate_event" in result.quality.codes
    assert result.quality.anomalous_operation_count_lower_bound == 1


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"raw_time_ns": 201}, "event_conflict"),
        ({"data1": (1 << 32) | 98}, "event_conflict"),
    ],
)
def test_conflicting_anomaly_boundary_is_rejected(changes: dict[str, int], code: str) -> None:
    start = opening()
    duplicate = changed(start, event_sequence=start.record.event_sequence + 1, **changes)
    with pytest.raises(ObservationError) as caught:
        reconstruct((start, duplicate))
    assert caught.value.code == code
    assert len(caught.value.sources) == 2


@pytest.mark.parametrize("field,value", [("data1", (1 << 32) | 98), ("data3", 77), ("data4", 99)])
def test_conflicting_repeated_raw_facts_are_rejected(field: str, value: int) -> None:
    start = opening()
    end = changed(terminal(start), **{field: value})
    with pytest.raises(ObservationError) as caught:
        reconstruct((start, end))
    assert caught.value.code == "identity_conflict"
    assert len(caught.value.sources) == 2


def test_conflicting_repeated_local_result_is_rejected() -> None:
    end = terminal(opening())
    other = changed(end, event_sequence=end.record.event_sequence + 1, result_code=ResultCode.SHUTDOWN)
    with pytest.raises(ObservationError) as caught:
        reconstruct((end, other))
    assert caught.value.code == "event_conflict"


def test_ordinary_and_anomalous_representations_of_one_operation_conflict() -> None:
    ordinary = next(e for e in events_for(scenario("normal_dense")) if e.record.kind == EventKind.REQUEST_OBSERVED)
    anomaly = changed(opening(), event_sequence=ordinary.record.event_sequence + 100)
    with pytest.raises(ObservationError) as caught:
        reconstruct((ordinary, anomaly))
    assert caught.value.code == "identity_conflict"


def test_absent_anomalies_never_establish_exact_zero_requests() -> None:
    result = reconstruct((), metadata=(opening().metadata,))
    assert result.quality.anomalous_operation_count_lower_bound == 0
    assert result.quality.anomalous_operation_count_exact is False
    assert "tail_unknown" in result.quality.codes


def test_quality_reference_counts_operations_rather_than_event_attempts() -> None:
    case = next(c for c in fixture("quality")["cases"] if c["id"] == "anomaly_count_is_only_an_observed_lower_bound")
    events = tuple(
        changed(opening(), operation_sequence=operation) for operation in case["input"]["retained_anomalous_operations"]
    )
    events = tuple(changed(event, event_sequence=i + 1) for i, event in enumerate(events))
    data = fixture("quality")["checkpoint"]
    attempted, dropped, lost = (case["input"][key] for key in ("event_attempts", "dropped_full", "post_commit_lost"))
    data.update(
        producer_id=events[0].source.identity.producer,
        collection_state="stopped",
        export_state="failed",
        stop_reason="io_error",
        tail=case["input"]["tail"],
        pending=None,
        exact_totals=False,
        last_attempted_sequence=attempted,
    )
    data["counters"].update(
        attempted=attempted,
        committed=attempted - dropped,
        dropped_full=dropped,
        post_commit_lost=lost,
        copied=attempted - dropped - lost,
        exported=attempted - dropped - lost,
    )
    point = QualityCheckpoint.model_validate(data)
    quality = reconstruct(events, checkpoints=(point,)).quality
    assert {"tail_unknown", "dropped_full", "post_commit_lost"} <= quality.codes
    assert quality.anomalous_operation_count_lower_bound == case["expected"]["anomalous_operation_count_lower_bound"]
    assert quality.anomalous_operation_count_exact == case["expected"]["anomalous_operation_count_exact"]
    empty = reconstruct((), metadata=(events[0].metadata,), checkpoints=(point,)).quality
    assert empty.anomalous_operation_count_lower_bound == 0
    assert empty.anomalous_operation_count_exact is False


@pytest.mark.parametrize("raw", [0, 2**31, 2**32 - 1])
def test_raw_enum_preserves_the_complete_received_u32_pattern(raw: int) -> None:
    start = changed(opening(), data1=(1 << 32) | raw)
    (observation,) = reconstruct((start,)).of_kind("operation")[0].observations.values()
    assert observation.facts["raw_forward_mode"] == raw


def test_legal_enums_with_invalid_request_geometry_remain_ordinary_evidence() -> None:
    ordinary = next(e for e in events_for(scenario("normal_dense")) if e.record.kind == EventKind.REQUEST_OBSERVED)
    event = changed(ordinary, payload_rows=2**64 - 1)
    result = reconstruct((event,))
    assert "invalid_request" not in result.quality.codes
    assert result.events[event.source.identity].record.payload_rows == 2**64 - 1


def test_terminal_only_anomaly_has_no_borrowed_instance_result() -> None:
    result = reconstruct((terminal(opening()),))
    assert len(result.of_kind("operation")[0].observations) == 1
    assert not any(event.record.role == EventRole.INSTANCE for event in result.events.values())


def test_reversed_anomaly_times_preserve_events_and_report_temporal_conflict() -> None:
    start = opening()
    end = terminal(start, time=start.record.raw_time_ns - 1)
    result = reconstruct((start, end))
    assert len(result.events) == 2
    assert "temporal_conflict" in result.quality.codes
    assert all(interval.duration_ns is None for interval in result.intervals)
