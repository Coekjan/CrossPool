"""Cumulative checkpoint conservation and unknown final evidence."""

import copy

import pytest
from pydantic import ValidationError
from tests.support.observability import FixtureData, events_for, fixture, scenario

from xpool.observability import ObservationError, QualityCheckpoint, reconstruct


def checkpoint(revision: int, attempted: int, exported: int) -> QualityCheckpoint:
    data = copy.deepcopy(fixture("quality")["checkpoint"])
    data["revision"] = revision
    data["host_time_ns"] = revision * 100
    data["last_attempted_sequence"] = attempted
    data["counters"].update(
        attempted=attempted, committed=attempted, copied=exported, exported=exported, dropped_full=0, abandoned=0
    )
    return QualityCheckpoint.model_validate(data)


def test_cumulative_checkpoints_are_not_summed() -> None:
    case = next(case for case in fixture("quality")["cases"] if case["id"] == "cumulative_checkpoints")
    points = [
        checkpoint(i + 1, item["attempted"], item["exported"]) for i, item in enumerate(case["input"]["checkpoints"])
    ]
    result = reconstruct((), checkpoints=reversed(points))
    latest = result.quality.checkpoints[(points[0].run_id, points[0].producer_id)]
    assert latest.counters.attempted == case["expected"]["attempted"]
    assert latest.counters.exported == case["expected"]["exported"]


@pytest.mark.parametrize("case", fixture("quality")["cases"][:3], ids=lambda case: case["id"])
def test_checkpoint_conservation_fixtures(case: FixtureData) -> None:
    data = copy.deepcopy(fixture("quality")["checkpoint"])
    values = copy.deepcopy(case["input"])
    counters = values.pop("counters")
    data["counters"].update(counters)
    data.update(values)
    data["last_attempted_sequence"] = counters["attempted"]
    point = QualityCheckpoint.model_validate(data)
    result = reconstruct((), checkpoints=(point,))
    actual = result.quality.final_counters[(point.run_id, point.producer_id)]
    if point.collection_state == "active":
        assert actual is None
    else:
        assert actual is not None and actual["attempted"] == counters["attempted"]
    if point.stop_reason:
        assert point.stop_reason in result.quality.codes


def test_unknown_tail_never_upgrades_prior_counters_to_final_totals() -> None:
    point = checkpoint(1, 5, 3).model_copy(update={"tail": "unknown", "pending": None})
    result = reconstruct((), checkpoints=(point,))
    assert result.quality.final_counters[(point.run_id, point.producer_id)] is None
    assert "tail_unknown" in result.quality.codes


def test_counter_saturation_retains_lower_bound_without_wrapping() -> None:
    data = fixture("quality")["checkpoint"]
    data["counters"]["attempted"] = 2**64 - 1
    data.update(counter_exhausted=True, exact_totals=False)
    point = QualityCheckpoint.model_validate(data)
    result = reconstruct((), checkpoints=(point,))
    assert point.counters.attempted == 2**64 - 1
    assert result.quality.final_counters[(point.run_id, point.producer_id)] is None
    assert "counter_exhausted" in result.quality.codes
    assert "totals_inexact" in result.quality.codes


def test_invalid_conservation_and_regression_are_rejected() -> None:
    data = fixture("quality")["checkpoint"]
    data["counters"]["exported"] = 10
    with pytest.raises(ValidationError):
        QualityCheckpoint.model_validate(data)
    with pytest.raises(ObservationError) as caught:
        reconstruct((), checkpoints=(checkpoint(1, 5, 3), checkpoint(2, 3, 2)))
    assert caught.value.code == "identity_conflict"


@pytest.mark.parametrize(
    "changes,final",
    [
        ({}, True),
        ({"collection_state": "stopped", "export_state": "failed"}, True),
        ({"collection_state": "active"}, False),
        ({"export_state": "active"}, False),
        ({"tail": "unknown"}, False),
        ({"tail": "pending"}, False),
        ({"exact_totals": False}, False),
    ],
)
def test_final_counters_require_exact_closed_evidence(changes: FixtureData, final: bool) -> None:
    data = fixture("quality")["checkpoint"]
    data.update(collection_state="closed", export_state="closed", tail="complete", exact_totals=True)
    data.update(changes)
    point = QualityCheckpoint.model_validate(data)
    result = reconstruct((), checkpoints=(point,))
    counters = result.quality.final_counters[(point.run_id, point.producer_id)]
    assert (counters is not None) == final
    if final:
        assert counters == point.counters.model_dump()


def test_single_pass_inputs_preserve_the_complete_reconstruction() -> None:
    events = events_for(scenario("normal_dense"))
    points = (checkpoint(1, 50, 50),)
    metadata = (events[0].metadata,)
    expected = reconstruct(events, metadata=metadata, checkpoints=points)
    actual = reconstruct(iter(events), metadata=iter(metadata), checkpoints=iter(points))
    assert actual == expected
