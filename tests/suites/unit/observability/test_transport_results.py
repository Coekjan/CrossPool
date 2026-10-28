"""Transport result evidence stays local to each actual boundary."""

from dataclasses import replace

import pytest
from tests.support.observability import events_for, scenario

from xpool.observability.applicability import boundary_applicability
from xpool.observability.errors import ObservationError
from xpool.observability.reconstruction import reconstruct
from xpool.observability.wire import EventKind, ResultCode


@pytest.mark.parametrize("retained", [(0, 1, 2), (0, 2), (1, 2), (2,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_boundary_results_survive_loss_and_input_order(retained: tuple[int, ...], reverse: bool) -> None:
    events = events_for(scenario("result_observed_then_shutdown"))
    selected = tuple(events[index] for index in retained)
    view = reconstruct(tuple(reversed(selected)) if reverse else selected)
    (observation,) = view.of_kind("operation")[0].observations.values()
    assert observation.transport_results == {
        EventKind(events[index].record.kind): ResultCode(events[index].record.result_code) for index in retained
    }
    assert "result_code" not in observation.facts
    assert "result_code" not in observation.fact_sources
    assert "unknown_result" not in view.quality.codes
    assert observation.transport_results[EventKind.CLOSED] == ResultCode.SHUTDOWN
    assert set(observation.events) == {event.record.kind for event in selected}


def test_same_boundary_different_result_still_conflicts() -> None:
    event = events_for(scenario("closed_shutdown_only"))[0]
    other = replace(
        event,
        source=replace(event.source, identity=replace(event.source.identity, event_sequence=9)),
        record=event.record.model_copy(update={"event_sequence": 9, "result_code": ResultCode.OK}),
    )
    with pytest.raises(ObservationError) as caught:
        reconstruct((event, other))
    assert caught.value.code == "event_conflict"
    assert set(caught.value.sources) == {event.source, other.source}


def test_changed_request_facts_still_conflict_across_distinct_boundaries() -> None:
    events = events_for(scenario("result_observed_then_shutdown"))
    changed = replace(events[-1], record=events[-1].record.model_copy(update={"payload_rows": 99}))
    with pytest.raises(ObservationError) as caught:
        reconstruct((events[0], changed))
    assert caught.value.code == "identity_conflict"


def test_results_are_derived_from_retained_events_without_caching() -> None:
    view = reconstruct(events_for(scenario("result_observed_then_shutdown")))
    (observation,) = view.of_kind("operation")[0].observations.values()
    before = observation.transport_results
    observation.events.pop(EventKind.CLOSED)
    assert EventKind.CLOSED in before
    assert EventKind.CLOSED not in observation.transport_results


def test_observed_success_without_terminal_keeps_shutdown_alternative() -> None:
    view = reconstruct(events_for(scenario("result_observed_then_shutdown"))[:2])
    (observation,) = view.of_kind("operation")[0].observations.values()
    path = boundary_applicability(observation, view.metadata)
    assert {EventKind.CLOSED, EventKind.RESULT_ACKNOWLEDGED} <= path.unknown
    assert not {EventKind.CLOSED, EventKind.RESULT_ACKNOWLEDGED} & path.inapplicable
