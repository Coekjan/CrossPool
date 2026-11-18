import pytest
from tests.support.observability import fixture

from xpool.observability.codec import kind_names
from xpool.observability.wire import (
    CalibrationEventKind,
    EventFamily,
    EventKind,
    FabricEventKind,
    GraphEventKind,
    RoutingEventKind,
    SampleFailure,
    SampleFailureCode,
    TransportEventKind,
    parse_event_kind,
)


@pytest.mark.parametrize("failure", tuple(SampleFailure))
def test_sample_failure_codes_round_trip_through_wire_values(failure: SampleFailure) -> None:
    assert SampleFailureCode(failure.code.value).wire == failure
    assert failure.code.name == failure.name


def test_unknown_sample_failure_is_rejected_in_both_representations() -> None:
    with pytest.raises(ValueError):
        SampleFailure(99)
    with pytest.raises(ValueError):
        SampleFailureCode("unknown")


@pytest.mark.parametrize(
    "enum,family,values",
    [
        (TransportEventKind, EventFamily.TRANSPORT, (*range(100, 111), 150)),
        (FabricEventKind, EventFamily.FABRIC, (*range(200, 206), *range(300, 308), *range(400, 409), 450)),
        (GraphEventKind, EventFamily.GRAPH, tuple(range(500, 513))),
        (RoutingEventKind, EventFamily.ROUTING, (550,)),
        (CalibrationEventKind, EventFamily.CALIBRATION, (600, 601)),
    ],
)
def test_event_family_owns_exact_wire_values(
    enum: type[EventKind], family: EventFamily, values: tuple[int, ...]
) -> None:
    assert tuple(int(kind) for kind in enum) == values
    assert {kind.family for kind in enum} == {family}
    for value in values:
        kind = parse_event_kind(value)
        assert type(kind) is enum
        assert kind == enum(value)
        assert kind.family.value == family.value


def test_event_kinds_match_independent_frozen_layout() -> None:
    kinds = [
        kind
        for enum in (TransportEventKind, FabricEventKind, GraphEventKind, RoutingEventKind, CalibrationEventKind)
        for kind in enum
    ]
    assert len({int(kind) for kind in kinds}) == len(kinds)
    expected = {entry["id"]: entry["name"] for entry in fixture("layout")["kinds"]}
    assert {int(kind): kind_names[kind] for kind in kinds} == expected
    assert {int(kind): kind.name.replace("_", "").lower() for kind in kinds} == {
        value: name.lower() for value, name in expected.items()
    }


@pytest.mark.parametrize("value", [0, 99, 111, 199, 206, 449, 451, 499, 513, 520, 549, 551, 599, 602, 2**32 - 1])
def test_unknown_event_wire_values_are_rejected(value: int) -> None:
    with pytest.raises(ValueError, match=f"unknown event kind {value}"):
        parse_event_kind(value)
