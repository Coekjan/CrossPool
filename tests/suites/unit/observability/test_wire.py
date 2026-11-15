import pytest

from xpool.observability.wire import SampleFailure, SampleFailureCode


@pytest.mark.parametrize("failure", tuple(SampleFailure))
def test_sample_failure_codes_round_trip_through_wire_values(failure: SampleFailure) -> None:
    assert SampleFailureCode(failure.code.value).wire == failure
    assert failure.code.name == failure.name


def test_unknown_sample_failure_is_rejected_in_both_representations() -> None:
    with pytest.raises(ValueError):
        SampleFailure(99)
    with pytest.raises(ValueError):
        SampleFailureCode("unknown")
