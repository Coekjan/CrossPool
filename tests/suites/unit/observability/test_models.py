import pytest
from pydantic import ValidationError
from tests.support.observability import events_for, scenario

from xpool.observability import EventRecord


@pytest.mark.parametrize("pe", [-(2**31), 2**31 - 1])
def test_event_record_preserves_full_signed_pe_width(pe: int) -> None:
    fields = events_for(scenario("normal_dense"))[0].record.model_dump()
    record = EventRecord.model_validate(fields | {"pe": pe})
    assert record.pe == pe


@pytest.mark.parametrize("pe", [-(2**31) - 1, 2**31, True, 1.5])
def test_event_record_rejects_invalid_signed_pe_representation(pe: int | float) -> None:
    fields = events_for(scenario("normal_dense"))[0].record.model_dump()
    with pytest.raises(ValidationError):
        EventRecord.model_validate(fields | {"pe": pe})
