from __future__ import annotations

import pytest
from pydantic import ValidationError

from xpool.runtime.transport import InstanceTransportAttributes


def test_transport_attributes_accept_supported_geometry() -> None:
    attributes = InstanceTransportAttributes(
        element_size=2,
        hidden_size=7168,
        max_tokens=4096,
        atn_tp_rank=1,
        atn_tp_size=2,
        atn_dp_rank=0,
        atn_dp_size=1,
    )

    assert attributes.element_size == 2
    assert attributes.hidden_size == 7168


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"element_size": 1}, "element_size must be 2 or 4"),
        ({"element_size": 8}, "element_size must be 2 or 4"),
        ({"hidden_size": 0}, "hidden_size must be positive and even"),
        ({"hidden_size": 3}, "hidden_size must be positive and even"),
        ({"max_tokens": 0}, "greater than or equal to 1"),
        ({"atn_tp_rank": 2}, "atn_tp_rank must be smaller than atn_tp_size"),
        ({"atn_dp_rank": 1}, "atn_dp_rank must be smaller than atn_dp_size"),
    ],
)
def test_transport_attributes_reject_invalid_geometry(
    updates: dict[str, int],
    message: str,
) -> None:
    payload = {
        "element_size": 4,
        "hidden_size": 4,
        "max_tokens": 1,
        "atn_tp_rank": 0,
        "atn_tp_size": 2,
        "atn_dp_rank": 0,
        "atn_dp_size": 1,
    }

    with pytest.raises(ValidationError, match=message):
        InstanceTransportAttributes.model_validate(payload | updates)
