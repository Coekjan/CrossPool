from __future__ import annotations

import pytest
from pydantic import ValidationError

from xpool.runtime.transport import InstanceRankTransportProfile


def test_transport_attributes_accept_supported_geometry() -> None:
    attributes = InstanceRankTransportProfile(
        hidden_size=7168,
        payload_row_capacity=4096,
        atn_tp_rank=1,
        atn_tp_size=2,
        atn_dp_rank=0,
        atn_dp_size=1,
    )

    assert attributes.hidden_size == 7168


def test_transport_attributes_accept_odd_hidden_size() -> None:
    attributes = InstanceRankTransportProfile(
        hidden_size=3,
        payload_row_capacity=1,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )

    assert attributes.hidden_size == 3


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"hidden_size": 0}, "greater than or equal to 1"),
        ({"payload_row_capacity": 0}, "greater than or equal to 1"),
        ({"atn_tp_rank": 2}, "atn_tp_rank must be smaller than atn_tp_size"),
        ({"atn_dp_rank": 1}, "atn_dp_rank must be smaller than atn_dp_size"),
        ({"atn_dp_size": 2}, "combined attention TP-by-DP"),
    ],
)
def test_transport_attributes_reject_invalid_geometry(
    updates: dict[str, int],
    message: str,
) -> None:
    payload = {
        "hidden_size": 4,
        "payload_row_capacity": 1,
        "atn_tp_rank": 0,
        "atn_tp_size": 2,
        "atn_dp_rank": 0,
        "atn_dp_size": 1,
    }

    with pytest.raises(ValidationError, match=message):
        InstanceRankTransportProfile.model_validate(payload | updates)
