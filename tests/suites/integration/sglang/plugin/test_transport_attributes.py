from __future__ import annotations

from dataclasses import replace

import pytest

import xpool.integrations.sglang.hooks.lifecycle
from tests.harness.support.sglang.plugin import binding, ffn_profile


@pytest.mark.parametrize(
    (
        "topology",
        "expected_topology",
        "decode_payload_row_capacity",
        "prefill_payload_row_capacity",
        "expected_payload_row_capacity",
    ),
    [
        (
            {"atn_tp_rank": 1, "atn_tp_size": 2},
            {"atn_tp_rank": 1, "atn_tp_size": 2, "atn_dp_rank": 0, "atn_dp_size": 1},
            32,
            64,
            64,
        ),
        (
            {"atn_dp_rank": 1, "atn_dp_size": 2},
            {"atn_tp_rank": 0, "atn_tp_size": 1, "atn_dp_rank": 1, "atn_dp_size": 2},
            128,
            64,
            128,
        ),
    ],
)
def test_transport_attributes_project_ffn_profile_and_attention_topology(
    topology: dict[str, int],
    expected_topology: dict[str, int],
    decode_payload_row_capacity: int,
    prefill_payload_row_capacity: int,
    expected_payload_row_capacity: int,
) -> None:
    model_binding = replace(binding(), **topology)
    profile = ffn_profile(
        hidden_size=7168,
        decode_payload_row_capacity=decode_payload_row_capacity,
        prefill_payload_row_capacity=prefill_payload_row_capacity,
    )

    attributes = xpool.integrations.sglang.hooks.lifecycle.derive_instance_rank_transport_profile(
        model_binding, profile
    )

    assert attributes.model_dump() == {
        "hidden_size": 7168,
        "payload_row_capacity": expected_payload_row_capacity,
        **expected_topology,
    }
