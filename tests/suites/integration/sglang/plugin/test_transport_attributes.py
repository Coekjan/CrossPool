from __future__ import annotations

from dataclasses import replace

import pytest

import xpool.integrations.sglang.plugin
from tests.harness.sglang.plugin import binding, ffn_workload


@pytest.mark.parametrize(
    ("topology", "expected_topology", "max_decode_rows", "max_prefill_rows", "expected_max_tokens"),
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
def test_transport_attributes_project_workload_and_attention_topology(
    topology: dict[str, int],
    expected_topology: dict[str, int],
    max_decode_rows: int,
    max_prefill_rows: int,
    expected_max_tokens: int,
) -> None:
    model_binding = replace(binding(), **topology)
    workload = ffn_workload(
        hidden_size=7168,
        max_decode_rows=max_decode_rows,
        max_prefill_rows=max_prefill_rows,
    )

    attributes = xpool.integrations.sglang.plugin.derive_transport_attributes(model_binding, workload)

    assert attributes.model_dump() == {
        "hidden_size": 7168,
        "max_tokens": expected_max_tokens,
        **expected_topology,
    }
