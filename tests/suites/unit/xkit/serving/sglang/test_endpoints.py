from __future__ import annotations

import pytest
from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA, ZMQ_TCP_PORT_DELTA

from xkit.serving.sglang.endpoints import (
    SglangEndpointFamily,
)


def test_endpoint_family_projects_pinned_sglang_dp_ports() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 2)

    assert family.ports == (
        20_000,
        21_000,
        22_000,
        20_000 + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
        *(20_000 + ZMQ_TCP_PORT_DELTA + offset for offset in range(7)),
    )


def test_endpoint_family_dp_one_owns_exact_root_endpoints() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)

    assert family.ports == (20_000, 21_000, 22_000)


def test_endpoint_family_rejects_duplicate_root_endpoints() -> None:
    with pytest.raises(ValueError, match="duplicate ports"):
        SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 20_000, 1)
