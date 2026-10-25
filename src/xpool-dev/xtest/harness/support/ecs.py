"""Shared Elastic Context State byte bounds for focused tests."""

from __future__ import annotations

from xpool.service.wire import ElasticContextStatePartitionProfile


def context_state_profile() -> ElasticContextStatePartitionProfile:
    """Return a small physical floor and reservation for one attention rank."""

    return ElasticContextStatePartitionProfile(
        floor_bytes=4096,
        reservation_bytes=32768,
    )
