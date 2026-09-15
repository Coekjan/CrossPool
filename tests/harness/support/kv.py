"""Shared elastic KV geometry for focused tests."""

from __future__ import annotations

from xpool.service.wire import KvCapacityPartitionProfile


def kv_capacity_profile() -> KvCapacityPartitionProfile:
    """Return one internally consistent elastic KV partition profile."""

    return KvCapacityPartitionProfile(
        bundle_bytes=4096,
        bundle_capacity=8,
        floor_bundles=1,
        token_capacity=28,
        mapping_granularity_bytes=4096,
        row_bytes=1024,
        tokens_per_row=1,
        token_page_size=4,
    )
