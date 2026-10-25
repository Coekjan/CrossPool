from __future__ import annotations

import pytest

from xpool.service.wire import ElasticContextStatePartitionProfile


@pytest.mark.parametrize(("floor_bytes", "reservation_bytes"), [(0, 0), (4096, 32768)])
def test_attention_partition_profile_round_trips_byte_bounds(floor_bytes: int, reservation_bytes: int) -> None:
    profile = ElasticContextStatePartitionProfile(floor_bytes=floor_bytes, reservation_bytes=reservation_bytes)

    assert ElasticContextStatePartitionProfile.model_validate_json(profile.model_dump_json()) == profile


def test_attention_partition_profile_requires_floor_within_reservation() -> None:
    with pytest.raises(ValueError, match="floor exceeds its reservation"):
        ElasticContextStatePartitionProfile(floor_bytes=8192, reservation_bytes=4096)
