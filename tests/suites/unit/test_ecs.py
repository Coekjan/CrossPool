import pytest

from xpool import ecs


def test_context_state_profiles_price_heterogeneous_pools_and_bound_total_bytes() -> None:
    mib = 1 << 20
    profile = ecs.ElasticContextStateProfile((ecs.PoolProfile(8 * mib, 12, 2), ecs.PoolProfile(4 * mib, 8, 1)))

    assert profile.floor_bytes == 20 * mib
    assert profile.reservation_bytes == 128 * mib
    assert profile.pools[0].bytes_for(10) == 80 * mib
    assert profile.pools[1].bytes_for(6) == 24 * mib
    with pytest.raises(ValueError, match="reservation"):
        profile.pools[0].bytes_for(13)


@pytest.mark.parametrize("invalid", (-1, True, 1.5, 1 << 64))
def test_context_state_profile_rejects_invalid_wire_sizes(invalid: int | float) -> None:
    with pytest.raises(ValueError, match="uint64"):
        ecs.PoolProfile(8, invalid, 0)  # ty: ignore[invalid-argument-type]


def test_context_state_profiles_reject_invalid_geometry_and_total_overflow() -> None:
    with pytest.raises(ValueError, match="positive bundle"):
        ecs.PoolProfile(0, 1, 1)
    with pytest.raises(ValueError, match="floor"):
        ecs.PoolProfile(8, 1, 2)
    with pytest.raises(ValueError, match="uint64"):
        ecs.PoolProfile(1 << 63, 2, 0)
    with pytest.raises(ValueError, match="uint64"):
        ecs.ElasticContextStateProfile((ecs.PoolProfile(1 << 63, 1, 0), ecs.PoolProfile(1 << 63, 1, 0)))


def test_context_state_quota_preserves_charges_until_physical_release() -> None:
    mib = 1 << 20
    quota = ecs.MemoryQuota(112 * mib)
    assert quota.try_reserve(80 * mib)
    assert quota.try_reserve(24 * mib)
    assert not quota.try_reserve(16 * mib)
    assert quota.charged_bytes == 104 * mib

    quota.release(8 * mib)
    assert quota.try_reserve(16 * mib)
    assert quota.charged_bytes == 112 * mib
    with pytest.raises(ValueError, match="retained charge"):
        quota.set_limit(104 * mib)
    assert quota.limit_bytes == 112 * mib

    quota.release(16 * mib)
    quota.set_limit(104 * mib)
    with pytest.raises(ValueError, match="retained charge"):
        quota.release(97 * mib)
    assert quota.charged_bytes == 96 * mib
    assert quota.limit_bytes == 104 * mib
