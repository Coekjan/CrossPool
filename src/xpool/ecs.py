"""Elastic Context State physical geometry and rank-local shared byte permission."""

from __future__ import annotations

from dataclasses import dataclass, field

from xpool.utils.integer import ensure_uint64

__all__ = ["ElasticContextStateProfile", "MemoryQuota", "PoolProfile"]


@dataclass(frozen=True, slots=True)
class PoolProfile:
    """Immutable physical bundle cost and reservation bounds for one local pool."""

    bundle_bytes: int
    bundle_capacity: int
    floor_bundles: int

    def __post_init__(self) -> None:
        ensure_uint64(self.bundle_bytes, self.bundle_capacity, self.floor_bundles)
        if self.bundle_bytes == 0 or self.floor_bundles > self.bundle_capacity:
            raise ValueError(
                "elastic context state pool requires positive bundle bytes and a floor within its reservation"
            )
        ensure_uint64(self.bundle_bytes * self.bundle_capacity)

    def bytes_for(self, bundles: int) -> int:
        """Return physical cost for a count within this pool's reservation."""

        ensure_uint64(bundles)
        if bundles > self.bundle_capacity:
            raise ValueError("elastic context state bundle count exceeds the pool reservation")
        return bundles * self.bundle_bytes


@dataclass(frozen=True, slots=True)
class ElasticContextStateProfile:
    """Ordered local pool geometry; it contains no mutable residency inventory."""

    pools: tuple[PoolProfile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pools, tuple):
            raise ValueError("elastic context state pools require an immutable tuple")
        ensure_uint64(self.floor_bytes, self.reservation_bytes)

    @property
    def floor_bytes(self) -> int:
        """Return the joint minimum retained backing cost."""

        return sum(pool.floor_bundles * pool.bundle_bytes for pool in self.pools)

    @property
    def reservation_bytes(self) -> int:
        """Return the physical upper bound across the complete pool inventory."""

        return sum(pool.bundle_capacity * pool.bundle_bytes for pool in self.pools)


@dataclass(slots=True)
class MemoryQuota:
    """Shared permission ledger for all managed backings of one Instance Rank.

    Reserve before creating physical storage. Pending, resident and retiring
    bytes retain one charge until their backing confirms rollback or release.
    Logical free alone is not a refund. The startup/scheduler owner serializes
    mutations; this ledger performs no device operation or synchronization.
    """

    limit_bytes: int
    charged_bytes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        ensure_uint64(self.limit_bytes)

    def try_reserve(self, byte_count: int) -> bool:
        """Charge physical growth, leaving the ledger unchanged on quota shortage."""

        ensure_uint64(byte_count)
        if byte_count > self.limit_bytes - self.charged_bytes:
            return False
        self.charged_bytes += byte_count
        return True

    def release(self, byte_count: int) -> None:
        """Refund confirmed physical releases; reject an excessive refund."""

        ensure_uint64(byte_count)
        if byte_count > self.charged_bytes:
            raise ValueError("elastic context state quota release exceeds its retained charge")
        self.charged_bytes -= byte_count

    def set_limit(self, limit_bytes: int) -> None:
        """Adopt a grant only after retained charges fit it."""

        ensure_uint64(limit_bytes)
        if limit_bytes < self.charged_bytes:
            raise ValueError("elastic context state quota limit is below its retained charge")
        self.limit_bytes = limit_bytes
