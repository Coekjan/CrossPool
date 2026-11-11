"""Owned logical cache identities exchanged by state implementations."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Literal

from sglang.srt.mem_cache.radix_cache import RadixKey

from xpool.utils.integer import ensure_uint64


@dataclass(frozen=True, slots=True)
class StateCacheKey:
    """Immutable complete cache namespace and full-width media/token identity."""

    tokens: tuple[int, ...]
    extra_key: str | None
    is_bigram: bool
    cache_salt: str | None

    @classmethod
    def capture(cls, key: RadixKey) -> StateCacheKey:
        return cls(tuple(key.raw_token_ids()), key.extra_key, key.is_bigram, key.cache_salt)

    def radix_key(self) -> RadixKey:
        return RadixKey(array("q", self.tokens), self.extra_key, is_bigram=self.is_bigram, cache_salt=self.cache_salt)


@dataclass(frozen=True, slots=True)
class StateCacheEffect:
    """One ordered logical free, independent of rank-local physical IDs."""

    key: StateCacheKey
    pool_index: int
    unit_count: int

    def __post_init__(self) -> None:
        ensure_uint64(self.pool_index, self.unit_count)


@dataclass(frozen=True, slots=True)
class StateCacheProposal:
    """Exact endpoint or single-resource operation and its complete cascade."""

    key: StateCacheKey
    pool_index: int
    scope: Literal["endpoint", "resource"]
    effects: tuple[StateCacheEffect, ...]

    def __post_init__(self) -> None:
        ensure_uint64(self.pool_index)
        if self.scope not in ("endpoint", "resource") or not self.effects:
            raise ValueError("state cache proposals require a scope and logical effects")
