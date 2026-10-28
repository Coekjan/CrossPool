"""Immutable event identity and physical source locations."""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["EventIdentity", "Source"]


@dataclass(frozen=True, order=True)
class EventIdentity:
    """One attempted event isolated by Run and registered Producer."""

    run_id: str
    producer: int
    event_sequence: int


@dataclass(frozen=True)
class Source:
    """Physical and protocol provenance at a byte offset in a published file.

    Process creation distinguishes reused PIDs. Endpoint, Instance Rank and PE
    remain absent when their semantic presence was not recorded. The raw clock
    domain is retained independently of any later calibration.
    """

    identity: EventIdentity
    path: Path
    offset: int
    process_creation_id: str
    clock_domain_id: int
    endpoint: tuple[int, int] | None
    instance_rank: int | None
    pe: int | None
