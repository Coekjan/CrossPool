"""Typed messages for subprocess-owned Transport arenas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from xpool.abi import TensorDType

TRANSPORT_COMMAND_TIMEOUT_SECONDS = 30.0


class TransportOwnerCommand(StrEnum):
    """Lifecycle commands accepted by a Transport owner child."""

    ACTIVATE = "activate"
    HEALTH = "health"
    DRAIN = "drain"
    DESTROY = "destroy"


class TransportOwnerState(StrEnum):
    """Command outcomes published by a Transport owner child."""

    ACTIVATED = "activated"
    HEALTHY = "healthy"
    FAILED = "failed"
    DRAINED = "drained"
    DESTROYED = "destroyed"


@dataclass(frozen=True, slots=True)
class TransportOwnerSpec:
    """Complete Transport owner startup configuration."""

    cuda_device: int
    max_tokens: int
    hidden_size: int
    dtype: TensorDType
    atn_dp_size: int
    activate_resident: bool
    atnagent_loopback_enabled: bool
    observer_output_path: Path | None
    instance_index: int
    instance_rank: int
    atn_tp_rank: int
    atn_tp_size: int
    atn_dp_rank: int


@dataclass(frozen=True, slots=True)
class TransportArenaPublished:
    """Owner startup acknowledgement carrying the native arena handle."""

    handle: str


@dataclass(frozen=True, slots=True)
class TransportOwnerStatus:
    """Typed response to one owner lifecycle command."""

    state: TransportOwnerState
    message: str = ""
