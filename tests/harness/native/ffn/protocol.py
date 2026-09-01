"""Typed child-process messages for installed real-FFN qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from xpool.fabric import FabricGenerationId, InstanceFfnProfile
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement
from xpool.runtime.transport import InstanceRankTransportProfile


class FfnInstanceCommand(StrEnum):
    """Commands accepted by one attached Instance-rank child."""

    RUN = "run"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class FfnInvocationSpec:
    """One input/output exchange through the sole FFN shim."""

    case_id: str
    input_path: Path
    output_path: Path
    layer_ordinal: int
    forward_mode: ForwardMode
    output_requirement: OutputRequirement
    dp_row_layout: DpRowLayout
    dp_rank_payload_rows: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class FfnInstanceSpec:
    """Complete startup and execution contract for one Instance rank."""

    environment: dict[str, str]
    instance_id: str
    rank: int
    ffn_tp_size: int
    ffn_profile: InstanceFfnProfile
    transport: InstanceRankTransportProfile
    invocations: tuple[FfnInvocationSpec, ...]


@dataclass(frozen=True, slots=True)
class FfnInstanceReady:
    """Child acknowledgement carrying the admitted production identity."""

    generation: FabricGenerationId
    instance_index: int
    layer_execution_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class FfnInstanceCompleted:
    """Child acknowledgement that every invocation output is durable."""

    case_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FfnInstanceClosed:
    """Child acknowledgement that registration and arena ownership ended."""
