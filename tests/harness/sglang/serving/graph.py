"""Model requested SGLang graph modes and read graph-observer evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

type JsonValue = str | int | float | bool | list[int] | None

type GraphEvent = dict[str, JsonValue]

type SglangGraphBackend = Literal["disabled", "full", "breakable"]


@dataclass(frozen=True, slots=True)
class SglangGraphSettings:
    """Requested decode and prefill CUDA graph backends for one server run."""

    decode_backend: SglangGraphBackend
    prefill_backend: SglangGraphBackend

    def id(self) -> str:
        """Return a deterministic artifact suffix for this graph mode."""

        return f"decode-{self.decode_backend}-prefill-{self.prefill_backend}"


class SglangGraphMode(StrEnum):
    """Declarative graph modes accepted by the E2E manifest."""

    EAGER = "eager"
    DECODE_FULL = "decode-full"
    PREFILL_BREAKABLE = "prefill-breakable"

    def settings(self) -> SglangGraphSettings:
        """Project this manifest value to concrete SGLang graph settings."""

        match self:
            case SglangGraphMode.EAGER:
                return SglangGraphSettings(decode_backend="disabled", prefill_backend="disabled")
            case SglangGraphMode.DECODE_FULL:
                return SglangGraphSettings(decode_backend="full", prefill_backend="disabled")
            case SglangGraphMode.PREFILL_BREAKABLE:
                return SglangGraphSettings(decode_backend="disabled", prefill_backend="breakable")


def read_graph_events(outdir: Path) -> list[GraphEvent]:
    """Read every process-local graph event in deterministic file and line order."""

    events: list[GraphEvent] = []
    for path in sorted(outdir.glob("xpool.graph-observer.*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    events.append(cast(GraphEvent, json.loads(line)))
    return events
