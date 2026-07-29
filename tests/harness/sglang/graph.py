"""Model requested SGLang graph modes and read graph-observer evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

type JsonValue = str | int | float | bool | list[int] | None

type GraphEvent = dict[str, JsonValue]

type GraphSettings = tuple[bool, bool]


@dataclass(frozen=True, slots=True)
class SglangGraphSettings:
    """Requested full and piecewise CUDA graph modes for one SGLang server run."""

    cuda_graph: bool
    piecewise_cuda_graph: bool

    def id(self) -> str:
        """Return a deterministic artifact suffix for this graph mode."""

        return f"full-{int(self.cuda_graph)}-piecewise-{int(self.piecewise_cuda_graph)}"


class SglangGraphMode(StrEnum):
    """Declarative graph modes accepted by the E2E manifest."""

    EAGER = "eager"
    FULL = "full"
    PIECEWISE = "piecewise"

    def settings(self) -> SglangGraphSettings:
        """Project this manifest value to concrete SGLang graph settings."""

        match self:
            case SglangGraphMode.EAGER:
                return SglangGraphSettings(cuda_graph=False, piecewise_cuda_graph=False)
            case SglangGraphMode.FULL:
                return SglangGraphSettings(cuda_graph=True, piecewise_cuda_graph=False)
            case SglangGraphMode.PIECEWISE:
                return SglangGraphSettings(cuda_graph=False, piecewise_cuda_graph=True)


def read_graph_events(outdir: Path) -> list[GraphEvent]:
    events: list[GraphEvent] = []
    for path in sorted(outdir.glob("xpool.graph-observer.*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    events.append(cast(GraphEvent, json.loads(line)))
    return events
