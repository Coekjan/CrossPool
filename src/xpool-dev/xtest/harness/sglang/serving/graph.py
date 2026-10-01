"""Model requested SGLang graph modes and read graph-observer evidence."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from xpool.integrations.sglang.devkit import SglangGraphEvent


def read_graph_events(outdir: Path) -> list[SglangGraphEvent]:
    """Read every process-local graph event in deterministic file and line order."""

    adapter = TypeAdapter(SglangGraphEvent)
    events: list[SglangGraphEvent] = []
    for path in sorted(outdir.glob("xpool.graph-observer.*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    events.append(adapter.validate_json(line, strict=True, extra="allow"))
    return events
