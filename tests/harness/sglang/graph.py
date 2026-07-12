from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tests.harness.sglang.offline_probe import (
    SglangGraphSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

type JsonValue = str | int | float | bool | list[int] | None

type GraphEvent = dict[str, JsonValue]

type GraphSettings = tuple[bool, bool]

PROBE_TIMEOUT_SECONDS = 30 * 60

PROCESS_TERMINATE_TIMEOUT_SECONDS = 30

PROCESS_KILL_TIMEOUT_SECONDS = 30

PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS = 30


def read_graph_events(outdir: Path) -> list[GraphEvent]:
    events: list[GraphEvent] = []
    for path in sorted(outdir.glob("xpool.graph-observer.*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    events.append(cast(GraphEvent, json.loads(line)))
    return events


def assert_graph_events(graph_settings: SglangGraphSettings, events: list[GraphEvent]) -> None:
    full_graph_phases = graph_phases(events, kind="full_cuda_graph")
    pcg_phases = graph_phases(events, kind="piecewise_cuda_graph")
    full_graph_phases_expected = {"capture_begin", "capture_end", "replay_begin", "replay_end"}
    piecewise_graph_phases_expected = {"capture_begin", "capture_end", "replay_begin", "replay_end"}
    if graph_settings.cuda_graph:
        assert full_graph_phases_expected <= full_graph_phases
    else:
        assert full_graph_phases == set()
    if graph_settings.piecewise_cuda_graph:
        assert piecewise_graph_phases_expected <= pcg_phases
    else:
        assert pcg_phases == set()


def graph_phases(events: list[GraphEvent], *, kind: str) -> set[str]:
    phases: set[str] = set()
    for event in events:
        if event.get("kind") == kind and isinstance(event.get("phase"), str):
            phases.add(cast(str, event["phase"]))
    return phases
