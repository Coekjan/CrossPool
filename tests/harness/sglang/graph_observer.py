from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import xpool.devkit.sglang.graph_observer as graph_observer
from xpool.config import XpoolConfig


@pytest.fixture(autouse=True)
def reset_graph_observer(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    event_handle = graph_observer.event_handle
    if event_handle is not None and not event_handle.closed:
        event_handle.close()
    monkeypatch.setattr(graph_observer, "event_file", None)
    monkeypatch.setattr(graph_observer, "event_handle", None)
    monkeypatch.setattr(graph_observer, "installed", False)
    yield
    event_handle = graph_observer.event_handle
    if event_handle is not None and not event_handle.closed:
        event_handle.close()
    monkeypatch.setattr(graph_observer, "event_file", None)
    monkeypatch.setattr(graph_observer, "event_handle", None)
    monkeypatch.setattr(graph_observer, "installed", False)


class FakeMode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCudaGraphRunnerState:
    def __init__(self) -> None:
        self.device = "cuda"
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.capture_forward_mode = FakeMode("DECODE")
        self.capture_bs = [1, 2]
        self.max_bs = 2
        self.max_num_token = 2


class FakePiecewiseCudaGraphRunnerState:
    def __init__(self) -> None:
        self.device = "cuda"
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.capture_forward_mode = FakeMode("EXTEND")
        self.capture_num_tokens = [4, 8]
        self.max_bs = 2
        self.max_num_tokens = 8


def install_fake_sglang_runner_classes(
    monkeypatch: pytest.MonkeyPatch,
    cuda_graph_runner: type[object],
    piecewise_cuda_graph_runner: type[object],
) -> None:
    monkeypatch.setattr(graph_observer, "CudaGraphRunner", cuda_graph_runner)
    monkeypatch.setattr(graph_observer, "PiecewiseCudaGraphRunner", piecewise_cuda_graph_runner)


def read_events(event_dir: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for event_file in sorted(event_dir.glob("xpool.graph-observer.*.jsonl")):
        for line in event_file.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def graph_observer_config(outdir: Path) -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )
