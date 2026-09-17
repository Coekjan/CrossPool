"""Install isolated graph-observer fixtures and inspect their event streams."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pytest

import xpool.integrations.sglang.devkit.graph_observer
from xpool.config import XpoolConfig


@pytest.fixture
def reset_graph_observer(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> Iterator[None]:
    event_handle = xpool.integrations.sglang.devkit.graph_observer.event_handle
    if event_handle is not None and not event_handle.closed:
        event_handle.close()
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "event_file", None)
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "event_handle", None)
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "installed", False)
    yield
    event_handle = xpool.integrations.sglang.devkit.graph_observer.event_handle
    if event_handle is not None and not event_handle.closed:
        event_handle.close()
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "event_file", None)
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "event_handle", None)
    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "installed", False)


class FullCudaGraphBackend:
    """Fake backend whose name matches SGLang's full backend evidence."""


class FakeDecodeCudaGraphRunnerState:
    def __init__(self) -> None:
        self.backend = FullCudaGraphBackend()


class BreakableCudaGraphBackend:
    """Fake backend whose name matches SGLang's Breakable backend evidence."""


class FakePrefillCudaGraphRunnerState:
    def __init__(self) -> None:
        self.backend = BreakableCudaGraphBackend()


class SuccessfulGraphRunner(Protocol):
    """Call surface exercised after observer installation."""

    def capture(self) -> str: ...

    def execute(self) -> str: ...


def successful_runner_classes() -> tuple[type[SuccessfulGraphRunner], type[SuccessfulGraphRunner]]:
    """Return decode and prefill runner classes with deterministic calls."""

    class DecodeCudaGraphRunner(FakeDecodeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def execute(self) -> str:
            return "executed"

    class PrefillCudaGraphRunner(FakePrefillCudaGraphRunnerState):
        def capture(self) -> str:
            return "prefill-captured"

        def execute(self) -> str:
            return "prefill-executed"

    return DecodeCudaGraphRunner, PrefillCudaGraphRunner


def install_fake_sglang_runner_classes(
    monkeypatch: pytest.MonkeyPatch,
    decode_cuda_graph_runner: type[object],
    prefill_cuda_graph_runner: type[object],
) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "DecodeCudaGraphRunner",
        decode_cuda_graph_runner,
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "PrefillCudaGraphRunner",
        prefill_cuda_graph_runner,
    )


def read_events(event_dir: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for event_file in sorted(event_dir.glob("xpool.graph-observer.*.jsonl")):
        for line in event_file.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def graph_observer_config(outdir: Path) -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )
