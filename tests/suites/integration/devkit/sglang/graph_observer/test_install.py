from __future__ import annotations

import os
from pathlib import Path

import pytest

import xpool.integrations.sglang.devkit.graph_observer
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.sglang.graph_observer import (
    graph_observer_config,
    install_fake_sglang_runner_classes,
    read_events,
    reset_graph_observer,
    successful_runner_classes,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_graph_observer.__name__)


def test_graph_observer_install_truncates_process_event_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CudaGraphRunner, PiecewiseCudaGraphRunner = successful_runner_classes()
    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    event_file.write_text("stale\n", encoding="utf-8")
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    install_test_config(config=graph_observer_config(tmp_path.resolve()))

    xpool.integrations.sglang.devkit.graph_observer.install()

    assert event_file.read_text(encoding="utf-8") == ""


def test_graph_observer_repeat_install_preserves_process_event_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CudaGraphRunner, PiecewiseCudaGraphRunner = successful_runner_classes()
    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.integrations.sglang.devkit.graph_observer.install()
    first_capture = CudaGraphRunner.capture
    assert CudaGraphRunner().capture() == "captured"
    event_file_text = event_file.read_text(encoding="utf-8")

    xpool.integrations.sglang.devkit.graph_observer.install()

    assert CudaGraphRunner.capture is first_capture
    assert event_file.read_text(encoding="utf-8") == event_file_text

    assert CudaGraphRunner().capture() == "captured"
    assert [(event["kind"], event["phase"]) for event in read_events(tmp_path)] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
    ]


def test_graph_observer_install_fails_when_event_file_cannot_be_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CudaGraphRunner, PiecewiseCudaGraphRunner = successful_runner_classes()
    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    event_file.mkdir()
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    install_test_config(config=graph_observer_config(tmp_path.resolve()))

    with pytest.raises(OSError):
        xpool.integrations.sglang.devkit.graph_observer.install()
