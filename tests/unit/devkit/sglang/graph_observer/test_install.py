from __future__ import annotations

import os

from tests.harness.sglang.graph_observer import (
    FakeCudaGraphRunnerState,
    FakePiecewiseCudaGraphRunnerState,
    Path,
    graph_observer,
    graph_observer_config,
    install_fake_sglang_runner_classes,
    pytest,
    read_events,
)
from xpool.config import init_global_config


def test_graph_observer_install_truncates_process_event_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    event_file.write_text("stale\n", encoding="utf-8")
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))

    graph_observer.install()

    assert event_file.read_text(encoding="utf-8") == ""


def test_graph_observer_repeat_install_preserves_process_event_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()
    first_capture = CudaGraphRunner.capture
    assert CudaGraphRunner().capture() == "captured"
    event_file_text = event_file.read_text(encoding="utf-8")

    graph_observer.install()

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
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    event_file = tmp_path / f"xpool.graph-observer.{os.getpid()}.jsonl"
    event_file.mkdir()
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))

    with pytest.raises(OSError):
        graph_observer.install()


def test_graph_observer_install_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)

    outdir = tmp_path / "events"
    config = graph_observer_config(outdir.resolve())
    init_global_config(config=config)
    graph_observer.install()
    first_capture = CudaGraphRunner.capture
    assert init_global_config(config=graph_observer_config(outdir.resolve())) is config
    graph_observer.install()

    assert CudaGraphRunner.capture is first_capture

    assert CudaGraphRunner().capture() == "captured"
    assert [(event["kind"], event["phase"]) for event in read_events(outdir)] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
    ]
