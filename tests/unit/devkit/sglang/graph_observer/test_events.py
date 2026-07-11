from __future__ import annotations

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


def test_graph_observer_records_success_events(
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

    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    decode_runner = CudaGraphRunner()
    pcg_runner = PiecewiseCudaGraphRunner()

    assert decode_runner.capture() == "captured"
    assert decode_runner.replay() == "replayed"
    assert pcg_runner.capture() == "pcg-captured"
    assert pcg_runner.replay() == "pcg-replayed"

    events = read_events(tmp_path)
    assert [(event["kind"], event["phase"]) for event in events] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
        ("full_cuda_graph", "replay_begin"),
        ("full_cuda_graph", "replay_end"),
        ("piecewise_cuda_graph", "capture_begin"),
        ("piecewise_cuda_graph", "capture_end"),
        ("piecewise_cuda_graph", "replay_begin"),
        ("piecewise_cuda_graph", "replay_end"),
    ]
    assert events[0]["capture_forward_mode"] == "DECODE"
    assert events[0]["capture_bs"] == [1, 2]
    assert events[0]["max_num_tokens"] == 2
    assert events[4]["capture_forward_mode"] == "EXTEND"
    assert events[4]["capture_num_tokens"] == [4, 8]
    assert events[4]["max_num_tokens"] == 8


def test_graph_observer_records_error_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def capture(self) -> None:
            raise RuntimeError("capture failed")

        def replay(self) -> None:
            return None

    class PiecewiseCudaGraphRunner(FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> None:
            return None

        def replay(self) -> None:
            return None

    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    with pytest.raises(RuntimeError, match="capture failed"):
        CudaGraphRunner().capture()

    events = read_events(tmp_path)
    assert [(event["kind"], event["phase"]) for event in events] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_error"),
    ]


def test_graph_observer_flushes_each_event(
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
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    assert CudaGraphRunner().capture() == "captured"

    assert [(event["kind"], event["phase"]) for event in read_events(tmp_path)] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
    ]


def test_graph_observer_write_failure_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    class BadEventHandle:
        closed = False

        def write(self, line: str) -> int:
            raise OSError("write failed")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(graph_observer, "event_handle", BadEventHandle())

    with caplog.at_level("WARNING", logger="xpool.devkit.sglang.graph_observer"):
        assert CudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text


def test_graph_observer_payload_failure_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def __init__(self) -> None:
            super().__init__()
            self.capture_forward_mode = None

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
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    with caplog.at_level("WARNING", logger="xpool.devkit.sglang.graph_observer"):
        assert CudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text


def test_graph_observer_accepts_relative_outdir(
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

    monkeypatch.chdir(tmp_path)
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)

    init_global_config(config=graph_observer_config(Path("relative/events")))
    graph_observer.install()

    assert CudaGraphRunner().capture() == "captured"
    assert [(event["kind"], event["phase"]) for event in read_events(tmp_path / "relative/events")] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
    ]
