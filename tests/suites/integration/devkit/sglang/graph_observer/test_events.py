from __future__ import annotations

from pathlib import Path

import pytest

import xpool.devkit.sglang.graph_observer
from tests.harness.config import install_test_config
from tests.harness.sglang.graph_observer import (
    FakeCudaGraphRunnerState,
    FakePiecewiseCudaGraphRunnerState,
    graph_observer_config,
    install_fake_sglang_runner_classes,
    read_events,
    reset_graph_observer,
    successful_runner_classes,
)

pytestmark = pytest.mark.usefixtures(reset_graph_observer.__name__)


def test_graph_observer_records_success_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CudaGraphRunner, PiecewiseCudaGraphRunner = successful_runner_classes()
    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)

    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.devkit.sglang.graph_observer.install()

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
    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.devkit.sglang.graph_observer.install()

    with pytest.raises(RuntimeError, match="capture failed"):
        CudaGraphRunner().capture()

    events = read_events(tmp_path)
    assert [(event["kind"], event["phase"]) for event in events] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_error"),
    ]


@pytest.mark.parametrize("fault", ["payload", "write"])
def test_graph_observer_event_fault_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fault: str,
) -> None:
    class CudaGraphRunner(FakeCudaGraphRunnerState):
        def __init__(self) -> None:
            super().__init__()
            if fault == "payload":
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
    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.devkit.sglang.graph_observer.install()

    if fault == "write":

        class BadEventHandle:
            closed = False

            def write(self, line: str) -> int:
                raise OSError("write failed")

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        monkeypatch.setattr(xpool.devkit.sglang.graph_observer, "event_handle", BadEventHandle())

    with caplog.at_level("WARNING", logger="xpool.devkit.sglang.graph_observer"):
        assert CudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text
