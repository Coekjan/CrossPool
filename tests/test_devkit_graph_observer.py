from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import xpool.config as config_module
import xpool.devkit.sglang.plugins.graph_observer as graph_observer
from xpool.config import XpoolConfig, init_global_config


@pytest.fixture(autouse=True)
def reset_graph_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    monkeypatch.setattr(graph_observer, "_event_file", None)
    monkeypatch.setattr(graph_observer, "_installed", False)
    yield
    monkeypatch.setattr(graph_observer, "_event_file", None)
    monkeypatch.setattr(graph_observer, "_installed", False)
    monkeypatch.setattr(config_module, "_global_config", None)


def test_graph_observer_records_success_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> None:
            raise RuntimeError("capture failed")

        def replay(self) -> None:
            return None

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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


def test_graph_observer_install_truncates_process_event_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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


def test_graph_observer_write_failure_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()
    bad_event_file = tmp_path / "event-file-as-directory"
    bad_event_file.mkdir()
    monkeypatch.setattr(graph_observer, "_event_file", bad_event_file)

    with caplog.at_level("WARNING", logger="xpool.devkit.sglang.plugins.graph_observer"):
        assert CudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text


def test_graph_observer_payload_failure_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def __init__(self) -> None:
            super().__init__()
            self.capture_forward_mode = None

        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)
    init_global_config(config=graph_observer_config(tmp_path.resolve()))
    graph_observer.install()

    with caplog.at_level("WARNING", logger="xpool.devkit.sglang.plugins.graph_observer"):
        assert CudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text


def test_graph_observer_is_idempotent_and_updates_outdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
        def capture(self) -> str:
            return "pcg-captured"

        def replay(self) -> str:
            return "pcg-replayed"

    install_fake_sglang_runner_classes(monkeypatch, CudaGraphRunner, PiecewiseCudaGraphRunner)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    init_global_config(config=graph_observer_config(first_dir.resolve()))
    graph_observer.install()
    first_capture = CudaGraphRunner.capture
    init_global_config(config=graph_observer_config(second_dir.resolve()))
    graph_observer.install()

    assert CudaGraphRunner.capture is first_capture

    assert CudaGraphRunner().capture() == "captured"
    assert not first_dir.exists() or read_events(first_dir) == []
    assert [(event["kind"], event["phase"]) for event in read_events(second_dir)] == [
        ("full_cuda_graph", "capture_begin"),
        ("full_cuda_graph", "capture_end"),
    ]


def test_graph_observer_accepts_relative_outdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CudaGraphRunner(_FakeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def replay(self) -> str:
            return "replayed"

    class PiecewiseCudaGraphRunner(_FakePiecewiseCudaGraphRunnerState):
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


class _FakeMode:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCudaGraphRunnerState:
    def __init__(self) -> None:
        self.device = "cuda"
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.capture_forward_mode = _FakeMode("DECODE")
        self.capture_bs = [1, 2]
        self.max_bs = 2
        self.max_num_token = 2


class _FakePiecewiseCudaGraphRunnerState:
    def __init__(self) -> None:
        self.device = "cuda"
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.capture_forward_mode = _FakeMode("EXTEND")
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
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )
