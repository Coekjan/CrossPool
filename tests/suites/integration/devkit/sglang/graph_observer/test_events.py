from __future__ import annotations

from pathlib import Path

import pytest

import xpool.integrations.sglang.devkit.graph_observer
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.sglang.graph_observer import (
    FakeDecodeCudaGraphRunnerState,
    FakePrefillCudaGraphRunnerState,
    graph_observer_config,
    install_fake_sglang_runner_classes,
    read_events,
    reset_graph_observer,
    successful_runner_classes,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_graph_observer.__name__)


def test_graph_observer_records_success_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DecodeCudaGraphRunner, PrefillCudaGraphRunner = successful_runner_classes()
    install_fake_sglang_runner_classes(monkeypatch, DecodeCudaGraphRunner, PrefillCudaGraphRunner)

    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.integrations.sglang.devkit.graph_observer.install()

    decode_runner = DecodeCudaGraphRunner()
    prefill_runner = PrefillCudaGraphRunner()

    assert decode_runner.capture() == "captured"
    assert decode_runner.execute() == "executed"
    assert prefill_runner.capture() == "prefill-captured"
    assert prefill_runner.execute() == "prefill-executed"

    events = read_events(tmp_path)
    assert [(event["forward_phase"], event["event"]) for event in events] == [
        ("decode", "capture_begin"),
        ("decode", "capture_end"),
        ("decode", "execute_begin"),
        ("decode", "execute_end"),
        ("prefill", "capture_begin"),
        ("prefill", "capture_end"),
        ("prefill", "execute_begin"),
        ("prefill", "execute_end"),
    ]
    assert {event["backend_class"] for event in events[:4]} == {"FullCudaGraphBackend"}
    assert {event["backend_class"] for event in events[4:]} == {"BreakableCudaGraphBackend"}


def test_graph_observer_records_error_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DecodeCudaGraphRunner(FakeDecodeCudaGraphRunnerState):
        def capture(self) -> None:
            raise RuntimeError("capture failed")

        def execute(self) -> None:
            return None

    class PrefillCudaGraphRunner(FakePrefillCudaGraphRunnerState):
        def capture(self) -> None:
            return None

        def execute(self) -> None:
            return None

    install_fake_sglang_runner_classes(monkeypatch, DecodeCudaGraphRunner, PrefillCudaGraphRunner)
    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.integrations.sglang.devkit.graph_observer.install()

    with pytest.raises(RuntimeError, match="capture failed"):
        DecodeCudaGraphRunner().capture()

    events = read_events(tmp_path)
    assert [(event["forward_phase"], event["event"]) for event in events] == [
        ("decode", "capture_begin"),
        ("decode", "capture_error"),
    ]


def test_graph_observer_event_fault_does_not_block_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    install_fake_sglang_runner_classes(monkeypatch, DecodeCudaGraphRunner, PrefillCudaGraphRunner)
    install_test_config(config=graph_observer_config(tmp_path.resolve()))
    xpool.integrations.sglang.devkit.graph_observer.install()

    class BadEventHandle:
        closed = False

        def write(self, line: str) -> int:
            raise OSError("write failed")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(xpool.integrations.sglang.devkit.graph_observer, "event_handle", BadEventHandle())

    with caplog.at_level("WARNING", logger="xpool.integrations.sglang.devkit.graph_observer"):
        assert DecodeCudaGraphRunner().capture() == "captured"

    assert "Failed to record xpool graph observer event" in caplog.text
