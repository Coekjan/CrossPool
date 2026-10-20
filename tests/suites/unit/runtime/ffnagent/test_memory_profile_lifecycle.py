from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from cuda.bindings import runtime  # ty: ignore[unresolved-import]

from xpool.fabric import FabricPlan, FabricRole
from xpool.runtime.ffnagent.memory_profile import corpus, runner
from xpool.utils.cuda import CudaRuntimeError, is_terminal_device_error
from xtest.harness.support.config import synthetic_config


@pytest.fixture
def participant(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError]]:
    parent, child = multiprocessing.Pipe()
    events: list[str] = []
    recorders: list[runner.PhaseRecorder] = []
    capture_error = RuntimeError("capture failed")
    recorder_type = runner.PhaseRecorder

    def recorder(**kwargs: int) -> runner.PhaseRecorder:
        result = recorder_type(**kwargs)
        recorders.append(result)
        return result

    def capture(**kwargs: object) -> None:
        events.append("capture failed")
        raise capture_error

    def warmup(*args: object, **kwargs: object) -> object:
        events.append("warmup")
        return object()

    plan = cast(
        FabricPlan,
        SimpleNamespace(
            pe_placements=(SimpleNamespace(role=FabricRole.FFNAGENT, device=1),),
            instance_plans=(SimpleNamespace(ffn_profile=None),),
        ),
    )
    monkeypatch.setattr(runner, "initialize_child_signals", lambda mask: None)
    monkeypatch.setattr(runner, "init_global_config", lambda **kwargs: synthetic_config())
    monkeypatch.setattr(runner.bootstrap, "init", lambda device, role: None)
    monkeypatch.setattr(runner, "ensure_supported_cuda_allocator", lambda: None)
    monkeypatch.setattr(runner, "memory_info", lambda: (100, 100))
    monkeypatch.setattr(
        runner,
        "runtime",
        SimpleNamespace(
            cudaSetDevice=lambda device: (runtime.cudaError_t.cudaSuccess,),
        ),
    )
    monkeypatch.setattr(
        runner,
        "torch",
        SimpleNamespace(
            empty=warmup,
            uint8=torch.uint8,
            cuda=SimpleNamespace(
                synchronize=lambda *args: events.append("synchronize"),
                empty_cache=lambda: None,
            ),
        ),
    )
    monkeypatch.setattr(
        runner, "DeviceMemoryEstimator", lambda **kwargs: SimpleNamespace(allocation_ledger=lambda **kw: ())
    )
    monkeypatch.setattr(runner, "PhaseRecorder", recorder)
    monkeypatch.setattr(runner, "materialize_calibration_weights", lambda **kwargs: ())
    monkeypatch.setattr(runner, "project_fabric_arena", lambda plan: None)
    monkeypatch.setattr(runner.xpool.native.fabric, "join", lambda arena, pe: events.append("joined"))
    monkeypatch.setattr(runner, "FfnExecutionRegistry", SimpleNamespace(materialize=capture))
    monkeypatch.setattr(runner, "drain_and_finalize", lambda: events.append("finalized"))

    def run() -> None:
        runner.participant_child(
            child,
            config_path=Path("config.toml"),
            config_environment={},
            model_specs=(),
            fabric_plan=plan,
            pe=0,
            environment={},
            signal_mask=set(),
        )

    try:
        yield run, parent, events, recorders, capture_error
    finally:
        for active in recorders:
            active.stop_requested.set()
            active.thread.join()
        parent.close()
        child.close()


def test_joined_failure_waits_for_finalize_and_preserves_original_error(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
) -> None:
    run, parent, events, recorders, original = participant
    for command in ("join", "execute", "finalize"):
        parent.send(command)
    with pytest.raises(RuntimeError, match="capture failed") as caught:
        run()
    assert caught.value is original
    assert parent.recv() == ("prepared", 0)
    assert parent.recv() == ("joined", 0)
    assert parent.recv() == ("failure", runner.ParticipantFailure("RuntimeError: capture failed", True))
    assert parent.recv() == ("stopped", 0)
    assert events[events.index("capture failed") + 1 :] == ["finalized"]
    assert all(not recorder.thread.is_alive() for recorder in recorders)


@pytest.mark.parametrize("stage", ("allocator", "estimator"))
def test_prejoin_setup_failure_acquires_no_sampler(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    run, parent, events, recorders, _ = participant

    def fail(**kwargs: object) -> None:
        raise RuntimeError(f"{stage} failed")

    monkeypatch.setattr(
        runner, "ensure_supported_cuda_allocator" if stage == "allocator" else "DeviceMemoryEstimator", fail
    )
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        run()
    assert recorders == []
    assert parent.recv() == ("failure", runner.ParticipantFailure(f"RuntimeError: {stage} failed", False))
    assert "finalized" not in events
    if stage == "allocator":
        assert "warmup" not in events


class OwnerRetained(BaseException):
    pass


def test_parent_retention_logs_operation_and_cleanup_tracebacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def fail_start() -> None:
        raise RuntimeError("UID spawn failed")

    def fail_cleanup(*, deadline: float) -> None:
        raise RuntimeError("MPS cleanup failed")

    def observe_retention(seconds: float) -> None:
        raise OwnerRetained

    process = SimpleNamespace(pid=None, start=fail_start, is_alive=lambda: False, close=lambda: None)
    context = SimpleNamespace(Pipe=multiprocessing.Pipe, Process=lambda **kwargs: process)
    monkeypatch.setattr(runner, "multiprocessing", SimpleNamespace(get_context=lambda method: context))
    monkeypatch.setattr(runner, "MpsScope", lambda endpoint: SimpleNamespace(endpoint=endpoint, stop=fail_cleanup))
    monkeypatch.setattr(
        runner,
        "visible_uuids",
        lambda: tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(2)),
    )
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=time.monotonic, sleep=observe_retention))
    config = synthetic_config().model_copy(update={"cache_root": tmp_path})
    experiments, _ = corpus.calibration_experiments(config)

    with pytest.raises(OwnerRetained):
        runner.run_world(experiments[0], config, repetition=1)

    assert "Traceback" in caplog.text
    assert "UID spawn failed" in caplog.text
    assert "MPS cleanup failed" in caplog.text


@pytest.mark.parametrize("stage", ("join", "finalize"))
def test_uncertain_native_retirement_keeps_owner_without_retry(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    run, parent, events, recorders, _ = participant

    def fail(*args: object) -> None:
        events.append(f"{stage} attempted")
        raise RuntimeError(f"{stage} failed")

    def observe_retention(seconds: float) -> None:
        if threading.current_thread() is threading.main_thread():
            raise OwnerRetained
        time.sleep(seconds)

    if stage == "join":
        monkeypatch.setattr(runner.xpool.native.fabric, "join", fail)
    else:
        monkeypatch.setattr(runner, "drain_and_finalize", fail)
    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(
            monotonic_ns=time.monotonic_ns,
            monotonic=time.monotonic,
            sleep=observe_retention,
        ),
    )
    for command in ("join", "execute", "finalize"):
        parent.send(command)
    with pytest.raises(OwnerRetained):
        run()
    assert events.count(f"{stage} attempted") == 1
    assert "finalized" not in events
    assert all(not recorder.thread.is_alive() for recorder in recorders)
    assert parent.recv() == ("prepared", 0)
    if stage == "finalize":
        assert parent.recv() == ("joined", 0)
        assert parent.recv() == ("failure", runner.ParticipantFailure("RuntimeError: capture failed", True))
    assert parent.recv() == (
        "failure",
        runner.ParticipantFailure(f"{stage} failed" if stage == "finalize" else "RuntimeError: join failed", False),
    )


def test_terminal_failure_does_not_start_device_retirement(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, parent, events, _, original = participant

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(runner, "is_terminal_device_error", lambda error: error is original)
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}, _exit=exit_process))
    for command in ("join", "execute"):
        parent.send(command)
    with pytest.raises(SystemExit) as caught:
        run()
    assert caught.value.code == 1
    assert events[events.index("capture failed") + 1 :] == []


def test_failed_phase_has_no_success_only_synchronization(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
) -> None:
    events = participant[2]
    recorder = runner.PhaseRecorder(device=1, baseline_free_bytes=100)
    with pytest.raises(RuntimeError, match="operation failed"):
        with recorder.phase("test"):
            raise RuntimeError("operation failed")
    recorder.close()
    assert "synchronize" not in events


def test_stopped_sampler_failure_prevents_further_phase_device_calls(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CudaRuntimeError(runtime.cudaError_t.cudaErrorIllegalAddress, "cudaMemGetInfo")

    def memory_info() -> tuple[int, int]:
        if threading.current_thread() is not threading.main_thread():
            raise original
        return 100, 100

    monkeypatch.setattr(runner, "memory_info", memory_info)
    recorder = runner.PhaseRecorder(device=1, baseline_free_bytes=100)
    recorder.begin("active")
    recorder.thread.join(1.0)
    assert not recorder.thread.is_alive()
    for operation in (lambda: recorder.begin("next"), lambda: recorder.finish("active"), recorder.close):
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value.__cause__ is original
        assert is_terminal_device_error(caught.value)
    assert "synchronize" not in participant[2]


def test_terminal_failure_in_a_stopped_sampler_prevents_collective_retirement(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, parent, events, recorders, original = participant
    terminal = CudaRuntimeError(runtime.cudaError_t.cudaErrorIllegalAddress, "cudaMemGetInfo")

    def capture(**kwargs: object) -> None:
        recorder = recorders[0]
        recorder.close()
        recorder.failure = terminal
        raise original

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(runner, "FfnExecutionRegistry", SimpleNamespace(materialize=capture))
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}, _exit=exit_process))
    for command in ("join", "execute"):
        parent.send(command)
    with pytest.raises(SystemExit) as caught:
        run()
    assert caught.value.code == 1
    assert "finalized" not in events
    assert all(not recorder.thread.is_alive() for recorder in recorders)
