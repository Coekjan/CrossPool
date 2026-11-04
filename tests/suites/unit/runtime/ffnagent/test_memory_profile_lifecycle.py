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
from cuda.bindings import runtime  # ty: ignore[unresolved-import]

from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.runtime.ffnagent.device_memory import StartupPoint
from xpool.runtime.ffnagent.memory_profile import corpus, runner
from xpool.utils.cuda import CudaRuntimeError, is_terminal_device_error
from xtest.harness.support.config import (
    TEST_CALIBRATION_ENVIRONMENT,
    TEST_EXECUTION_CONTRACT,
    synthetic_config,
)


@pytest.fixture
def participant(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
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

    def capture(registry: runner.FfnExecutionRegistry, **kwargs: object) -> None:
        events.append("capture failed")
        raise capture_error

    config = synthetic_config()
    experiments, _ = corpus.calibration_experiments(config)
    plan = experiments[0].fabric_plan(
        "a" * FABRIC_UID_HEX_LENGTH, config=config, ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,)
    )
    initial = getattr(request, "param", "plan")
    parent.send(plan if initial == "plan" else initial)
    monkeypatch.setattr(runner.memory_calibration, "observe_environment", lambda device: TEST_CALIBRATION_ENVIRONMENT)
    monkeypatch.setattr(runner, "initialize_child_signals", lambda mask: None)
    monkeypatch.setattr(runner, "init_global_config", lambda **kwargs: synthetic_config())
    monkeypatch.setattr(runner.bootstrap, "init", lambda device, role: None)
    monkeypatch.setattr(runner.capture_workspace, "prepare_execution_sources", lambda: (TEST_EXECUTION_CONTRACT, ()))
    monkeypatch.setattr(runner, "memory_info", lambda: (100, TEST_CALIBRATION_ENVIRONMENT.total_memory_bytes))
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
            cuda=SimpleNamespace(
                synchronize=lambda *args: events.append("synchronize"),
                empty_cache=lambda: None,
                get_device_properties=lambda device: SimpleNamespace(uuid="GPU-test"),
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "DeviceMemoryEstimator",
        lambda **kwargs: SimpleNamespace(allocation_ledger_for_model_plans=lambda **kw: ()),
    )
    monkeypatch.setattr(runner, "PhaseRecorder", recorder)
    monkeypatch.setattr(runner, "materialize_calibration_weights", lambda **kwargs: ())
    monkeypatch.setattr(runner, "project_fabric_arena", lambda plan: None)
    monkeypatch.setattr(runner.xpool.native.fabric, "join", lambda arena, pe: events.append("joined"))
    monkeypatch.setattr(runner.FfnExecutionRegistry, "materialize", capture)
    monkeypatch.setattr(runner.FfnExecutionRegistry, "close", lambda self: events.append("released"))
    monkeypatch.setattr(runner, "drain_and_finalize", lambda: events.append("finalized"))

    def run() -> None:
        runner.participant_child(
            child,
            config_path=Path("config.toml"),
            config_environment={},
            model_specs=(),
            placement=plan.pe_placements[1],
            pe=1,
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
    assert parent.recv()[0] == "facts"
    assert parent.recv() == ("prepared", 1)
    assert parent.recv() == ("joined", 1)
    assert parent.recv() == ("failure", runner.ParticipantFailure("RuntimeError: capture failed", True))
    assert parent.recv() == ("stopped", 1)
    assert events[events.index("capture failed") + 1 :] == ["finalized", "released"]
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

    if stage == "allocator":
        monkeypatch.setattr(runner.capture_workspace, "prepare_execution_sources", fail)
    else:
        monkeypatch.setattr(runner, "DeviceMemoryEstimator", fail)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        run()
    assert recorders == []
    if stage == "estimator":
        assert parent.recv()[0] == "facts"
    assert parent.recv() == ("failure", runner.ParticipantFailure(f"RuntimeError: {stage} failed", False))
    assert "finalized" not in events


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
    assert parent.recv()[0] == "facts"
    assert parent.recv() == ("prepared", 1)
    if stage == "finalize":
        assert parent.recv() == ("joined", 1)
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
        with recorder.phase("graph_capture"):
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
    recorder.begin("graph_capture")
    recorder.thread.join(1.0)
    assert not recorder.thread.is_alive()
    for operation in (
        lambda: recorder.begin("execution_installation"),
        lambda: recorder.finish("graph_capture"),
        recorder.close,
    ):
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

    def capture(registry: runner.FfnExecutionRegistry, **kwargs: object) -> None:
        recorder = recorders[0]
        recorder.close()
        recorder.failure = terminal
        raise original

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(runner.FfnExecutionRegistry, "materialize", capture)
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}, _exit=exit_process))
    for command in ("join", "execute"):
        parent.send(command)
    with pytest.raises(SystemExit) as caught:
        run()
    assert caught.value.code == 1
    assert "finalized" not in events
    assert all(not recorder.thread.is_alive() for recorder in recorders)


@pytest.mark.parametrize("participant", ("stop", "invalid"), indirect=True)
def test_facts_wait_stops_or_rejects_invalid_plan_before_weights(
    participant: tuple[Callable[[], None], Connection, list[str], list[runner.PhaseRecorder], RuntimeError],
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    run, parent, events, recorders, _ = participant
    monkeypatch.setattr(
        runner,
        "materialize_calibration_weights",
        lambda **kwargs: pytest.fail("weights allocated before Plan acceptance"),
    )
    initial = request.node.callspec.params["participant"]
    if initial == "stop":
        run()
        assert parent.recv()[0] == "facts"
        assert parent.recv() == ("stopped", 1)
    else:
        with pytest.raises(RuntimeError, match=r"complete Fabric Plan|differs from local"):
            run()
        assert parent.recv()[0] == "facts"
        tag, failure = parent.recv()
        assert tag == "failure"
        assert not failure.ready_to_finalize
    assert not recorders
    assert "joined" not in events


def test_first_phase_failure_restores_native_installation() -> None:
    original_install = runner.xpool.native.ffnagent.install_execution
    original_error = RuntimeError("sampler failed before capture")

    def fail_begin(point: StartupPoint) -> None:
        raise original_error

    recorder = cast(runner.PhaseRecorder, SimpleNamespace(begin=fail_begin))
    with pytest.raises(RuntimeError) as caught, runner.record_registry_phases(recorder):
        pytest.fail("capture began after sampler failure")
    assert caught.value is original_error
    assert runner.xpool.native.ffnagent.install_execution is original_install
