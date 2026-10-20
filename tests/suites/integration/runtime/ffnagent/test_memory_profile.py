from __future__ import annotations

import functools
import multiprocessing
import os
import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import FABRIC_UID_HEX_LENGTH, FabricPlan, FabricRole
from xpool.runtime.ffnagent.memory_profile import corpus, runner
from xpool.utils.mps import MpsEndpoint
from xpool.utils.sighandler import block_signals, initialize_child_signals
from xtest.harness.support.config import synthetic_config


def uid_peer(
    connection: Connection,
    config_path: Path,
    *,
    config_environment: dict[str, str],
    signal_mask: set[int],
) -> None:
    initialize_child_signals(signal_mask)
    connection.send(("uid", "0" * FABRIC_UID_HEX_LENGTH))
    runner.require_command(connection, "stop")
    (config_path.parent.parent / "retired-uid").write_text(str(os.getpid()))
    connection.send(("stopped", None))
    connection.close()


def participant_peer(
    connection: Connection,
    *,
    config_path: Path,
    config_environment: dict[str, str],
    model_specs: tuple[ffn.FfnModelSpec, ...],
    fabric_plan: FabricPlan,
    pe: int,
    environment: dict[str, str],
    signal_mask: set[int],
    failure_stage: str,
) -> None:
    initialize_child_signals(signal_mask)
    failed = fabric_plan.pe_placements[pe].role is FabricRole.FFNAGENT and failure_stage != "cancel"
    try:
        if failed and failure_stage == "prepare":
            connection.send(("failure", runner.ParticipantFailure("preparation failed", False)))
            raise SystemExit(1)
        if failure_stage == "cancel":
            # Withhold readiness briefly to observe whether stop is queued
            # independently of preparation, then let either implementation retire.
            (config_path.parent.parent / f"stop-before-prepared-pe{pe}").write_text(str(connection.poll(1.0)))
        connection.send(("prepared", pe))
        command = connection.recv()
        if command == "join":
            connection.send(("joined", pe))
            if failed:
                runner.require_command(connection, "execute")
                connection.send(("failure", runner.ParticipantFailure("execution failed", True)))
            runner.require_command(connection, "finalize")
        else:
            assert command == "stop"
        connection.send(("stopped", pe))
    finally:
        (config_path.parent.parent / f"retired-pe{pe}").write_text(str(os.getpid()))
        connection.close()
    if failed:
        raise SystemExit(1)


@pytest.fixture
def host_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[XpoolConfig, list[str], list[tuple[str, bool]]]:
    # Real Host spawn and pipe exchange exercise the owner protocol without
    # injecting a failure into a device collective.
    events: list[str] = []
    retired_at_mps_stop: list[tuple[str, bool]] = []

    class HostScope:
        def __init__(self, endpoint: MpsEndpoint) -> None:
            self.endpoint = endpoint

        def start(self) -> None:
            events.append("mps started")

        def stop(self, *, deadline: float) -> None:
            events.append("mps stopped")
            retired_at_mps_stop.extend(
                (path.name, not psutil.pid_exists(int(path.read_text())))
                for path in sorted((tmp_path / "memory-profile").glob("retired-*"))
            )

    monkeypatch.setattr(runner, "MpsScope", HostScope)
    monkeypatch.setattr(
        runner,
        "visible_uuids",
        lambda: tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(2)),
    )
    monkeypatch.setattr(runner, "uid_child", uid_peer)
    config = synthetic_config().model_copy(update={"cache_root": tmp_path})
    return config, events, retired_at_mps_stop


@pytest.mark.parametrize("failure_stage", ("prepare", "execute"))
def test_failed_world_reaps_participants_and_uid_before_releasing_mps(
    host_world: tuple[XpoolConfig, list[str], list[tuple[str, bool]]],
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    config, events, retired_at_mps_stop = host_world
    monkeypatch.setattr(runner, "participant_child", functools.partial(participant_peer, failure_stage=failure_stage))
    experiments, _ = corpus.calibration_experiments(config)

    with pytest.raises(RuntimeError, match="preparation failed" if failure_stage == "prepare" else "execution failed"):
        runner.run_world(experiments[0], config, repetition=1)

    assert events == ["mps started", "mps stopped"]
    assert retired_at_mps_stop == [("retired-pe0", True), ("retired-pe1", True), ("retired-uid", True)]


@pytest.mark.parametrize("cancel_after", (0, 1, 2), ids=("uid", "first-participant", "last-participant"))
def test_cancellation_stops_admission_and_queues_prejoin_retirement(
    host_world: tuple[XpoolConfig, list[str], list[tuple[str, bool]]],
    monkeypatch: pytest.MonkeyPatch,
    cancel_after: int,
) -> None:
    config, events, retired_at_mps_stop = host_world
    context = multiprocessing.get_context("spawn")
    process_type = context.Process
    started: list[BaseProcess] = []
    pids: list[int] = []

    def process(
        *,
        target: Callable[..., None],
        kwargs: dict[str, object],
        name: str,
        args: tuple[object, ...] = (),
    ) -> BaseProcess:
        child = process_type(target=target, args=args, kwargs=kwargs, name=name)
        started.append(child)
        return child

    @contextmanager
    def cancel_after_spawn(signals: set[signal.Signals]) -> Generator[set[int], None, None]:
        with block_signals(signals) as mask:
            yield mask
            pid = started[-1].pid
            assert pid is not None
            pids.append(pid)
            if len(started) == cancel_after + 1:
                # Deliver cancellation after the protected spawn has published
                # its owner, before another participant can be admitted.
                signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(context, "Process", process)
    monkeypatch.setattr(runner, "multiprocessing", SimpleNamespace(get_context=lambda method: context))
    monkeypatch.setattr(runner, "block_signals", cancel_after_spawn)
    monkeypatch.setattr(runner, "participant_child", functools.partial(participant_peer, failure_stage="cancel"))
    experiments, _ = corpus.calibration_experiments(config)

    with pytest.raises(KeyboardInterrupt, match="cancelled by signal"):
        runner.run_world(experiments[0], config, repetition=1)

    assert len(started) == cancel_after + 1
    assert all(not psutil.pid_exists(pid) for pid in pids)
    if cancel_after == 0:
        assert events == ["mps stopped"]
        assert retired_at_mps_stop == []
    else:
        assert events == ["mps started", "mps stopped"]
        assert retired_at_mps_stop == [
            *((f"retired-pe{pe}", True) for pe in range(cancel_after)),
            ("retired-uid", True),
        ]
        for pe in range(cancel_after):
            assert (config.cache_root / "memory-profile" / f"stop-before-prepared-pe{pe}").read_text() == "True"
