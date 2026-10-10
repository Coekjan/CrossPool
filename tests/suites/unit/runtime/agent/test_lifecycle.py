"""Agent main-loop lifecycle behavior."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch

import xpool.runtime.agent
from xpool.fabric import FabricGenerationId, FabricGenerationPhase, FabricParticipantPhase, FabricPlan
from xpool.native import RuntimeRole
from xpool.runtime.agent import Agent, AgentError
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.wire import FabricParticipantReport, FabricQuiesceRequest, HeartbeatResponse
from xtest.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xtest.harness.support.runtime.atnagent import reset_agent_runtime

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_agent_runtime.__name__)

ShutdownPoint = Literal["prepare", "bootstrap", "advance"]
SignalHandler = Callable[[int, object], object]


class RunLoopClient:
    """Record terminal client ownership for one Agent run-loop test."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        """Record client closure."""

        self.events.append("client:close")

    def request_fabric_quiesce(self, request: FabricQuiesceRequest) -> None:
        self.events.append("fabric:quiesce")

    def report_fabric_participant(self, report: FabricParticipantReport) -> None:
        self.events.append(report.phase.value)


class RunLoopHeartbeat:
    """Minimal deterministic heartbeat boundary for Agent.run()."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self) -> None:
        """Record heartbeat startup."""

        self.events.append("heartbeat:start")

    def stop(self) -> None:
        """Record heartbeat stop when requested by production control flow."""

        self.events.append("heartbeat:stop")

    def close(self) -> None:
        """Record heartbeat ownership release."""

        self.events.append("heartbeat:close")

    def raise_if_failed(self) -> None:
        """Expose no background failure."""

    def consume_registration_missing(self) -> bool:
        """Report that the current registration remains live."""

        return False

    def consume_response(self) -> HeartbeatResponse | None:
        """Return no newer daemon snapshot."""

        return None


class RunLoopAgent(Agent):
    """Record Agent.run() operation boundaries around one shutdown request."""

    def __init__(
        self,
        *,
        events: list[str],
        request_shutdown: Callable[[], None],
        shutdown_point: ShutdownPoint,
    ) -> None:
        install_test_config(synthetic_config())
        super().__init__(device=0, runtime_role=RuntimeRole.ATNAGENT)
        self.events = events
        self.request_shutdown = request_shutdown
        self.shutdown_point = shutdown_point
        self.client = cast(XpoolClient, RunLoopClient(events))
        self.heartbeat_worker = cast(xpool.runtime.agent.AgentHeartbeat, RunLoopHeartbeat(events))

    def register(self) -> None:
        """Complete one registration acknowledgement."""

        self.events.append("register")
        self.registered = True

    def send_heartbeat(self) -> HeartbeatResponse:
        """Return an empty heartbeat snapshot when called unexpectedly."""

        return HeartbeatResponse(warnings=[], generation=None, fabric_phase=None)

    def prepare_fabric_join(self) -> bool:
        """Complete join preparation and optionally request shutdown within it."""

        self.events.append("prepare:start")
        if self.shutdown_point == "prepare":
            self.request_shutdown()
        self.events.append("prepare:end")
        return True

    def prepare_fabric_execution(self) -> None:
        """Model the indivisible post-join execution preparation transaction."""

        self.events.append("bootstrap:start")
        if self.shutdown_point == "bootstrap":
            self.request_shutdown()
        self.events.append("bootstrap:active")

    def activate_fabric(self) -> None:
        """Satisfy the abstract Agent role contract."""

    def quiesce_fabric(self) -> None:
        """Satisfy the abstract Agent role contract."""

    def poll_fabric_health(self) -> None:
        """Record one completed health boundary."""

        self.events.append("health")

    def advance_fabric_lifecycle(self) -> None:
        """Complete one action/report boundary and optionally request shutdown."""

        if self.shutdown_point == "prepare":
            self.prepare_fabric_join()
        elif self.shutdown_point == "bootstrap":
            self.prepare_fabric_execution()
        else:
            self.events.append("advance:start")
            self.request_shutdown()
        self.events.append("advance:reported")

    def shutdown_fabric(self) -> None:
        """Record entry into coordinated shutdown."""

        self.events.append("shutdown")

    def close_role(self) -> None:
        """Record role-resource closure."""

        self.events.append("role:close")


@pytest.mark.parametrize(
    ("shutdown_point", "expected"),
    [
        pytest.param(
            "prepare",
            [
                "prepare:start",
                "signal",
                "prepare:end",
                "advance:reported",
            ],
            id="after-preparation-before-plan",
        ),
        pytest.param(
            "bootstrap",
            [
                "bootstrap:start",
                "signal",
                "bootstrap:active",
                "advance:reported",
            ],
            id="after-bootstrap-active",
        ),
        pytest.param(
            "advance",
            [
                "advance:start",
                "signal",
                "advance:reported",
            ],
            id="after-action-report",
        ),
    ],
)
def test_agent_run_observes_shutdown_only_at_safe_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    shutdown_point: ShutdownPoint,
    expected: list[str],
) -> None:
    """A signal request never splits preparation, bootstrap, or action/report."""

    events: list[str] = []
    handlers: dict[int, SignalHandler] = {}
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: events.append("device:complete"))

    @contextmanager
    def capture_handler(
        signal_number: signal.Signals | int,
        handler: signal.Handlers | SignalHandler,
    ) -> Generator[None, None, None]:
        assert callable(handler)
        handlers[int(signal_number)] = handler
        try:
            yield
        finally:
            handlers.pop(int(signal_number))

    def request_shutdown() -> None:
        events.append("signal")
        assert set(handlers) == {int(signal.SIGINT), int(signal.SIGTERM)}
        handlers[int(signal.SIGTERM)](int(signal.SIGTERM), None)
        handlers[int(signal.SIGINT)](int(signal.SIGINT), None)

    monkeypatch.setattr(xpool.runtime.agent, "sighandle", capture_handler)
    agent = RunLoopAgent(
        events=events,
        request_shutdown=request_shutdown,
        shutdown_point=shutdown_point,
    )

    agent.run()

    assert events == [
        "register",
        "heartbeat:start",
        *expected,
        "shutdown",
        "heartbeat:close",
        "role:close",
        "device:complete",
        "client:close",
    ]


@pytest.mark.parametrize("participant_phase", [None, FabricParticipantPhase.JOIN_READY])
def test_agent_run_closes_local_owners_when_pre_join_work_fails(
    monkeypatch: pytest.MonkeyPatch,
    participant_phase: FabricParticipantPhase | None,
) -> None:
    """A failure before native join releases local owners, including JOIN_READY."""

    events: list[str] = []
    failure = RuntimeError("failed")
    agent = RunLoopAgent(
        events=events,
        request_shutdown=lambda: None,
        shutdown_point="advance",
    )
    if participant_phase is not None:
        agent.participant_report = FabricParticipantReport(
            owner=agent.process_ref,
            generation=FabricGenerationId.create(),
            pe=0,
            phase=participant_phase,
        )

    def fail_lifecycle() -> None:
        raise failure

    monkeypatch.setattr(agent, "advance_fabric_lifecycle", fail_lifecycle)
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: events.append("device:complete"))

    with pytest.raises(RuntimeError, match="failed") as error:
        agent.run()

    assert error.value is failure
    assert events == ["register", "heartbeat:start", "heartbeat:close", "role:close", "device:complete", "client:close"]


@pytest.mark.parametrize("participant_phase", [None, FabricParticipantPhase.JOIN_READY])
def test_agent_run_continues_pre_join_cleanup_after_owner_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    participant_phase: FabricParticipantPhase | None,
) -> None:
    """One failed local close does not strand later pre-join owners."""

    events: list[str] = []
    failure = RuntimeError("failed")
    agent = RunLoopAgent(
        events=events,
        request_shutdown=lambda: None,
        shutdown_point="advance",
    )
    if participant_phase is not None:
        agent.participant_report = FabricParticipantReport(
            owner=agent.process_ref,
            generation=FabricGenerationId.create(),
            pe=0,
            phase=participant_phase,
        )

    def fail_lifecycle() -> None:
        raise failure

    def fail_heartbeat_close() -> None:
        events.append("heartbeat:close")
        raise RuntimeError("heartbeat close failed")

    class Retained(Exception):
        pass

    def retain(seconds: float) -> None:
        raise Retained

    monkeypatch.setattr(agent, "advance_fabric_lifecycle", fail_lifecycle)
    monkeypatch.setattr(agent.heartbeat_worker, "close", fail_heartbeat_close)
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: events.append("device:complete"))
    monkeypatch.setattr(xpool.runtime.agent, "time", SimpleNamespace(sleep=retain))

    with pytest.raises(Retained):
        agent.run()

    assert agent.failure is failure
    assert events == ["register", "heartbeat:start", "heartbeat:close", "role:close", "device:complete", "client:close"]


@pytest.mark.parametrize("participant_phase", [None, FabricParticipantPhase.JOIN_READY])
def test_unjoined_agent_shutdown_releases_local_owners(
    monkeypatch: pytest.MonkeyPatch,
    participant_phase: FabricParticipantPhase | None,
) -> None:
    events: list[str] = []
    failure = AgentError("generation aborted before join")
    agent = RunLoopAgent(events=events, request_shutdown=lambda: None, shutdown_point="advance")
    generation = FabricGenerationId.create()
    # Shutdown consumes only generation identity at this retained-plan seam.
    agent.fabric_plan = cast(FabricPlan, SimpleNamespace(generation=generation))
    if participant_phase is not None:
        agent.participant_report = FabricParticipantReport(
            owner=agent.process_ref,
            generation=generation,
            pe=0,
            phase=participant_phase,
        )

    def fail_control(shutdown_requested: threading.Event) -> None:
        agent.fail(failure)

    monkeypatch.setattr(agent, "run_control_loop", fail_control)
    monkeypatch.setattr(agent, "shutdown_fabric", lambda: Agent.shutdown_fabric(agent))
    monkeypatch.setattr(agent, "advance_fabric_lifecycle", lambda: pytest.fail("unjoined participant must not drain"))
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: events.append("device:complete"))

    with pytest.raises(AgentError) as error:
        agent.run()

    assert error.value is failure
    assert events == ["fabric:quiesce", "heartbeat:close", "role:close", "device:complete", "client:close"]


@pytest.mark.parametrize("quiesce_committed", [False, True])
def test_agent_shutdown_reconciles_control_outage_before_native_retirement(
    monkeypatch: pytest.MonkeyPatch,
    quiesce_committed: bool,
) -> None:
    events: list[str] = []
    failure = RuntimeError("original control failure")
    agent = RunLoopAgent(events=events, request_shutdown=lambda: None, shutdown_point="advance")
    generation = FabricGenerationId.create()
    agent.participant_report = FabricParticipantReport(
        owner=agent.process_ref,
        generation=generation,
        pe=0,
        phase=FabricParticipantPhase.JOINED,
    )
    # Only generation identity is consumed; placement is substituted at its owner.
    agent.fabric_plan = cast(FabricPlan, SimpleNamespace(generation=generation))
    agent.fabric_phase = FabricGenerationPhase.PREPARING_EXECUTION
    attempts = 0
    snapshots = 0

    def fail_control(shutdown_requested: threading.Event) -> None:
        raise failure

    def request_quiesce(request: FabricQuiesceRequest) -> None:
        nonlocal attempts
        attempts += 1
        events.append("fabric:quiesce")
        if attempts <= 2:
            raise XpoolClientError("transport", "quiesce response unavailable")

    def heartbeat_response() -> HeartbeatResponse | None:
        nonlocal snapshots
        snapshots += 1
        report = agent.participant_report
        assert report is not None
        if not quiesce_committed and snapshots <= 3:
            assert "role:quiesce" not in events
            if snapshots != 1:
                return None
            phase = FabricGenerationPhase.PREPARING_EXECUTION
        else:
            phase = (
                FabricGenerationPhase.STOPPED
                if report.phase is FabricParticipantPhase.DRAINED
                else FabricGenerationPhase.ABORTING
            )
        return HeartbeatResponse(warnings=[], generation=generation, fabric_phase=phase)

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(agent, "run_control_loop", fail_control)
    monkeypatch.setattr(agent, "shutdown_fabric", lambda: Agent.shutdown_fabric(agent))
    monkeypatch.setattr(agent, "advance_fabric_lifecycle", lambda: Agent.advance_fabric_lifecycle(agent))
    monkeypatch.setattr(agent, "fabric_pe", lambda: 0)
    monkeypatch.setattr(agent.client, "request_fabric_quiesce", request_quiesce)
    monkeypatch.setattr(agent.heartbeat_worker, "consume_response", heartbeat_response)
    monkeypatch.setattr(
        agent, "prepare_fabric_execution", lambda: pytest.fail("stale phase must not prepare execution")
    )
    monkeypatch.setattr(agent, "quiesce_fabric", lambda: events.append("role:quiesce"))
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "drain_async", lambda: events.append("fabric:drain"))
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "drain_pending", lambda: False)
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "finalize", lambda: pytest.fail("unsafe finalization"))
    monkeypatch.setattr(xpool.runtime.agent.os, "_exit", exit_process)
    monkeypatch.setattr(xpool.runtime.agent, "time", SimpleNamespace(sleep=lambda _: None))

    with pytest.raises(SystemExit) as error:
        agent.run()

    assert error.value.code == 1
    assert agent.failure is failure
    assert events == [
        "joined",
        *(["fabric:quiesce"] if quiesce_committed else ["fabric:quiesce"] * 3),
        "role:quiesce",
        "quiesced",
        "fabric:drain",
        "draining",
        "drained",
    ]


@pytest.mark.parametrize("quiesce_fails", [False, True])
def test_failed_agent_reports_drain_and_waits_for_authoritative_stop(
    monkeypatch: pytest.MonkeyPatch,
    quiesce_fails: bool,
) -> None:
    events: list[str] = []
    failure = RuntimeError("control failure")
    agent = RunLoopAgent(events=events, request_shutdown=lambda: None, shutdown_point="advance")
    agent.participant_report = FabricParticipantReport(
        owner=agent.process_ref,
        generation=FabricGenerationId.create(),
        pe=0,
        phase=FabricParticipantPhase.ACTIVE,
    )
    # Only generation identity is consumed; placement is substituted at its owner.
    agent.fabric_plan = cast(FabricPlan, SimpleNamespace(generation=agent.participant_report.generation))
    agent.fabric_phase = FabricGenerationPhase.ABORTING

    def fail_control(shutdown_requested: threading.Event) -> None:
        raise failure

    class Retained(Exception):
        pass

    def retain(seconds: float) -> None:
        raise Retained

    def quiesce() -> None:
        events.append("role:quiesce")
        if quiesce_fails:
            raise RuntimeError("consumers remain attached")

    def drain_pending() -> bool:
        events.append("fabric:complete")
        return False

    def heartbeat_response() -> HeartbeatResponse:
        report = agent.participant_report
        assert report is not None
        return HeartbeatResponse(
            warnings=[],
            generation=report.generation,
            fabric_phase=(
                FabricGenerationPhase.STOPPED
                if report.phase is FabricParticipantPhase.DRAINED
                else FabricGenerationPhase.ABORTING
            ),
        )

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(agent, "run_control_loop", fail_control)
    monkeypatch.setattr(agent, "shutdown_fabric", lambda: Agent.shutdown_fabric(agent))
    monkeypatch.setattr(agent, "advance_fabric_lifecycle", lambda: Agent.advance_fabric_lifecycle(agent))
    monkeypatch.setattr(agent, "fabric_pe", lambda: 0)
    monkeypatch.setattr(agent.heartbeat_worker, "consume_response", heartbeat_response)
    monkeypatch.setattr(agent, "quiesce_fabric", quiesce)
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "drain_async", lambda: events.append("fabric:drain"))
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "drain_pending", drain_pending)
    monkeypatch.setattr(xpool.runtime.agent.xpool.native.fabric, "finalize", lambda: pytest.fail("unsafe finalization"))
    monkeypatch.setattr(xpool.runtime.agent.os, "_exit", exit_process)
    monkeypatch.setattr(xpool.runtime.agent, "time", SimpleNamespace(sleep=retain if quiesce_fails else lambda _: None))
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: pytest.fail("unsafe device wait"))
    if quiesce_fails:
        with pytest.raises(Retained):
            agent.run()
        assert events == ["active", "fabric:quiesce", "role:quiesce"]
    else:
        with pytest.raises(SystemExit) as exit_result:
            agent.run()
        assert exit_result.value.code == 1
        assert events == [
            "active",
            "fabric:quiesce",
            "role:quiesce",
            "quiesced",
            "fabric:drain",
            "draining",
            "fabric:complete",
            "drained",
        ]
    assert agent.failure is failure


def test_agent_preserves_background_failure_after_confirmed_retirement(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    failure = RuntimeError("background failure")
    agent = RunLoopAgent(events=events, request_shutdown=lambda: None, shutdown_point="advance")

    def fail_control(shutdown_requested: threading.Event) -> None:
        agent.fail(failure)

    monkeypatch.setattr(agent, "run_control_loop", fail_control)
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: events.append("device:complete"))

    with pytest.raises(RuntimeError, match="background failure") as error:
        agent.run()

    assert error.value is failure
    assert events == ["shutdown", "heartbeat:close", "role:close", "device:complete", "client:close"]


def test_agent_failure_callback_preserves_first_failure_and_obeys_later_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = RunLoopAgent(events=[], request_shutdown=lambda: None, shutdown_point="advance")
    failure = RuntimeError("heartbeat failed")
    terminal = torch.AcceleratorError("terminal device result")
    setattr(terminal, "error_code", 719)

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(xpool.runtime.agent.os, "_exit", exit_process)
    monkeypatch.setattr(xpool.runtime.agent.torch.cuda, "synchronize", lambda device: pytest.fail("unsafe device wait"))
    agent.fail(failure)
    assert agent.shutdown_requested.is_set()
    with pytest.raises(SystemExit) as error:
        agent.fail(terminal)
    assert error.value.code == 1
    assert agent.failure is failure


def test_agent_reregisters_missing_registration_before_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    shutdown_requested = threading.Event()
    agent = RunLoopAgent(
        events=events,
        request_shutdown=lambda: None,
        shutdown_point="advance",
    )
    agent.registered = True
    monkeypatch.setattr(agent.heartbeat_worker, "consume_registration_missing", lambda: True)

    def register() -> None:
        events.append("register")
        agent.registered = True
        shutdown_requested.set()

    monkeypatch.setattr(agent, "register", register)

    agent.run_control_loop(shutdown_requested)

    assert events == ["heartbeat:stop", "register", "heartbeat:start"]


def test_agent_fails_when_registration_disappears_after_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    agent = RunLoopAgent(
        events=events,
        request_shutdown=lambda: None,
        shutdown_point="advance",
    )
    agent.registered = True
    agent.fabric_plan = cast(FabricPlan, object())
    monkeypatch.setattr(agent.heartbeat_worker, "consume_registration_missing", lambda: True)

    with pytest.raises(AgentError):
        agent.run_control_loop(threading.Event())

    assert events == ["heartbeat:stop"]
