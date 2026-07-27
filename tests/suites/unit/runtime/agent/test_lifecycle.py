"""Agent main-loop lifecycle behavior."""

from __future__ import annotations

import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Literal, cast

import pytest

import xpool.runtime.agent
from tests.harness.runtime.atnagent import reset_agent_runtime
from xpool.runtime import RuntimeRole
from xpool.runtime.agent import Agent
from xpool.service.client import XpoolClient
from xpool.service.wire import HeartbeatResponse

pytestmark = pytest.mark.usefixtures(reset_agent_runtime.__name__)

ShutdownPoint = Literal["prepare", "bootstrap", "advance"]
SignalHandler = Callable[[int, object], object]


class RunLoopClient:
    """Record terminal client ownership for one Agent run-loop test."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        """Record client closure."""

        self.events.append("client:close")


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
        super().__init__(cuda_device=0, runtime_role=RuntimeRole.ATNAGENT)
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

    def prepare_fabric(self) -> bool:
        """Complete role preparation and optionally request shutdown within it."""

        self.events.append("prepare:start")
        if self.shutdown_point == "prepare":
            self.request_shutdown()
        self.events.append("prepare:end")
        return True

    def join_fabric(self) -> None:
        """Model the indivisible Fabric bootstrap transaction."""

        self.events.append("bootstrap:start")
        if self.shutdown_point == "bootstrap":
            self.request_shutdown()
        self.events.append("bootstrap:active")

    def activate_fabric_plan(self) -> None:
        """Satisfy the abstract Agent role contract."""

    def quiesce_fabric(self) -> None:
        """Satisfy the abstract Agent role contract."""

    def poll_fabric_health(self) -> None:
        """Record one completed health boundary."""

        self.events.append("health")

    def advance_fabric_lifecycle(self) -> None:
        """Complete one action/report boundary and optionally request shutdown."""

        self.events.append("advance:start")
        if self.shutdown_point == "advance":
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
                "register",
                "heartbeat:start",
                "prepare:start",
                "signal",
                "prepare:end",
                "shutdown",
                "heartbeat:close",
                "role:close",
                "client:close",
            ],
            id="after-preparation-before-plan",
        ),
        pytest.param(
            "bootstrap",
            [
                "register",
                "heartbeat:start",
                "prepare:start",
                "prepare:end",
                "bootstrap:start",
                "signal",
                "bootstrap:active",
                "shutdown",
                "heartbeat:close",
                "role:close",
                "client:close",
            ],
            id="after-bootstrap-active",
        ),
        pytest.param(
            "advance",
            [
                "register",
                "heartbeat:start",
                "prepare:start",
                "prepare:end",
                "bootstrap:start",
                "bootstrap:active",
                "health",
                "advance:start",
                "signal",
                "advance:reported",
                "shutdown",
                "heartbeat:close",
                "role:close",
                "client:close",
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

    assert events == expected
