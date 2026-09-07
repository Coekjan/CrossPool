from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import pytest

from tests.harness.support.wait import wait_until, wait_until_raise
from xpool.runtime.agent import Agent, AgentError, AgentHeartbeat
from xpool.service.client import XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import HeartbeatResponse


class HeartbeatAgent:
    """Minimal agent boundary consumed by the heartbeat worker."""

    cuda_device = 0

    def __init__(self, sender: Callable[[], HeartbeatResponse]) -> None:
        self.sender = sender

    def send_heartbeat(self) -> HeartbeatResponse:
        return self.sender()

    def handle_heartbeat_response(self, response: HeartbeatResponse) -> None:
        pass


def heartbeat_response() -> HeartbeatResponse:
    return HeartbeatResponse(
        warnings=[],
        generation=None,
        fabric_phase=None,
    )


def test_heartbeat_reports_missing_registration() -> None:
    events: list[tuple[object, ...]] = []

    def sender() -> HeartbeatResponse:
        events.append(("heartbeat", 0))
        raise XpoolDaemonError("not_ready", "registration missing")

    worker = AgentHeartbeat(agent=cast(Agent, HeartbeatAgent(sender)))
    worker.start()
    assert wait_until(worker.consume_registration_missing)
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())
    worker.close()

    assert events == [("heartbeat", 0)]


def test_heartbeat_surfaces_unrecoverable_daemon_error() -> None:
    def sender() -> HeartbeatResponse:
        raise XpoolDaemonError("conflict", "pid mismatch")

    worker = AgentHeartbeat(agent=cast(Agent, HeartbeatAgent(sender)))
    worker.start()
    with pytest.raises(AgentError, match="unrecoverable daemon error"):
        wait_until_raise(worker.raise_if_failed)
    worker.close()


def test_heartbeat_retries_recoverable_client_errors(caplog: pytest.LogCaptureFixture) -> None:
    heartbeat_count = 0

    def sender() -> HeartbeatResponse:
        nonlocal heartbeat_count
        heartbeat_count += 1
        if heartbeat_count == 1:
            raise XpoolClientError("transport", "daemon unavailable")
        return heartbeat_response()

    worker = AgentHeartbeat(
        agent=cast(Agent, HeartbeatAgent(sender)),
        interval_s=0.01,
    )
    with caplog.at_level(logging.DEBUG, logger="xpool.runtime.agent"):
        worker.start()
        assert wait_until(lambda: heartbeat_count >= 2)
    worker.close()
    worker.raise_if_failed()

    assert [record.levelno for record in caplog.records] == [logging.DEBUG]
