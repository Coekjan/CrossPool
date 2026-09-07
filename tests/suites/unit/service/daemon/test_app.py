from __future__ import annotations

import asyncio
import logging
import threading
from http import HTTPStatus

import httpx
import pytest
from pydantic import ValidationError

import xpool.service.daemon.app
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xpool.fabric import FabricGenerationId
from xpool.native import RuntimeRole
from xpool.service.daemon.control import ServingHealthTargets
from xpool.service.wire import ServingListener

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_daemon_failure_retains_first_exception() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    failure = xpool.service.daemon.app.DaemonFailure()

    failure.record(first)
    failure.record(second)

    assert failure.failed
    assert failure.exception is first


@pytest.mark.parametrize(
    ("host", "port"),
    [
        pytest.param("", 30000, id="empty-host"),
        pytest.param("127.0.0.1", 0, id="zero-port"),
        pytest.param("127.0.0.1", 65536, id="port-too-large"),
    ],
)
def test_serving_listener_rejects_invalid_address(host: str, port: int) -> None:
    with pytest.raises(ValidationError):
        ServingListener(host=host, port=port)


def test_serving_listener_is_immutable() -> None:
    listener = ServingListener(host="127.0.0.1", port=30000)

    with pytest.raises(ValidationError, match="frozen"):
        listener.port = 30001


@pytest.mark.parametrize(
    ("host", "expected_host"),
    [
        pytest.param("127.0.0.1", "127.0.0.1", id="ipv4"),
        pytest.param("localhost", "localhost", id="hostname"),
        pytest.param("::1", "::1", id="ipv6"),
        pytest.param("0.0.0.0", "127.0.0.1", id="wildcard-ipv4"),
        pytest.param("::", "::1", id="wildcard-ipv6"),
    ],
)
def test_probe_serving_listener_uses_health_endpoint(host: str, expected_host: str) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(HTTPStatus.NO_CONTENT)

    async def probe() -> bool:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await xpool.service.daemon.app.probe_serving_listener(
                client,
                ServingListener(host=host, port=30000),
            )

    assert asyncio.run(probe())
    assert len(requests) == 1
    assert requests[0].url.host == expected_host
    assert requests[0].url.port == 30000
    assert requests[0].url.path == "/health"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(HTTPStatus.OK, True, id="ok"),
        pytest.param(HTTPStatus.NO_CONTENT, True, id="no-content"),
        pytest.param(HTTPStatus.SERVICE_UNAVAILABLE, False, id="unavailable"),
    ],
)
def test_probe_serving_listener_accepts_only_success_status(status: HTTPStatus, expected: bool) -> None:
    async def probe() -> bool:
        transport = httpx.MockTransport(lambda request: httpx.Response(status, request=request))
        async with httpx.AsyncClient(transport=transport) as client:
            return await xpool.service.daemon.app.probe_serving_listener(
                client,
                ServingListener(host="127.0.0.1", port=30000),
            )

    assert asyncio.run(probe()) is expected


def test_probe_serving_listener_treats_request_error_as_not_ready() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("not listening", request=request)

    async def probe() -> bool:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            return await xpool.service.daemon.app.probe_serving_listener(
                client,
                ServingListener(host="127.0.0.1", port=30000),
            )

    assert not asyncio.run(probe())


def test_daemon_lifespan_latches_concurrent_serving_health(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda device, role: None)
    app = xpool.service.daemon.app.create_daemon()
    control = app.state.control_plane
    targets = ServingHealthTargets(
        generation=FabricGenerationId(high=1, low=1),
        listeners=(
            ("a", ServingListener(host="127.0.0.1", port=30000)),
            ("b", ServingListener(host="127.0.0.1", port=30001)),
        ),
    )
    confirmed = threading.Event()
    entered: list[int] = []
    attempts: dict[int, int] = {}

    monkeypatch.setattr(control, "watchdog", lambda: None)
    monkeypatch.setattr(control, "capture_serving_health_targets", lambda: targets)

    def confirm(candidate: ServingHealthTargets) -> bool:
        assert candidate is targets
        confirmed.set()
        return True

    monkeypatch.setattr(control, "confirm_serving_health", confirm)

    async def run() -> None:
        all_entered = asyncio.Event()

        async def probe(client: httpx.AsyncClient, listener: ServingListener) -> bool:
            entered.append(listener.port)
            attempts[listener.port] = attempts.get(listener.port, 0) + 1
            if len(entered) == len(targets.listeners):
                all_entered.set()
            if attempts[listener.port] == 1:
                await asyncio.wait_for(all_entered.wait(), timeout=0.5)
            return listener.port == 30000 or attempts[listener.port] == 2

        monkeypatch.setattr(xpool.service.daemon.app, "probe_serving_listener", probe)
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(confirmed.wait, 2.0)

    with caplog.at_level(logging.INFO, logger="xpool.service.daemon.app"):
        asyncio.run(run())

    assert entered == [30000, 30001, 30001]
    messages = [record.getMessage() for record in caplog.records]
    assert "serving healthy generation=00000000000000010000000000000001 instance_count=2" in messages
    assert "serving listener instance=a host=127.0.0.1 port=30000" in messages
    assert "serving listener instance=b host=127.0.0.1 port=30001" in messages


def test_create_daemon_initializes_native_daemon_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daemon construction initializes the process-wide native role."""

    calls: list[tuple[int | None, RuntimeRole]] = []
    monkeypatch.setattr(
        xpool.service.daemon.app.bootstrap,
        "init",
        lambda device, role: calls.append((device, role)),
    )

    app = xpool.service.daemon.app.create_daemon()

    assert calls == [(None, RuntimeRole.DAEMON)]
    assert isinstance(app.state.control_plane, xpool.service.daemon.app.ControlPlane)
    assert isinstance(app.state.daemon_failure, xpool.service.daemon.app.DaemonFailure)
    assert not app.state.daemon_failure.failed


def test_daemon_openapi_operations_have_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every supported HTTP operation exposes useful generated API documentation."""

    monkeypatch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda device, role: None)

    schema = xpool.service.daemon.app.create_daemon().openapi()

    missing = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation.get("description")
    ]
    assert missing == []
