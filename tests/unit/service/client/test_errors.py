from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.service.client as client_module
from tests.harness.service.client import (
    FakeHttpClient,
    config,
    httpx,
    response,
    transport_attributes,
)
from xpool.abi import ABI_VERSION
from xpool.service.client import DAEMON_HEALTH_RETRY_ATTEMPTS, XpoolClient, XpoolDaemonError
from xpool.service.errors import XpoolClientError
from xpool.service.wire import AtnAgentRegistration, InstanceRegistration, ProcessRef


def test_client_constructor_retries_and_rejects_unhealthy_daemon(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.SERVICE_UNAVAILABLE, None) for attempt in range(DAEMON_HEALTH_RETRY_ATTEMPTS)
    ]
    FakeHttpClient.calls = []
    FakeHttpClient.close_count = 0
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    with caplog.at_level("WARNING", logger="xpool.service.client"):
        with pytest.raises(XpoolClientError, match="xpool daemon returned HTTP 503"):
            XpoolClient()

    assert FakeHttpClient.calls == [("GET", "/health", None)] * DAEMON_HEALTH_RETRY_ATTEMPTS
    assert FakeHttpClient.close_count == 1
    assert len(caplog.messages) == DAEMON_HEALTH_RETRY_ATTEMPTS


@pytest.mark.parametrize("participant", ["atnagent", "instance"])
def test_client_rejects_registration_after_config_conflict(
    monkeypatch: pytest.MonkeyPatch,
    participant: str,
) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(
            HTTPStatus.CONFLICT,
            {
                "detail": {
                    "kind": "conflict",
                    "message": "client xpool config differs from daemon config",
                }
            },
        ),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolDaemonError, match="client xpool config differs") as exc_info:
            if participant == "atnagent":
                client.register_atnagent(AtnAgentRegistration(pid=11, abi_version=ABI_VERSION, cuda_device=0))
            else:
                client.register_instance(
                    InstanceRegistration(
                        pid=12,
                        abi_version=ABI_VERSION,
                        instance_id="m",
                        rank=0,
                        transport=transport_attributes(),
                    )
                )
    finally:
        client.close()

    assert exc_info.value.kind == "conflict"
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        ("POST", "/config/check", config().model_dump(mode="json")),
    ]


def test_client_instance_arena_reports_not_ready_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {"detail": {"kind": "not_ready", "message": "transport arena is not published"}},
        ),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolDaemonError) as exc_info:
            client.acquire_instance_transport_arena(
                "deepseek-ai/DeepSeek-V2-Lite-Chat",
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.is_recoverable
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/deepseek-ai/DeepSeek-V2-Lite-Chat/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]


def test_client_instance_arena_does_not_retry_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(HTTPStatus.NOT_FOUND, {"detail": {"kind": "not_found", "message": "unknown instance"}}),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolDaemonError) as exc_info:
            client.acquire_instance_transport_arena(
                "missing",
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.kind == "not_found"
    assert exc_info.value.status_code is HTTPStatus.NOT_FOUND
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/missing/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]


def test_client_instance_arena_reports_transport_errors_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        httpx.ConnectError("daemon restarting"),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolClientError) as exc_info:
            client.acquire_instance_transport_arena(
                "m",
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.is_recoverable
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/m/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]


def test_client_instance_arena_reports_retryable_gateway_status(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(HTTPStatus.SERVICE_UNAVAILABLE, None),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolClientError) as exc_info:
            client.acquire_instance_transport_arena(
                "m",
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.is_recoverable
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/m/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]


def test_client_instance_arena_reports_protocol_errors_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(HTTPStatus.OK, {"rank": 0}),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        with pytest.raises(XpoolClientError) as exc_info:
            client.acquire_instance_transport_arena(
                "m",
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert not exc_info.value.is_recoverable
    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/m/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]
