from __future__ import annotations

from http import HTTPStatus

import httpx
import pytest

import xpool.service.client
from xpool.model import ModelId
from xpool.native import ABI_VERSION
from xpool.service.client import DAEMON_HEALTH_RETRY_ATTEMPTS, XpoolClient, XpoolDaemonError
from xpool.service.errors import XpoolClientError
from xpool.service.wire import ProcessRef
from xtest.harness.support.config import TEST_MODEL_ID, minimal_config, reset_global_config
from xtest.harness.support.service.client import (
    initialize_client_config,
    install_scripted_http_client,
    response,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, initialize_client_config.__name__)


def test_client_constructor_retries_and_rejects_unhealthy_daemon(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [response(HTTPStatus.SERVICE_UNAVAILABLE, None) for _ in range(DAEMON_HEALTH_RETRY_ATTEMPTS)],
    )
    monkeypatch.setattr(xpool.service.client.time, "sleep", lambda seconds: None)

    with caplog.at_level("WARNING", logger="xpool.service.client"):
        with pytest.raises(XpoolClientError, match="xpool daemon returned HTTP 503"):
            XpoolClient()

    assert probe.calls == [("GET", "/health", None)] * DAEMON_HEALTH_RETRY_ATTEMPTS
    assert probe.close_count == 1


def test_client_reports_config_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visibility = ("GPU-00000000-0000-0000-0000-000000000001",)
    monkeypatch.setattr(xpool.service.client, "visible_uuids", lambda: visibility)
    probe = install_scripted_http_client(
        monkeypatch,
        [
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
        ],
    )

    client = XpoolClient()
    try:
        with pytest.raises(XpoolDaemonError, match="client xpool config differs") as exc_info:
            client.check_config()
    finally:
        client.close()

    assert exc_info.value.kind == "conflict"
    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/config/check",
            {"config": minimal_config().model_dump(mode="json"), "visible_devices": list(visibility)},
        ),
    ]


@pytest.mark.parametrize(
    ("model_id", "status", "kind", "message"),
    [
        (TEST_MODEL_ID, HTTPStatus.SERVICE_UNAVAILABLE, "not_ready", "transport arena is not published"),
        (ModelId("test/missing"), HTTPStatus.NOT_FOUND, "not_found", "unknown instance"),
    ],
)
def test_client_instance_arena_preserves_structured_daemon_error(
    monkeypatch: pytest.MonkeyPatch,
    model_id: ModelId,
    status: HTTPStatus,
    kind: str,
    message: str,
) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(status, {"detail": {"kind": kind, "message": message}}),
        ],
    )

    client = XpoolClient()
    try:
        with pytest.raises(XpoolDaemonError) as exc_info:
            client.acquire_instance_transport_arena(
                model_id,
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.kind == kind
    assert exc_info.value.status_code is status
    assert len(probe.calls) == 2
    assert probe.calls[1] == (
        "POST",
        f"/instance/{model_id}/transport-arena/acquire",
        {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
    )


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("daemon restarting"),
        response(HTTPStatus.SERVICE_UNAVAILABLE, None),
    ],
)
def test_client_instance_arena_classifies_recoverable_client_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: httpx.Response | httpx.HTTPError,
) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [response(HTTPStatus.OK, None), failure],
    )

    client = XpoolClient()
    try:
        with pytest.raises(XpoolClientError) as exc_info:
            client.acquire_instance_transport_arena(
                TEST_MODEL_ID,
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert exc_info.value.is_recoverable
    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            f"/instance/{TEST_MODEL_ID}/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]


def test_client_instance_arena_reports_protocol_errors_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(HTTPStatus.OK, {"rank": 0}),
        ],
    )

    client = XpoolClient()
    try:
        with pytest.raises(XpoolClientError) as exc_info:
            client.acquire_instance_transport_arena(
                TEST_MODEL_ID,
                rank=0,
                owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
            )
    finally:
        client.close()

    assert not exc_info.value.is_recoverable
    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            f"/instance/{TEST_MODEL_ID}/transport-arena/acquire",
            {"params": {"rank": 0}, "json": {"pid": 12, "abi_version": ABI_VERSION}},
        ),
    ]
