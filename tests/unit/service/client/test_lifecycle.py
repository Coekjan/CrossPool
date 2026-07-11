from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.config as config_module
import xpool.service.client as client_module
from tests.harness.service.client import (
    FakeHttpClient,
    XpoolConfig,
    response,
)
from xpool.service.client import XpoolClient
from xpool.service.errors import XpoolClientError


def test_client_health_requires_ok_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(HTTPStatus.OK, None),
        response(HTTPStatus.SERVICE_UNAVAILABLE, None),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        assert client.health() is HTTPStatus.OK
        with pytest.raises(XpoolClientError) as exc_info:
            client.health()
    finally:
        client.close()

    assert exc_info.value.kind == "status"
    assert exc_info.value.status_code is HTTPStatus.SERVICE_UNAVAILABLE
    assert FakeHttpClient.calls == [("GET", "/health", None), ("GET", "/health", None), ("GET", "/health", None)]


def test_client_close_closes_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [response(HTTPStatus.OK, None)]
    FakeHttpClient.calls = []
    FakeHttpClient.close_count = 0
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    client.close()

    assert FakeHttpClient.close_count == 1


def test_client_brackets_ipv6_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    base_urls: list[str] = []

    class RecordingHttpClient(FakeHttpClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            base_urls.append(str(kwargs["base_url"]))
            super().__init__(*args, **kwargs)

    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "::1", "port": 9810},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    monkeypatch.setattr(config_module, "global_config", config)
    RecordingHttpClient.responses = [response(HTTPStatus.OK, None)]
    monkeypatch.setattr("xpool.service.client.httpx.Client", RecordingHttpClient)

    client = XpoolClient()
    client.close()

    assert base_urls == ["http://[::1]:9810"]


def test_client_constructor_accepts_daemon_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.SERVICE_UNAVAILABLE, None),
        response(HTTPStatus.OK, None),
    ]
    FakeHttpClient.calls = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    client = XpoolClient()
    client.close()

    assert FakeHttpClient.calls == [("GET", "/health", None), ("GET", "/health", None)]
