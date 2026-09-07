from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.config
import xpool.service.client
from tests.harness.support.config import reset_global_config
from tests.harness.support.service.client import (
    initialize_client_config,
    install_scripted_http_client,
    response,
)
from xpool.config import XpoolConfig
from xpool.service.client import XpoolClient
from xpool.service.errors import XpoolClientError

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, initialize_client_config.__name__)


def test_client_health_requires_ok_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(HTTPStatus.OK, None),
            response(HTTPStatus.SERVICE_UNAVAILABLE, None),
        ],
    )

    client = XpoolClient()
    try:
        assert client.health() is HTTPStatus.OK
        with pytest.raises(XpoolClientError) as exc_info:
            client.health()
    finally:
        client.close()

    assert exc_info.value.kind == "status"
    assert exc_info.value.status_code is HTTPStatus.SERVICE_UNAVAILABLE
    assert probe.calls == [("GET", "/health", None), ("GET", "/health", None), ("GET", "/health", None)]


def test_client_close_closes_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(monkeypatch, [response(HTTPStatus.OK, None)])

    client = XpoolClient()
    client.close()

    assert probe.close_count == 1


def test_client_brackets_ipv6_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "::1", "port": 9810},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    monkeypatch.setattr(xpool.config, "global_config", config)
    probe = install_scripted_http_client(monkeypatch, [response(HTTPStatus.OK, None)])

    client = XpoolClient()
    client.close()

    assert probe.base_urls == ["http://[::1]:9810"]


def test_client_constructor_accepts_daemon_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.SERVICE_UNAVAILABLE, None),
            response(HTTPStatus.OK, None),
        ],
    )
    monkeypatch.setattr(xpool.service.client.time, "sleep", lambda seconds: None)

    client = XpoolClient()
    client.close()

    assert probe.calls == [("GET", "/health", None), ("GET", "/health", None)]
