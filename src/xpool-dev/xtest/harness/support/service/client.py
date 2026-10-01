"""Provide controlled HTTP transports and fixtures for daemon client tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
import pytest

import xpool.service.client
from xpool.transport import TransportArenaHandle
from xtest.harness.support.config import install_test_config, minimal_config


@dataclass(slots=True)
class ScriptedHttpClientProbe:
    """Test-local HTTP script and observations for one XpoolClient test."""

    responses: list[httpx.Response | httpx.HTTPError]
    calls: list[tuple[str, str, object | None]] = field(default_factory=list)
    post_timeouts: list[tuple[str, float | None]] = field(default_factory=list)
    base_urls: list[str] = field(default_factory=list)
    close_count: int = 0


def install_scripted_http_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response | httpx.HTTPError],
) -> ScriptedHttpClientProbe:
    """Install one isolated scripted HTTP client and return its observations."""

    probe = ScriptedHttpClientProbe(responses=list(responses))

    class ScriptedHttpClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            probe.base_urls.append(str(kwargs["base_url"]))

        def close(self) -> None:
            probe.close_count += 1

        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, int | str | list[str]] | None = None,
            json: object | None = None,
            timeout: float | None = None,
        ) -> httpx.Response:
            match method:
                case "GET":
                    return self.get(path, params=params, timeout=timeout)
                case "POST":
                    return self.post(path, params=params, json=json, timeout=timeout)
                case _:
                    raise AssertionError(f"unexpected method: {method}")

        def get(
            self,
            path: str,
            *,
            params: dict[str, int | str | list[str]] | None = None,
            timeout: float | None = None,
        ) -> httpx.Response:
            probe.calls.append(("GET", path, params))
            return next_response()

        def post(
            self,
            path: str,
            *,
            params: dict[str, int | str | list[str]] | None = None,
            json: object | None = None,
            timeout: float | None = None,
        ) -> httpx.Response:
            probe.calls.append(("POST", path, json if params is None else {"params": params, "json": json}))
            probe.post_timeouts.append((path, timeout))
            return next_response()

    def next_response() -> httpx.Response:
        if not probe.responses:
            raise AssertionError("scripted HTTP client received an unexpected request")
        scripted = probe.responses.pop(0)
        if isinstance(scripted, httpx.HTTPError):
            raise scripted
        return scripted

    monkeypatch.setattr(xpool.service.client.httpx, "Client", ScriptedHttpClient)
    return probe


@pytest.fixture
def initialize_client_config(reset_global_config: None) -> None:
    """Install the default config required by daemon client tests."""

    install_test_config(minimal_config())


def response(status: HTTPStatus, payload: object | None) -> httpx.Response:
    return httpx.Response(
        int(status),
        json=payload,
        request=httpx.Request("GET", "http://daemon"),
    )


def arena_record(*, rank: int) -> TransportArenaHandle:
    return TransportArenaHandle(handle=f"{rank:02x}" * 64)
