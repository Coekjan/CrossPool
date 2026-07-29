"""Provide controlled HTTP transports and fixtures for daemon client tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
import pytest

import xpool.service.client
from tests.harness.support.config import install_test_config
from xpool.abi import TensorDType
from xpool.config import XpoolConfig
from xpool.fabric import FfnLayerKind, FfnLayerSpec, FfnWorkload
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.transport import TransportArenaHandle


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


def config() -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )


@pytest.fixture
def initialize_client_config(reset_global_config: None) -> None:
    """Install the default config required by daemon client tests."""

    install_test_config(config())


def response(status: HTTPStatus, payload: object | None) -> httpx.Response:
    return httpx.Response(
        int(status),
        json=payload,
        request=httpx.Request("GET", "http://daemon"),
    )


def transport_attributes() -> InstanceTransportAttributes:
    return InstanceTransportAttributes(
        hidden_size=4,
        max_tokens=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def arena_record(*, rank: int) -> TransportArenaHandle:
    return TransportArenaHandle(handle=f"{rank:02x}" * 64)


def workload() -> FfnWorkload:
    """Return one valid rank-independent workload for client requests."""

    return FfnWorkload(
        model_config_digest="a" * 64,
        dtype=TensorDType.BF16,
        hidden_size=4,
        layers=(FfnLayerSpec(layer_id=0, kind=FfnLayerKind.DENSE),),
        max_decode_rows=1,
        max_prefill_rows=1,
    )
