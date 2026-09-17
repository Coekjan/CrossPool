"""Provide controlled HTTP transports and fixtures for daemon client tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
import pytest
import torch

import xpool.service.client
from tests.harness.support.config import install_test_config
from xpool.config import XpoolConfig
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.native.ffn import LayerKind
from xpool.runtime.transport import InstanceRankTransportProfile
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
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
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


def transport_attributes() -> InstanceRankTransportProfile:
    return InstanceRankTransportProfile(
        hidden_size=4,
        payload_row_capacity=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def arena_record(*, rank: int) -> TransportArenaHandle:
    return TransportArenaHandle(handle=f"{rank:02x}" * 64)


def ffn_profile() -> InstanceFfnProfile:
    """Return one valid rank-independent FFN Profile for client requests."""

    return InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.bfloat16,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=1,
        group_sum_complete_admitted=False,
    )
