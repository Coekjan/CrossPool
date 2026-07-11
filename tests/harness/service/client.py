from __future__ import annotations

from http import HTTPStatus
from typing import ClassVar

import httpx

from xpool.abi import TransportArenaHandle
from xpool.config import XpoolConfig
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.service.wire import (
    TransportArenaHandleRecord,
)


class FakeHttpClient:
    responses: ClassVar[list[httpx.Response | httpx.HTTPError]] = []
    calls: ClassVar[list[tuple[str, str, object | None]]] = []
    post_timeouts: ClassVar[list[tuple[str, float | None]]] = []
    close_count: ClassVar[int] = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    def close(self) -> None:
        type(self).close_count += 1
        return None

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
        self.calls.append(("GET", path, params))
        response = self.responses.pop(0)
        if isinstance(response, httpx.HTTPError):
            raise response
        return response

    def post(
        self,
        path: str,
        *,
        params: dict[str, int | str | list[str]] | None = None,
        json: object | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        self.calls.append(("POST", path, json if params is None else {"params": params, "json": json}))
        self.post_timeouts.append((path, timeout))
        response = self.responses.pop(0)
        if isinstance(response, httpx.HTTPError):
            raise response
        return response


def config() -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )


def response(status: HTTPStatus, payload: object | None) -> httpx.Response:
    return httpx.Response(
        int(status),
        json=payload,
        request=httpx.Request("GET", "http://daemon"),
    )


def transport_attributes() -> InstanceTransportAttributes:
    return InstanceTransportAttributes(
        element_size=4,
        hidden_size=4,
        max_tokens=8,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=0,
        atn_dp_size=1,
    )


def arena_record(*, rank: int) -> TransportArenaHandleRecord:
    return TransportArenaHandleRecord(handle=f"{rank:02x}" * 64)


def arena(*, rank: int) -> TransportArenaHandle:
    return arena_record(rank=rank).to_handle()
