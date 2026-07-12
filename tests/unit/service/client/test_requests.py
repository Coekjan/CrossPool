from __future__ import annotations

from http import HTTPStatus

import pytest

from tests.harness.service.client import (
    FakeHttpClient,
    arena_record,
    config,
    response,
    transport_attributes,
)
from xpool.abi import ABI_VERSION
from xpool.service.client import ATNAGENT_TRANSPORT_DRAIN_TIMEOUT_S, XpoolClient
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportArenaDrainResponse,
    InstanceRegistration,
    ProcessRef,
)


def test_client_posts_registration_and_transport_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeHttpClient.responses = [
        response(HTTPStatus.OK, None),
        response(HTTPStatus.NO_CONTENT, None),
        response(HTTPStatus.NO_CONTENT, None),
        response(HTTPStatus.NO_CONTENT, None),
        response(HTTPStatus.OK, {"in_use": []}),
        response(HTTPStatus.NO_CONTENT, None),
        response(HTTPStatus.NO_CONTENT, None),
        response(HTTPStatus.NO_CONTENT, None),
    ]
    FakeHttpClient.calls = []
    FakeHttpClient.post_timeouts = []
    monkeypatch.setattr("xpool.service.client.httpx.Client", FakeHttpClient)

    client = XpoolClient()
    try:
        client.register_atnagent(AtnAgentRegistration(pid=11, abi_version=ABI_VERSION, cuda_device=0))
        client.upsert_atnagent_transport_arenas(
            0,
            [AtnAgentTransportArenaBinding(instance_id="m", rank=0, handle=arena_record(rank=0))],
            publisher=ProcessRef(pid=11, abi_version=ABI_VERSION),
        )
        assert client.drain_atnagent_transport_arenas(
            0,
            publisher=ProcessRef(pid=11, abi_version=ABI_VERSION),
        ) == AtnAgentTransportArenaDrainResponse(in_use=[])
        client.register_instance(
            InstanceRegistration(
                pid=12,
                abi_version=ABI_VERSION,
                instance_id="m",
                rank=0,
                transport=transport_attributes(),
            )
        )
        client.deregister_instance(
            "m",
            rank=0,
            owner=ProcessRef(pid=12, abi_version=ABI_VERSION),
        )
    finally:
        client.close()

    assert FakeHttpClient.calls == [
        ("GET", "/health", None),
        ("POST", "/config/check", config().model_dump(mode="json")),
        (
            "POST",
            "/atnagent/register",
            {"pid": 11, "abi_version": ABI_VERSION, "cuda_device": 0},
        ),
        (
            "POST",
            "/atnagent/0/transport-arenas",
            {
                "publisher": {"pid": 11, "abi_version": ABI_VERSION},
                "bindings": [{"instance_id": "m", "rank": 0, "handle": arena_record(rank=0).model_dump(mode="json")}],
            },
        ),
        (
            "POST",
            "/atnagent/0/transport-arenas/drain",
            {"pid": 11, "abi_version": ABI_VERSION},
        ),
        (
            "POST",
            "/config/check",
            config().model_dump(mode="json"),
        ),
        (
            "POST",
            "/instance/register",
            {
                "pid": 12,
                "abi_version": ABI_VERSION,
                "instance_id": "m",
                "rank": 0,
                "transport": transport_attributes().model_dump(mode="json"),
            },
        ),
        (
            "POST",
            "/instance/m/deregister",
            {
                "params": {"rank": 0},
                "json": {"pid": 12, "abi_version": ABI_VERSION},
            },
        ),
    ]
    assert (
        "/atnagent/0/transport-arenas/drain",
        ATNAGENT_TRANSPORT_DRAIN_TIMEOUT_S,
    ) in FakeHttpClient.post_timeouts
