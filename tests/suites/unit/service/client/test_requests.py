from __future__ import annotations

from http import HTTPStatus

import pytest

from tests.harness.service.client import (
    arena_record,
    config,
    initialize_client_config,
    install_scripted_http_client,
    response,
    transport_attributes,
    workload,
)
from xpool.abi import ABI_VERSION
from xpool.service.client import ATNAGENT_TRANSPORT_LEASE_QUIESCE_TIMEOUT_S, XpoolClient
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
    FfnAgentRegistration,
    InstanceRegistration,
    ProcessRef,
)

pytestmark = pytest.mark.usefixtures(initialize_client_config.__name__)


@pytest.mark.parametrize("participant", ["atnagent", "ffnagent", "instance"])
def test_participant_registration_follows_config_check(
    monkeypatch: pytest.MonkeyPatch,
    participant: str,
) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(HTTPStatus.NO_CONTENT, None),
            response(HTTPStatus.NO_CONTENT, None),
        ],
    )

    client = XpoolClient()
    try:
        if participant == "atnagent":
            registration = AtnAgentRegistration(pid=11, abi_version=ABI_VERSION, cuda_device=0)
            client.register_atnagent(registration)
            expected_path = "/atnagent/register"
        elif participant == "ffnagent":
            registration = FfnAgentRegistration(pid=12, abi_version=ABI_VERSION, cuda_device=1)
            client.register_ffnagent(registration)
            expected_path = "/ffnagent/register"
        else:
            registration = InstanceRegistration(
                pid=13,
                abi_version=ABI_VERSION,
                instance_id="m",
                rank=0,
                transport=transport_attributes(),
                workload=workload(),
            )
            client.register_instance(registration)
            expected_path = "/instance/register"
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
        ("POST", "/config/check", config().model_dump(mode="json")),
        ("POST", expected_path, registration.model_dump(mode="json")),
    ]


def test_transport_publication_and_quiesce_preserve_owner_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(HTTPStatus.NO_CONTENT, None),
            response(HTTPStatus.OK, {"in_use": []}),
        ],
    )
    publisher = ProcessRef(pid=11, abi_version=ABI_VERSION)

    client = XpoolClient()
    try:
        client.upsert_atnagent_transport_arenas(
            0,
            [AtnAgentTransportArenaBinding(instance_id="m", rank=0, handle=arena_record(rank=0))],
            publisher=publisher,
        )
        assert client.quiesce_atnagent_transport_leases(
            0,
            publisher=publisher,
        ) == AtnAgentTransportLeaseQuiesceResponse(in_use=[])
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/atnagent/0/transport-arenas",
            {
                "publisher": {"pid": 11, "abi_version": ABI_VERSION},
                "bindings": [{"instance_id": "m", "rank": 0, "handle": {"handle": arena_record(rank=0).handle}}],
            },
        ),
        (
            "POST",
            "/atnagent/0/transport-leases/quiesce",
            {"pid": 11, "abi_version": ABI_VERSION},
        ),
    ]
    assert probe.post_timeouts[-1] == (
        "/atnagent/0/transport-leases/quiesce",
        ATNAGENT_TRANSPORT_LEASE_QUIESCE_TIMEOUT_S,
    )


def test_instance_deregistration_preserves_rank_and_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch,
        [response(HTTPStatus.OK, None), response(HTTPStatus.NO_CONTENT, None)],
    )
    owner = ProcessRef(pid=12, abi_version=ABI_VERSION)

    client = XpoolClient()
    try:
        client.deregister_instance("m", rank=0, owner=owner)
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/instance/m/deregister",
            {
                "params": {"rank": 0},
                "json": {"pid": 12, "abi_version": ABI_VERSION},
            },
        ),
    ]
