from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

import pytest

import xpool.service.client
from xpool.fabric import FabricGenerationId, FabricRole
from xpool.native import ABI_VERSION
from xpool.service.client import ATNAGENT_TRANSPORT_LEASE_QUIESCE_TIMEOUT_S, XpoolClient
from xpool.service.errors import XpoolClientError
from xpool.service.wire import (
    AgentStartupAdmission,
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
    FfnAgentRegistration,
    InstanceRankRegistration,
    MpsClientTermination,
    ProcessRef,
)
from xpool.utils.mps import MPS_STARTUP_TIMEOUT_S, MPS_TERMINATION_TIMEOUT_S
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config
from xtest.harness.support.kv import kv_capacity_profile
from xtest.harness.support.runtime.instance import ffn_profile, transport_attributes
from xtest.harness.support.service.client import (
    arena_record,
    initialize_client_config,
    install_scripted_http_client,
    response,
)
from xtest.harness.support.service.daemon import ffnagent_registration

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, initialize_client_config.__name__)


def test_agent_startup_admission_http_preserves_management_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = install_scripted_http_client(
        monkeypatch, [response(HTTPStatus.OK, None), response(HTTPStatus.NO_CONTENT, None)]
    )
    payload = AgentStartupAdmission(
        pid=11,
        create_time=1.0,
        abi_version=ABI_VERSION,
        role=FabricRole.ATNAGENT,
        device=0,
    )
    client = XpoolClient()
    try:
        client.admit_agent_startup(payload)
    finally:
        client.close()
    assert probe.calls == [("GET", "/health", None), ("POST", "/startup/agent", payload.model_dump(mode="json"))]
    assert probe.post_timeouts == [("/startup/agent", MPS_STARTUP_TIMEOUT_S)]


@pytest.mark.parametrize("status", [HTTPStatus.NO_CONTENT, HTTPStatus.OK])
def test_mps_termination_requires_explicit_acknowledgement_within_cleanup_budget(
    status: HTTPStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xpool.service.client, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    probe = install_scripted_http_client(
        monkeypatch,
        [response(HTTPStatus.OK, None), response(status, None)],
    )
    payload = MpsClientTermination(pid=11, create_time=1.0, abi_version=ABI_VERSION, deadline=1100.0)
    client = XpoolClient()
    try:
        if status == HTTPStatus.NO_CONTENT:
            client.terminate_serving_client(payload)
        else:
            with pytest.raises(XpoolClientError, match="did not confirm MPS client termination"):
                client.terminate_serving_client(payload)
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
        ("POST", "/serving/mps/terminate-client", payload.model_dump(mode="json")),
    ]
    assert probe.post_timeouts == [("/serving/mps/terminate-client", MPS_TERMINATION_TIMEOUT_S)]


def test_expired_mps_termination_does_not_send_an_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.client, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    probe = install_scripted_http_client(monkeypatch, [response(HTTPStatus.OK, None)])
    payload = MpsClientTermination(pid=11, create_time=1.0, abi_version=ABI_VERSION, deadline=1000.0)
    client = XpoolClient()
    try:
        with pytest.raises(TimeoutError, match="deadline expired"):
            client.terminate_serving_client(payload)
    finally:
        client.close()

    assert probe.calls == [("GET", "/health", None)]


@pytest.mark.parametrize("participant", ["atnagent", "ffnagent", "instance"])
def test_participant_registration_sends_declared_contract(
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
            registration = AtnAgentRegistration(pid=11, abi_version=ABI_VERSION, device=0)
            client.register_atnagent(registration)
            expected_path = "/atnagent/register"
        elif participant == "ffnagent":
            registration = FfnAgentRegistration.model_validate(ffnagent_registration(pid=12))
            client.register_ffnagent(registration)
            expected_path = "/ffnagent/register"
        else:
            registration = InstanceRankRegistration(
                pid=13,
                abi_version=ABI_VERSION,
                model_id=TEST_MODEL_ID,
                rank=0,
                transport=transport_attributes(),
                ffn_profile=ffn_profile(),
                kv_capacity=kv_capacity_profile(),
                atn_runtime_headroom_bytes=0,
            )
            client.register_instance(registration)
            expected_path = "/instance/register"
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
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
            [AtnAgentTransportArenaBinding(model_id=TEST_MODEL_ID, rank=0, handle=arena_record(rank=0))],
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
                "bindings": [
                    {"model_id": str(TEST_MODEL_ID), "rank": 0, "handle": {"handle": arena_record(rank=0).handle}}
                ],
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
        client.deregister_instance(TEST_MODEL_ID, rank=0, owner=owner)
    finally:
        client.close()

    assert probe.calls == [
        ("GET", "/health", None),
        (
            "POST",
            f"/instance/{TEST_MODEL_ID}/deregister",
            {
                "params": {"rank": 0},
                "json": {"pid": 12, "abi_version": ABI_VERSION},
            },
        ),
    ]


def test_kv_control_channel_uses_generation_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    generation = FabricGenerationId(high=1, low=2)
    probe = install_scripted_http_client(
        monkeypatch,
        [
            response(HTTPStatus.OK, None),
            response(HTTPStatus.OK, {"generation": {"high": 1, "low": 2}, "name": "/xpool-kv-test"}),
        ],
    )

    client = XpoolClient()
    try:
        channel = client.kv_control_channel(generation)
    finally:
        client.close()

    assert channel.generation == generation
    assert channel.name == "/xpool-kv-test"
    assert probe.calls == [
        ("GET", "/health", None),
        ("GET", "/kv/control-channel/00000000000000010000000000000002", None),
    ]
