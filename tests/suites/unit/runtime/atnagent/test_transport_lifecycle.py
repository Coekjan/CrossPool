from __future__ import annotations

from typing import cast

import pytest

from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.runtime.atnagent import (
    instance_registration_view,
    patch_native_atnagent_ops,
    reset_agent_runtime,
    reset_atnagent_runtime,
    transport_entry,
)
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig
from xpool.runtime.agent import AgentError
from xpool.runtime.atnagent import AtnTransportCatalog
from xpool.service.client import XpoolClient
from xpool.service.wire import (
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
    InstanceRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__,
    reset_agent_runtime.__name__,
    reset_atnagent_runtime.__name__,
)


def catalog_config(*model_ids: str) -> XpoolConfig:
    """Install a config containing the requested model instances."""

    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": model_id, "path": f"/models/{model_id}"} for model_id in model_ids],
        }
    )
    install_test_config(config)
    return config


def transport_catalog(client: object) -> AtnTransportCatalog:
    """Return an empty rank-zero catalog using a test client."""

    return AtnTransportCatalog(
        client=cast(XpoolClient, client),
        cuda_device=0,
        local_rank=0,
        publisher=ProcessRef(abi_version=ABI_VERSION, pid=1),
    )


def test_catalog_drains_process_wide_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalog drain publishes and polls one process-wide Resident command."""

    catalog_config("a", "b")
    resources = (
        transport_entry(instance_id="a", rank=0, handle_rank=1),
        transport_entry(instance_id="b", rank=0, handle_rank=2),
    )
    events: list[str] = []
    poll_results = iter((True, False))

    def drain_async() -> None:
        events.append("drain")

    def drain_pending() -> bool:
        events.append("poll")
        return next(poll_results)

    patch_native_atnagent_ops(monkeypatch, drain_async=drain_async, drain_pending=drain_pending)
    monkeypatch.setattr("xpool.runtime.atnagent.time.sleep", lambda delay: events.append("sleep"))
    catalog = transport_catalog(object())
    catalog.entries = {entry.instance_id: entry for entry in resources}

    catalog.drain()

    assert events == ["drain", "poll", "sleep", "poll"]


def test_catalog_quiesces_leases_before_draining_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport admission closes before the process-wide Resident drains."""

    catalog_config("m")
    events: list[str] = []

    class FakeClient:
        def quiesce_atnagent_transport_leases(
            self,
            cuda_device: int,
            *,
            publisher: ProcessRef,
        ) -> AtnAgentTransportLeaseQuiesceResponse:
            events.append("lease_quiesce")
            return AtnAgentTransportLeaseQuiesceResponse(in_use=[])

    patch_native_atnagent_ops(
        monkeypatch,
        drain_async=lambda: events.append("resident_drain"),
        drain_pending=lambda: False,
    )
    catalog = transport_catalog(FakeClient())
    entry = transport_entry(instance_id="m", rank=0, handle_rank=1)
    entry.published_epoch = 1
    catalog.entries = {entry.instance_id: entry}

    catalog.quiesce()

    assert events == ["lease_quiesce", "resident_drain"]


def test_catalog_activates_and_checks_one_process_wide_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activation and health checks do not enumerate arena handles."""

    catalog_config("a", "b")
    events: list[str] = []
    patch_native_atnagent_ops(
        monkeypatch,
        activate=lambda: events.append("activate"),
        check_health=lambda: events.append("check"),
    )
    catalog = transport_catalog(object())

    catalog.activate()
    catalog.check_health()

    assert events == ["activate", "check"]


def test_catalog_publishes_instances_incrementally_and_republishes_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model load skew does not block ready rank-local arena publication."""

    catalog_config("a", "b")
    registrations = [InstanceRegistration.model_validate(instance_registration_view(instance_id="a", rank=0))]
    publications: list[list[str]] = []

    class FakeClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return list(registrations)

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: ProcessRef,
        ) -> None:
            assert cuda_device == 0
            publications.append([binding.instance_id for binding in bindings])

    next_handle = 1

    def create(*args: object) -> TransportArenaHandle:
        nonlocal next_handle
        handle = TransportArenaHandle(handle=f"{next_handle:02x}" * 64)
        next_handle += 1
        return handle

    patch_native_atnagent_ops(monkeypatch, create=create)
    catalog = transport_catalog(FakeClient())

    assert not catalog.reconcile(1)
    assert publications == [["a"]]

    registrations.append(InstanceRegistration.model_validate(instance_registration_view(instance_id="b", rank=0)))
    assert catalog.reconcile(1)
    assert publications == [["a"], ["b"]]

    assert catalog.reconcile(2)
    assert publications == [["a"], ["b"], ["a", "b"]]


def test_catalog_rejects_registration_geometry_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing native arena cannot be silently rebound to new geometry."""

    catalog_config("m")
    original = InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))
    changed_view = instance_registration_view(instance_id="m", rank=0)
    changed_view["transport"] = {**cast(dict[str, object], changed_view["transport"]), "hidden_size": 8}
    changed = InstanceRegistration.model_validate(changed_view)

    class FakeClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [changed]

    catalog = transport_catalog(FakeClient())
    catalog.entries["m"] = transport_entry(instance_id="m", rank=0, handle_rank=1)
    catalog.entries["m"].registration = original

    with pytest.raises(AgentError, match="hot resize is unsupported"):
        catalog.reconcile(1)


def test_catalog_close_uses_collection_destroy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal cleanup destroys one stable already-quiesced resource snapshot."""

    catalog_config("a", "b")
    catalog = transport_catalog(object())
    catalog.entries = {
        "a": transport_entry(instance_id="a", rank=0, handle_rank=1),
        "b": transport_entry(instance_id="b", rank=0, handle_rank=2),
    }
    events: list[tuple[str, int]] = []
    patch_native_atnagent_ops(
        monkeypatch,
        destroy=lambda handle: events.append(("destroy", int(handle.handle[:2], 16))),
    )

    catalog.close()

    assert catalog.resources == ()
    assert events == [("destroy", 1), ("destroy", 2)]
