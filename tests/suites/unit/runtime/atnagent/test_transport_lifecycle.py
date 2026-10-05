from __future__ import annotations

import time
from types import SimpleNamespace
from typing import cast

import pytest

from xpool.config import XpoolConfig
from xpool.model import ModelId
from xpool.native import ABI_VERSION
from xpool.runtime.agent import AgentError
from xpool.runtime.atnagent import AtnAgentTransportRuntime
from xpool.service.client import XpoolClient
from xpool.service.wire import (
    AtnAgentTransportArenaBinding,
    AtnAgentTransportLeaseQuiesceResponse,
    InstanceRankRegistration,
    ProcessRef,
)
from xpool.transport import TransportArenaHandle
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config
from xtest.harness.support.runtime.atnagent import (
    instance_registration_view,
    patch_native_atnagent_ops,
    reset_agent_runtime,
    reset_atnagent_runtime,
    transport_entry,
)

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__,
    reset_agent_runtime.__name__,
    reset_atnagent_runtime.__name__,
)


def transport_config(*model_ids: str) -> XpoolConfig:
    """Install a config containing the requested model instances."""

    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": model_id, "path": f"/models/{model_id}"} for model_id in model_ids],
        }
    )
    install_test_config(config)
    return config


def create_transport_runtime(client: object) -> AtnAgentTransportRuntime:
    """Return an empty rank-zero transport runtime using a test client."""

    return AtnAgentTransportRuntime(
        client=cast(XpoolClient, client),
        device=0,
        local_rank=0,
        publisher=ProcessRef(abi_version=ABI_VERSION, pid=1),
    )


def test_transport_runtime_drains_process_wide_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime drain publishes and polls one process-wide Resident command."""

    transport_config("test/a", "test/b")
    resources = (
        transport_entry(model_id=ModelId("test/a"), rank=0, handle_rank=1),
        transport_entry(model_id=ModelId("test/b"), rank=0, handle_rank=2),
    )
    drained = False
    completed = False
    poll_results = iter((True, False))

    def drain_async() -> None:
        nonlocal drained
        assert not drained
        drained = True

    def drain_pending() -> bool:
        nonlocal completed
        assert drained
        pending = next(poll_results, False)
        completed = not pending
        return pending

    patch_native_atnagent_ops(monkeypatch, drain_async=drain_async, drain_pending=drain_pending)
    monkeypatch.setattr(
        "xpool.runtime.atnagent.time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda delay: None)
    )
    runtime = create_transport_runtime(object())
    runtime.entries = {entry.model_id: entry for entry in resources}

    runtime.drain()

    assert drained and completed


def test_transport_runtime_quiesces_leases_before_draining_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport admission closes before the process-wide Resident drains."""

    transport_config(str(TEST_MODEL_ID))
    events: list[str] = []

    class FakeClient:
        def quiesce_atnagent_transport_leases(
            self,
            device: int,
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
    runtime = create_transport_runtime(FakeClient())
    entry = transport_entry(model_id=TEST_MODEL_ID, rank=0, handle_rank=1)
    entry.published_epoch = 1
    runtime.entries = {entry.model_id: entry}

    runtime.quiesce()

    assert events == ["lease_quiesce", "resident_drain"]


def test_transport_runtime_activates_and_checks_one_process_wide_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activation and health checks do not enumerate arena handles."""

    transport_config("test/a", "test/b")
    events: list[str] = []
    patch_native_atnagent_ops(
        monkeypatch,
        activate=lambda: events.append("activate"),
        check_health=lambda: events.append("check"),
    )
    runtime = create_transport_runtime(object())
    entry = transport_entry(model_id=ModelId("test/a"), rank=0, handle_rank=1)
    runtime.entries = {entry.model_id: entry}

    runtime.activate()
    runtime.check_health()

    assert events == ["activate", "check"]


def test_transport_runtime_skips_resident_without_assigned_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle AtnAgent joins Fabric without launching an empty Transport Resident."""

    transport_config(str(TEST_MODEL_ID))
    events: list[str] = []
    patch_native_atnagent_ops(
        monkeypatch,
        activate=lambda: events.append("activate"),
        check_health=lambda: events.append("check"),
    )
    runtime = create_transport_runtime(object())

    runtime.activate()
    runtime.check_health()

    assert events == []


def test_transport_runtime_publishes_instances_incrementally_and_republishes_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model load skew does not block ready rank-local arena publication."""

    transport_config("test/a", "test/b")
    registrations = [InstanceRankRegistration.model_validate(instance_registration_view(model_id="test/a", rank=0))]
    publications: list[list[ModelId]] = []

    class FakeClient:
        def list_instances(self) -> list[InstanceRankRegistration]:
            return list(registrations)

        def upsert_atnagent_transport_arenas(
            self,
            device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: ProcessRef,
        ) -> None:
            assert device == 0
            publications.append([binding.model_id for binding in bindings])

    next_handle = 1

    def create(*args: object) -> TransportArenaHandle:
        nonlocal next_handle
        handle = TransportArenaHandle(handle=f"{next_handle:02x}" * 64)
        next_handle += 1
        return handle

    patch_native_atnagent_ops(monkeypatch, create=create)
    runtime = create_transport_runtime(FakeClient())

    instance_ranks = {ModelId("test/a"): 0, ModelId("test/b"): 0}
    assert not runtime.prepare(1, instance_ranks)
    assert publications == [[ModelId("test/a")]]

    registrations.append(InstanceRankRegistration.model_validate(instance_registration_view(model_id="test/b", rank=0)))
    assert runtime.prepare(1, instance_ranks)
    assert publications == [[ModelId("test/a")], [ModelId("test/b")]]

    assert runtime.prepare(2, instance_ranks)
    assert publications == [[ModelId("test/a")], [ModelId("test/b")], [ModelId("test/a"), ModelId("test/b")]]


def test_transport_runtime_preserves_creation_failure_when_rollback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial-arena rollback diagnostics do not replace the creation cause."""

    transport_config("test/a", "test/b")
    registrations = [
        InstanceRankRegistration.model_validate(instance_registration_view(model_id=model_id, rank=0))
        for model_id in ("test/a", "test/b")
    ]

    class FakeClient:
        def list_instances(self) -> list[InstanceRankRegistration]:
            return registrations

    def create(instance_index: int, *args: object) -> TransportArenaHandle:
        if instance_index == 1:
            raise RuntimeError("creation failed")
        return TransportArenaHandle(handle="01" * 64)

    def destroy(handle: TransportArenaHandle) -> None:
        raise RuntimeError("rollback failed")

    patch_native_atnagent_ops(monkeypatch, create=create, destroy=destroy)
    runtime = create_transport_runtime(FakeClient())

    with pytest.raises(AgentError, match="release partial arenas") as error:
        runtime.prepare(1, {ModelId("test/a"): 0, ModelId("test/b"): 0})

    cause = error.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "creation failed"
    assert cause.__notes__ == ["transport arena rollback also failed: RuntimeError: rollback failed"]


def test_transport_runtime_rejects_registration_geometry_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing native arena cannot be silently rebound to new geometry."""

    transport_config(str(TEST_MODEL_ID))
    original = InstanceRankRegistration.model_validate(instance_registration_view(model_id=str(TEST_MODEL_ID), rank=0))
    changed_view = instance_registration_view(model_id=str(TEST_MODEL_ID), rank=0)
    changed_view["transport"] = {**original.transport.model_dump(), "hidden_size": 8}
    changed = InstanceRankRegistration.model_validate(changed_view)

    class FakeClient:
        def list_instances(self) -> list[InstanceRankRegistration]:
            return [changed]

    runtime = create_transport_runtime(FakeClient())
    runtime.entries[TEST_MODEL_ID] = transport_entry(model_id=TEST_MODEL_ID, rank=0, handle_rank=1)
    runtime.entries[TEST_MODEL_ID].registration = original

    with pytest.raises(AgentError, match="hot resize is unsupported"):
        runtime.prepare(1, {TEST_MODEL_ID: 0})


def test_transport_runtime_close_destroys_all_drained_arenas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal cleanup destroys one stable already-quiesced resource snapshot."""

    transport_config("test/a", "test/b")
    runtime = create_transport_runtime(object())
    runtime.entries = {
        ModelId("test/a"): transport_entry(model_id=ModelId("test/a"), rank=0, handle_rank=1),
        ModelId("test/b"): transport_entry(model_id=ModelId("test/b"), rank=0, handle_rank=2),
    }
    events: list[tuple[str, int]] = []
    patch_native_atnagent_ops(
        monkeypatch,
        destroy=lambda handle: events.append(("destroy", int(handle.handle[:2], 16))),
    )

    runtime.close()

    assert runtime.resources == ()
    assert sorted(events) == [("destroy", 1), ("destroy", 2)]
