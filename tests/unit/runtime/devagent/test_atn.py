from __future__ import annotations

import logging
from typing import cast

from tests.harness.runtime.devagent import (
    DevagentError,
    InstanceRegistration,
    TransportArenaHandle,
    XpoolClientError,
    XpoolConfig,
    XpoolDaemonError,
    atn_module,
    create_devagent,
    instance_registration_view,
    patch_native_devagent_ops,
    pytest,
    transport_arena,
    transport_arena_resource,
)
from xpool.abi import ABI_VERSION
from xpool.runtime.devagent.atn import AtnArenaCreated, AtnArenaPublished, AtnInitialized, AtnRegistered
from xpool.runtime.devagent.common import DEVAGENT_CONTROL_INTERVAL_S, DEVAGENT_SHUTDOWN_POLL_INTERVAL_S
from xpool.service.client import XpoolClient
from xpool.service.wire import (
    DevagentRegistration,
    DevagentTransportArenaBinding,
    DevagentTransportArenaDrainResponse,
    InstanceRankRef,
    ReadinessStatus,
)


def test_non_main_attention_devagent_publishes_local_arenas(monkeypatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append(("register", registration.model_dump(mode="json")))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=1))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_fourth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    patch_native_devagent_ops(monkeypatch, events=events)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_fourth_sleep)

    create_devagent(config, cuda_device=1).run()

    registration_payload = cast("dict[str, object]", events[0][1])
    assert isinstance(registration_payload, dict)
    assert "create_time" not in registration_payload
    assert "nvshmem_rank" not in registration_payload
    assert registration_payload["abi_version"] == ABI_VERSION
    assert events == [
        ("register", registration_payload),
        ("heartbeat", 1),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("arenas", (1, [{"instance_id": "m", "rank": 1, "handle": transport_arena(1)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("drain", 1),
        ("destroy", 1),
    ]


def test_attention_devagent_destroys_arenas_after_arenas_drain_retries(
    monkeypatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        drain_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append(("register", "devagent"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            FakeXpoolClient.drain_count += 1
            if FakeXpoolClient.drain_count < 3:
                events.append(("drain-failed", cuda_device))
                raise XpoolClientError("transport", "daemon unavailable")
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def interrupt_fourth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", interrupt_fourth_sleep)
    patch_native_devagent_ops(monkeypatch, events=events)

    create_devagent(config, cuda_device=0).run()

    assert events == [
        ("register", "devagent"),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("drain-failed", 0),
        ("sleep", DEVAGENT_SHUTDOWN_POLL_INTERVAL_S),
        ("drain-failed", 0),
        ("sleep", DEVAGENT_SHUTDOWN_POLL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


@pytest.mark.parametrize("status", [ReadinessStatus.ONLINE, ReadinessStatus.STALE])
def test_attention_devagent_waits_for_attached_instances_before_destroying_arenas(
    monkeypatch,
    status: ReadinessStatus,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    agent.state = AtnArenaPublished((transport_arena_resource(instance_id="m", rank=0, handle_rank=0),))
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        drain_count = 0

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            FakeXpoolClient.drain_count += 1
            if FakeXpoolClient.drain_count == 1:
                events.append(("drain", (cuda_device, status.value)))
                return DevagentTransportArenaDrainResponse(
                    in_use=[InstanceRankRef(pid=1, abi_version=ABI_VERSION, instance_id="m", rank=0)]
                )
            events.append(("drain", (cuda_device, ReadinessStatus.OFFLINE.value)))
            return DevagentTransportArenaDrainResponse(in_use=[])

    monkeypatch.setattr(atn_module.time, "sleep", lambda delay: events.append(("sleep", delay)))
    patch_native_devagent_ops(monkeypatch, events=events)

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    agent.shutdown()

    assert events == [
        ("drain", (0, status.value)),
        ("sleep", DEVAGENT_SHUTDOWN_POLL_INTERVAL_S),
        ("drain", (0, ReadinessStatus.OFFLINE.value)),
        ("destroy", 0),
    ]
    assert agent.state == AtnInitialized()


def test_attention_devagent_published_state_does_not_poll_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    old_resource = transport_arena_resource(instance_id="a", rank=0, handle_rank=0)
    stable_resource = transport_arena_resource(instance_id="b", rank=0, handle_rank=17)
    agent.state = AtnArenaPublished((old_resource, stable_resource))

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            pytest.fail("published steady state must not poll instance registrations")

    patch_native_devagent_ops(
        monkeypatch,
        create=lambda *args, **kwargs: pytest.fail("hot resize must fail before creating a replacement handle"),
    )

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    agent.advance_state()

    assert agent.state == AtnArenaPublished((old_resource, stable_resource))


def test_attention_devagent_fails_after_cleaning_partial_arena_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    agent.state = AtnRegistered()
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [
                InstanceRegistration.model_validate(instance_registration_view(instance_id="a", rank=0)),
                InstanceRegistration.model_validate(instance_registration_view(instance_id="b", rank=0)),
            ]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            pytest.fail("arena creation failure must fail before upserting")

    def fake_create(*args: object) -> TransportArenaHandle:
        create_index = sum(1 for event in events if event[0] == "create")
        events.append(("create", create_index))
        if create_index == 1:
            raise RuntimeError("cuda allocation failed")
        return TransportArenaHandle(handle=f"{create_index:02x}" * 64)

    patch_native_devagent_ops(
        monkeypatch,
        create=fake_create,
        events=events,
    )

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    with pytest.raises(DevagentError, match="failed to create transport arenas"):
        agent.advance_state()

    assert agent.state == AtnRegistered()
    assert events == [("create", 0), ("create", 1), ("destroy", 0)]


def test_attention_devagent_fails_when_kernel_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

    def fake_start(handle: TransportArenaHandle) -> None:
        events.append(("start", int(handle.handle[:2], 16)))
        raise RuntimeError("kernel launch failed")

    patch_native_devagent_ops(
        monkeypatch,
        events=events,
        launch=fake_start,
        destroy=lambda handle: pytest.fail("kernel launch failure must not destroy arena in upsert"),
    )

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    resource = transport_arena_resource(instance_id="m", rank=0, handle_rank=0)
    agent.state = AtnArenaCreated(
        resources=(resource,),
        created_resources=(resource,),
        publication_resources=(resource,),
    )
    with pytest.raises(DevagentError, match="devagent transport kernel launch failed"):
        agent.advance_state()

    assert isinstance(agent.state, AtnArenaCreated)
    assert events == [("start", 0)]


def test_attention_devagent_launches_kernel_before_upserting_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [binding.model_dump(mode="json") for binding in bindings])))

    def fake_start(handle: TransportArenaHandle) -> None:
        events.append(("start", int(handle.handle[:2], 16)))

    patch_native_devagent_ops(monkeypatch, events=events, launch=fake_start)

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    resource = transport_arena_resource(instance_id="m", rank=0, handle_rank=0)
    agent.state = AtnArenaCreated(
        resources=(resource,),
        created_resources=(resource,),
        publication_resources=(resource,),
    )
    agent.advance_state()

    assert isinstance(agent.state, atn_module.AtnKernelLaunched)
    assert events == [("start", 0)]

    agent.advance_state()

    assert isinstance(agent.state, AtnArenaPublished)
    assert events == [
        ("start", 0),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
    ]


def test_attention_devagent_republishes_arenas_after_daemon_reregister(monkeypatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        heartbeat_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append(("register", "devagent"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", FakeXpoolClient.heartbeat_count))
            if FakeXpoolClient.heartbeat_count == 5:
                raise XpoolDaemonError("not_ready", "registration missing")

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_sixth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 6:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_sixth_sleep)
    patch_native_devagent_ops(monkeypatch, events=events)

    create_devagent(config, cuda_device=0).run()

    assert events == [
        ("register", "devagent"),
        ("heartbeat", 1),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 2),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 3),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 4),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 5),
        ("register", "devagent"),
        ("heartbeat", 6),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 7),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_devagent_retries_not_ready_arena_publication(monkeypatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        upsert_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append(("register", "devagent"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            if bindings:
                FakeXpoolClient.upsert_count += 1
                if FakeXpoolClient.upsert_count == 1:
                    events.append(("arenas-not-ready", cuda_device))
                    raise XpoolDaemonError("not_ready", "devagent stale")
                events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))
                return

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_publication(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_publication)
    patch_native_devagent_ops(monkeypatch, events=events)

    create_devagent(config, cuda_device=0).run()

    assert events == [
        ("register", "devagent"),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas-not-ready", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_devagent_survives_transport_error_during_reregister(monkeypatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        heartbeat_count = 0
        register_count = 0

        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            FakeXpoolClient.register_count += 1
            events.append(("register", FakeXpoolClient.register_count))
            if FakeXpoolClient.register_count == 2:
                raise XpoolClientError("transport", "daemon restarting")

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", FakeXpoolClient.heartbeat_count))
            if FakeXpoolClient.heartbeat_count <= 2:
                raise XpoolDaemonError("not_ready", "registration missing")

        def list_instances(self) -> list[object]:
            if FakeXpoolClient.heartbeat_count < 3:
                return []
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_seventh_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 7:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_seventh_sleep)
    patch_native_devagent_ops(monkeypatch, events=events)

    create_devagent(config, cuda_device=0).run()

    assert events == [
        ("register", 1),
        ("heartbeat", 1),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("register", 2),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("register", 3),
        ("heartbeat", 2),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("register", 4),
        ("heartbeat", 3),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 4),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 5),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 6),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_devagent_initial_register_conflict_fails_loud_without_shutdown(
    monkeypatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[str] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append("register")
            raise XpoolDaemonError("not_found", "unknown devagent")

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)

    with pytest.raises(DevagentError, match="devagent registration received unrecoverable daemon error"):
        create_devagent(config, cuda_device=0).run()

    assert events == ["register"]


def test_attention_devagent_stops_on_daemon_conflict(
    monkeypatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: DevagentRegistration) -> None:
            events.append(("register", "devagent"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))
            raise XpoolDaemonError("conflict", "pid mismatch")

        def list_instances(self) -> list[object]:
            pytest.fail("conflicted devagent must not reconcile instance arenas")

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)

    with pytest.raises(DevagentError, match="unrecoverable daemon error"):
        create_devagent(config, cuda_device=0).run()

    assert events == [
        ("register", "devagent"),
        ("heartbeat", 0),
    ]


def test_attention_devagent_waits_for_local_instance_registrations(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: object) -> None:
            events.append(("register", "devagent"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return []

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_devagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> DevagentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return DevagentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_second_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_second_sleep)
    patch_native_devagent_ops(monkeypatch, events=events)

    with caplog.at_level(logging.INFO, logger="xpool.runtime.devagent"):
        create_devagent(config, cuda_device=0).run()

    assert caplog.messages == [
        "waiting for local instance registrations on CUDA device 0 rank 0; missing instances: a, b"
    ]
    assert events == [
        ("register", "devagent"),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", DEVAGENT_CONTROL_INTERVAL_S),
    ]


def test_attention_devagent_publishes_models_as_they_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    agent = cast("atn_module.AtnDevagent", create_devagent(config, cuda_device=0))
    events: list[tuple[object, ...]] = []

    registered_instance_ids = ["a"]

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [
                InstanceRegistration.model_validate(instance_registration_view(instance_id=instance_id, rank=0))
                for instance_id in registered_instance_ids
            ]

        def upsert_devagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[DevagentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [binding.model_dump(mode="json") for binding in bindings])))

    def fake_start(handle: TransportArenaHandle) -> None:
        events.append(("start", int(handle.handle[:2], 16)))

    create_count = 0

    def fake_create(*args: object) -> TransportArenaHandle:
        nonlocal create_count
        handle = TransportArenaHandle(handle=f"{create_count:02x}" * 64)
        events.append(("create", create_count))
        create_count += 1
        return handle

    patch_native_devagent_ops(
        monkeypatch,
        create=fake_create,
        launch=fake_start,
    )

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    agent.state = AtnRegistered()
    for attempt in range(5):
        agent.advance_state()
        if sum(event[0] == "arenas" for event in events) == 1:
            break
    assert events == [
        ("create", 0),
        ("start", 0),
        ("arenas", (0, [{"instance_id": "a", "rank": 0, "handle": transport_arena(0)}])),
    ]

    registered_instance_ids.append("b")
    for attempt in range(5):
        agent.advance_state()
        if sum(event[0] == "arenas" for event in events) == 2:
            break
    assert events[-3:] == [
        ("create", 1),
        ("start", 1),
        ("arenas", (0, [{"instance_id": "b", "rank": 0, "handle": transport_arena(1)}])),
    ]
