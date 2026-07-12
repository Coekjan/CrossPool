from __future__ import annotations

import logging
from typing import cast

from tests.harness.runtime.agent import reset_agent_runtime
from tests.harness.runtime.atnagent import (
    InstanceRegistration,
    TransportArenaHandle,
    XpoolClientError,
    XpoolConfig,
    atn_module,
    create_atnagent,
    instance_registration_view,
    patch_native_atnagent_ops,
    pytest,
    reset_atnagent_runtime,
    transport_arena,
)
from xpool.abi import ABI_VERSION
from xpool.runtime.agent import AGENT_CONTROL_INTERVAL_S, AGENT_SHUTDOWN_POLL_INTERVAL_S, AgentError
from xpool.service.client import XpoolClient, XpoolDaemonError
from xpool.service.wire import (
    AtnAgentRegistration,
    AtnAgentTransportArenaBinding,
    AtnAgentTransportArenaDrainResponse,
)

pytestmark = pytest.mark.usefixtures(reset_agent_runtime.__name__, reset_atnagent_runtime.__name__)


def test_non_main_attention_atnagent_publishes_local_arenas(monkeypatch) -> None:
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append(("register", registration.model_dump(mode="json")))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=1))]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_fourth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    patch_native_atnagent_ops(monkeypatch, events=events)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_fourth_sleep)

    create_atnagent(config, cuda_device=1).run()

    registration_payload = cast("dict[str, object]", events[0][1])
    assert isinstance(registration_payload, dict)
    assert "create_time" not in registration_payload
    assert "nvshmem_rank" not in registration_payload
    assert registration_payload["abi_version"] == ABI_VERSION
    assert events == [
        ("register", registration_payload),
        ("heartbeat", 1),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 1),
        ("arenas", (1, [{"instance_id": "m", "rank": 1, "handle": transport_arena(1)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("drain", 1),
        ("destroy", 1),
    ]


def test_attention_atnagent_destroys_arenas_after_arenas_drain_retries(
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append(("register", "atnagent"))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            FakeXpoolClient.drain_count += 1
            if FakeXpoolClient.drain_count < 3:
                events.append(("drain-failed", cuda_device))
                raise XpoolClientError("transport", "daemon unavailable")
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def interrupt_fourth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", interrupt_fourth_sleep)
    patch_native_atnagent_ops(monkeypatch, events=events)

    create_atnagent(config, cuda_device=0).run()

    assert events == [
        ("register", "atnagent"),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("drain-failed", 0),
        ("sleep", AGENT_SHUTDOWN_POLL_INTERVAL_S),
        ("drain-failed", 0),
        ("sleep", AGENT_SHUTDOWN_POLL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_atnagent_republishes_arenas_after_daemon_reregister(monkeypatch) -> None:
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append(("register", "atnagent"))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", FakeXpoolClient.heartbeat_count))
            if FakeXpoolClient.heartbeat_count == 5:
                raise XpoolDaemonError("not_ready", "registration missing")

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_sixth_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 6:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_sixth_sleep)
    patch_native_atnagent_ops(monkeypatch, events=events)

    create_atnagent(config, cuda_device=0).run()

    assert events == [
        ("register", "atnagent"),
        ("heartbeat", 1),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 2),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 3),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 4),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 5),
        ("register", "atnagent"),
        ("heartbeat", 6),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 7),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_atnagent_retries_not_ready_arena_publication(monkeypatch) -> None:
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append(("register", "atnagent"))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            if bindings:
                FakeXpoolClient.upsert_count += 1
                if FakeXpoolClient.upsert_count == 1:
                    events.append(("arenas-not-ready", cuda_device))
                    raise XpoolDaemonError("not_ready", "atnagent stale")
                events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))
                return

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_publication(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_publication)
    patch_native_atnagent_ops(monkeypatch, events=events)

    create_atnagent(config, cuda_device=0).run()

    assert events == [
        ("register", "atnagent"),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas-not-ready", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_atnagent_survives_transport_error_during_reregister(monkeypatch) -> None:
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            FakeXpoolClient.register_count += 1
            events.append(("register", FakeXpoolClient.register_count))
            if FakeXpoolClient.register_count == 2:
                raise XpoolClientError("transport", "daemon restarting")

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", FakeXpoolClient.heartbeat_count))
            if FakeXpoolClient.heartbeat_count <= 2:
                raise XpoolDaemonError("not_ready", "registration missing")

        def list_instances(self) -> list[object]:
            if FakeXpoolClient.heartbeat_count < 3:
                return []
            return [InstanceRegistration.model_validate(instance_registration_view(instance_id="m", rank=0))]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_seventh_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 7:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_seventh_sleep)
    patch_native_atnagent_ops(monkeypatch, events=events)

    create_atnagent(config, cuda_device=0).run()

    assert events == [
        ("register", 1),
        ("heartbeat", 1),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("register", 2),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("register", 3),
        ("heartbeat", 2),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("register", 4),
        ("heartbeat", 3),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 4),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 5),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 6),
        ("arenas", (0, [{"instance_id": "m", "rank": 0, "handle": transport_arena(0)}])),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("drain", 0),
        ("destroy", 0),
    ]


def test_attention_atnagent_initial_register_conflict_fails_loud_without_shutdown(
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append("register")
            raise XpoolDaemonError("not_found", "unknown atnagent")

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)

    with pytest.raises(AgentError, match="AtnAgent registration received unrecoverable daemon error"):
        create_atnagent(config, cuda_device=0).run()

    assert events == ["register"]


def test_attention_atnagent_stops_on_daemon_conflict(
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

        def register_atnagent(self, registration: AtnAgentRegistration) -> None:
            events.append(("register", "atnagent"))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))
            raise XpoolDaemonError("conflict", "pid mismatch")

        def list_instances(self) -> list[object]:
            pytest.fail("conflicted atnagent must not reconcile instance arenas")

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)

    with pytest.raises(AgentError, match="unrecoverable daemon error"):
        create_atnagent(config, cuda_device=0).run()

    assert events == [
        ("register", "atnagent"),
        ("heartbeat", 0),
    ]


def test_attention_atnagent_waits_for_local_instance_registrations(
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

        def register_atnagent(self, registration: object) -> None:
            events.append(("register", "atnagent"))

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append(("heartbeat", cuda_device))

        def list_instances(self) -> list[object]:
            return []

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
            *,
            publisher: object,
        ) -> None:
            events.append(("arenas", (cuda_device, [item.model_dump(mode="json") for item in bindings])))

        def drain_atnagent_transport_arenas(
            self,
            cuda_device: int,
            *,
            publisher: object,
        ) -> AtnAgentTransportArenaDrainResponse:
            events.append(("drain", cuda_device))
            return AtnAgentTransportArenaDrainResponse(in_use=[])

    sleep_count = 0

    def stop_after_second_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append(("sleep", delay))
        if sleep_count == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr(atn_module.time, "sleep", stop_after_second_sleep)
    patch_native_atnagent_ops(monkeypatch, events=events)

    with caplog.at_level(logging.INFO, logger="xpool.runtime.agent"):
        create_atnagent(config, cuda_device=0).run()

    assert caplog.messages == [
        "waiting for local instance registrations on CUDA device 0 rank 0; missing instances: a, b"
    ]
    assert events == [
        ("register", "atnagent"),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
        ("heartbeat", 0),
        ("sleep", AGENT_CONTROL_INTERVAL_S),
    ]


def test_attention_atnagent_publishes_models_as_they_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "a", "path": "/models/a"}, {"id": "b", "path": "/models/b"}],
        }
    )
    agent = cast("atn_module.AtnAgent", create_atnagent(config, cuda_device=0))
    events: list[tuple[object, ...]] = []

    registered_instance_ids = ["a"]

    class FakeXpoolClient:
        def list_instances(self) -> list[InstanceRegistration]:
            return [
                InstanceRegistration.model_validate(instance_registration_view(instance_id=instance_id, rank=0))
                for instance_id in registered_instance_ids
            ]

        def upsert_atnagent_transport_arenas(
            self,
            cuda_device: int,
            bindings: list[AtnAgentTransportArenaBinding],
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

    patch_native_atnagent_ops(
        monkeypatch,
        create=fake_create,
        launch=fake_start,
    )

    agent.client = cast("XpoolClient", FakeXpoolClient())
    agent.registered = True
    agent.advance_state()
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
