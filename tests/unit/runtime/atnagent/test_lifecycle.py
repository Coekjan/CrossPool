from __future__ import annotations

import xpool.runtime.agent as common_module
from tests.harness.runtime.agent import reset_agent_runtime
from tests.harness.runtime.atnagent import (
    XpoolClientError,
    XpoolConfig,
    create_atnagent,
    health_client_class,
    pytest,
    reset_atnagent_runtime,
)
from xpool.abi import RuntimeRole
from xpool.config import init_global_config
from xpool.runtime.agent import AgentError

pytestmark = pytest.mark.usefixtures(reset_agent_runtime.__name__, reset_atnagent_runtime.__name__)


def test_atnagent_initializes_native_runtime_during_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        common_module.bootstrap,
        "init",
        lambda cuda_device, role: events.append(("init", cuda_device, role)),
    )
    monkeypatch.setattr(common_module.devkit, "install", lambda: events.append(("devkit",)))

    create_atnagent(config, cuda_device=0)

    assert events == [("init", 0, RuntimeRole.ATNAGENT), ("devkit",)]


def test_cuda_device_selection_rejects_unknown_device() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")

    with pytest.raises(AgentError, match="CUDA device 9 has no local instance-rank arenas"):
        create_atnagent(config, cuda_device=9)


def test_atnagent_stays_resident_with_heartbeats(monkeypatch) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    events: list[str] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_atnagent(self, registration: object) -> None:
            events.append("register")

        def heartbeat_atnagent(self, cuda_device: int, heartbeat: object) -> None:
            events.append("heartbeat")

        def list_instances(self) -> list[object]:
            events.append("list_instances")
            return []

    sleep_count = 0

    def stop_after_second_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        events.append("sleep")
        if sleep_count == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("xpool.runtime.agent.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr("xpool.runtime.atnagent.time.sleep", stop_after_second_sleep)

    create_atnagent(config, cuda_device=0).run()

    assert events == ["register", "heartbeat", "sleep", "heartbeat", "list_instances", "sleep"]


def test_atnagent_construction_rejects_unhealthy_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    monkeypatch.setattr(common_module, "XpoolClient", health_client_class(False))

    with pytest.raises(XpoolClientError, match="daemon health check failed"):
        create_atnagent(config, cuda_device=0)
