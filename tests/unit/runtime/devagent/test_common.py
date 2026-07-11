from __future__ import annotations

from tests.harness.runtime.devagent import (
    DevagentError,
    ProcessHeartbeat,
    XpoolClientError,
    XpoolConfig,
    XpoolDaemonError,
    common_module,
    create_devagent,
    pytest,
    wait_until,
    wait_until_raise,
)
from xpool.abi import ABI_VERSION, RuntimeRole
from xpool.config import init_global_config
from xpool.runtime.devagent.common import DevagentHeartbeat


def test_devagent_initializes_native_runtime_during_construction(monkeypatch: pytest.MonkeyPatch) -> None:
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

    create_devagent(config, cuda_device=0)

    assert events == [("init", 0, RuntimeRole.DEVAGENT), ("devkit",)]


def test_devagent_heartbeat_worker_reports_missing_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            events.append(("client", "open"))

        def close(self) -> None:
            events.append(("client", "close"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: ProcessHeartbeat) -> None:
            events.append(("heartbeat", cuda_device))
            raise XpoolDaemonError("not_ready", "registration missing")

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    worker = DevagentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
    )

    worker.start()
    assert wait_until(worker.consume_registration_missing)
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())
    worker.stop()
    assert worker.thread is None

    worker.start()
    assert wait_until(worker.consume_registration_missing)
    worker.close()

    assert events == [("client", "open"), ("heartbeat", 0), ("heartbeat", 0), ("client", "close")]


def test_devagent_heartbeat_worker_surfaces_unrecoverable_daemon_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            events.append(("client", "open"))

        def close(self) -> None:
            events.append(("client", "close"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: ProcessHeartbeat) -> None:
            events.append(("heartbeat", cuda_device))
            raise XpoolDaemonError("conflict", "pid mismatch")

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    worker = DevagentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
    )

    worker.start()
    with pytest.raises(DevagentError, match="unrecoverable daemon error"):
        wait_until_raise(worker.raise_if_failed)
    worker.close()

    assert events == [("client", "open"), ("heartbeat", 0), ("client", "close")]


def test_devagent_heartbeat_worker_retries_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)
    events: list[tuple[object, ...]] = []

    class FakeXpoolClient:
        heartbeat_count = 0

        def __init__(self) -> None:
            events.append(("client", "open"))

        def close(self) -> None:
            events.append(("client", "close"))

        def heartbeat_devagent(self, cuda_device: int, heartbeat: ProcessHeartbeat) -> None:
            FakeXpoolClient.heartbeat_count += 1
            events.append(("heartbeat", cuda_device, FakeXpoolClient.heartbeat_count))
            if FakeXpoolClient.heartbeat_count == 1:
                raise XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    worker = DevagentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
        interval_s=0.01,
    )

    worker.start()
    assert wait_until(lambda: FakeXpoolClient.heartbeat_count >= 2)
    worker.close()
    worker.raise_if_failed()

    assert events == [
        ("client", "open"),
        ("heartbeat", 0, 1),
        ("heartbeat", 0, 2),
        ("client", "close"),
    ]


def test_cuda_device_selection_rejects_unknown_device() -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")

    with pytest.raises(DevagentError, match="unknown devagent CUDA device 9"):
        create_devagent(config, cuda_device=9)


def test_devagent_stays_resident_with_heartbeats(monkeypatch) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    events: list[str] = []

    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def register_devagent(self, registration: object) -> None:
            events.append("register")

        def heartbeat_devagent(self, cuda_device: int, heartbeat: object) -> None:
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

    monkeypatch.setattr("xpool.runtime.devagent.common.XpoolClient", FakeXpoolClient)
    monkeypatch.setattr("xpool.runtime.devagent.atn.time.sleep", stop_after_second_sleep)

    create_devagent(config, cuda_device=0).run()

    assert events == ["register", "heartbeat", "sleep", "heartbeat", "list_instances", "sleep"]
