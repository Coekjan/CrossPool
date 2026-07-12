from __future__ import annotations

from tests.harness.runtime.agent import reset_agent_runtime
from tests.harness.runtime.atnagent import (
    ProcessHeartbeat,
    XpoolClientError,
    XpoolConfig,
    pytest,
    wait_until,
    wait_until_raise,
)
from xpool.abi import ABI_VERSION
from xpool.config import init_global_config
from xpool.runtime.agent import AgentError, AgentHeartbeat
from xpool.service.client import XpoolDaemonError
from xpool.service.wire import HeartbeatResponse

pytestmark = pytest.mark.usefixtures(reset_agent_runtime.__name__)


def test_heartbeat_reports_missing_registration() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    init_global_config(config=config)
    events: list[tuple[object, ...]] = []

    def sender(cuda_device: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        events.append(("heartbeat", cuda_device))
        raise XpoolDaemonError("not_ready", "registration missing")

    worker = AgentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
        sender=sender,
    )
    worker.start()
    assert wait_until(worker.consume_registration_missing)
    assert wait_until(lambda: worker.thread is not None and not worker.thread.is_alive())
    worker.stop()
    assert worker.thread is None
    worker.start()
    assert wait_until(worker.consume_registration_missing)
    worker.close()

    assert events == [("heartbeat", 0), ("heartbeat", 0)]


def test_heartbeat_surfaces_unrecoverable_daemon_error() -> None:
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )

    def sender(cuda_device: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        raise XpoolDaemonError("conflict", "pid mismatch")

    worker = AgentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
        sender=sender,
    )
    worker.start()
    with pytest.raises(AgentError, match="unrecoverable daemon error"):
        wait_until_raise(worker.raise_if_failed)
    worker.close()


def test_heartbeat_retries_recoverable_client_errors() -> None:
    init_global_config(
        config=XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )
    )
    heartbeat_count = 0

    def sender(cuda_device: int, heartbeat: ProcessHeartbeat) -> HeartbeatResponse:
        nonlocal heartbeat_count
        heartbeat_count += 1
        if heartbeat_count == 1:
            raise XpoolClientError("transport", "daemon unavailable")
        return HeartbeatResponse(warnings=[])

    worker = AgentHeartbeat(
        cuda_device=0,
        heartbeat=ProcessHeartbeat(abi_version=ABI_VERSION, pid=1),
        sender=sender,
        interval_s=0.01,
    )
    worker.start()
    assert wait_until(lambda: heartbeat_count >= 2)
    worker.close()
    worker.raise_if_failed()
