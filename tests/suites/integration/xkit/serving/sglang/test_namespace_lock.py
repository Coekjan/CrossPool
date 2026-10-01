from __future__ import annotations

import errno
from multiprocessing.connection import Connection
from pathlib import Path

from tests import TEST_CATALOG_PATH

from xkit.child import PythonChildProcess
from xkit.network import TcpPortSpace
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease, reserve_namespace_lock
from xkit.supervisor import prepare_task_supervision


def probe_namespace_lock(connection: Connection, endpoint: tuple[str, int]) -> None:
    try:
        lock = reserve_namespace_lock(*endpoint)
    except OSError as error:
        connection.send(error.errno)
        return
    lock.close()
    connection.send(None)


def test_endpoint_family_lock_coordinates_independent_interpreters(tmp_path: Path) -> None:
    prepare_task_supervision()
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    endpoint = (lease.family.host, lease.family.http_port)
    try:
        occupied = PythonChildProcess.start(
            "occupied-endpoint-lock",
            probe_namespace_lock,
            endpoint,
            log_path=tmp_path / "occupied.log",
            import_paths=(TEST_CATALOG_PATH.resolve().parent.parent,),
        )
        try:
            assert occupied.receive(int, timeout_seconds=5) == errno.EADDRINUSE
            occupied.wait(timeout_seconds=5)
        finally:
            if occupied.process.is_alive():
                PythonChildProcess.terminate_all((occupied,))
            occupied.close()
    finally:
        lease.close()

    available = PythonChildProcess.start(
        "available-endpoint-lock",
        probe_namespace_lock,
        endpoint,
        log_path=tmp_path / "available.log",
        import_paths=(TEST_CATALOG_PATH.resolve().parent.parent,),
    )
    try:
        assert available.receive(type(None), timeout_seconds=5) is None
        available.wait(timeout_seconds=5)
    finally:
        if available.process.is_alive():
            PythonChildProcess.terminate_all((available,))
        available.close()
