from __future__ import annotations

import errno
from types import SimpleNamespace
from typing import cast

import pytest

from xkit.network import TcpEndpointConflict
from xkit.serving.cluster import XpoolCluster
from xkit.serving.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from xkit.serving.sglang.server import SglangServerProcess
from xkit.serving.sglang.system import XpoolServingSystem


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_system_cleanup_releases_endpoints_after_process_failure_or_bind_conflict(cleanup_failure: bool) -> None:
    system = XpoolServingSystem()
    events: list[str] = []
    family = SglangEndpointFamily("127.0.0.1", 20000, 21000, 22000, 1)

    def close_server() -> None:
        events.append("server-close")
        if cleanup_failure:
            raise RuntimeError("process still live")

    system.servers.append(cast(SglangServerProcess, SimpleNamespace(close=close_server)))
    system.cluster = cast(XpoolCluster, SimpleNamespace(close=lambda: events.append("cluster-close")))
    system.server_endpoints.append(
        cast(
            SglangEndpointFamilyLease,
            SimpleNamespace(
                family=family,
                tcp_released=True,
                reacquire_tcp=lambda: events.append("endpoint-inspect") or (family.nccl_port,),
                close=lambda: events.append("endpoint-close"),
            ),
        )
    )
    if cleanup_failure:
        with pytest.raises(RuntimeError, match="process still live"):
            system.close()
        assert events == ["server-close", "cluster-close", "endpoint-close"]
    else:
        with pytest.raises(TcpEndpointConflict) as error:
            system.close()
        assert error.value.addresses == ((family.host, family.nccl_port),)
        assert events == ["server-close", "cluster-close", "endpoint-inspect", "endpoint-close"]


def test_system_rejects_process_exit_and_preserves_inspection_error() -> None:
    system = XpoolServingSystem()
    owner = SimpleNamespace(name="daemon", process=SimpleNamespace(poll=lambda: 1))
    system.cluster = cast(XpoolCluster, SimpleNamespace(processes=[owner], close=lambda: None))
    with pytest.raises(RuntimeError, match="daemon exited"):
        system.check_alive()
    events: list[str] = []

    def inspect() -> tuple[int, ...]:
        raise OSError(errno.EACCES, "inspection denied")

    system.server_endpoints.append(
        cast(
            SglangEndpointFamilyLease,
            SimpleNamespace(tcp_released=True, reacquire_tcp=inspect, close=lambda: events.append("endpoint-close")),
        )
    )
    with pytest.raises(OSError) as error:
        system.close()
    assert error.value.errno == errno.EACCES
    assert events == ["endpoint-close"]
