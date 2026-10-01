from __future__ import annotations

import socket
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

import xkit.network
from xkit.network import TcpEndpointConflict, TcpEndpointReservation, TcpEndpointUnreachable, TcpPortSpace
from xkit.serving.cluster import XpoolCluster, XpoolClusterLaunch
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.config import XpoolConfig
from xpool.model import ModelId


@pytest.mark.parametrize("startup", ["success", "failure", "conflict"])
def test_system_complete_startup_or_partial_rollback(
    startup: Literal["success", "failure", "conflict"], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = serving_launch(tmp_path)
    events: list[str] = []
    leases: list[SglangEndpointFamilyLease] = []
    acquire = SglangEndpointFamilyLease.acquire

    def acquire_family(
        cls: type[SglangEndpointFamilyLease],
        host: str,
        *,
        dp_size: int,
        port_space: TcpPortSpace,
        deadline: float | None = None,
    ) -> SglangEndpointFamilyLease:
        lease = acquire(host, dp_size=dp_size, port_space=port_space, deadline=deadline)
        leases.append(lease)
        return lease

    monkeypatch.setattr(SglangEndpointFamilyLease, "acquire", classmethod(acquire_family))

    def cluster_start(
        cls: type[XpoolCluster], cluster_launch: XpoolClusterLaunch, endpoint: TcpEndpointReservation, **kwargs: object
    ) -> XpoolCluster:
        assert kwargs["startup_deadline"] is not None
        assert cluster_launch.config.daemon.port == endpoint.port
        assert cluster_launch.environment["SGLANG_PLUGINS"] == "xpool"
        assert cluster_launch.environment["HF_HUB_OFFLINE"] == "1"
        assert cluster_launch.environment["TRANSFORMERS_OFFLINE"] == "1"
        assert launch.environment == {"SGLANG_PLUGINS": "other-plugin", "HF_HUB_OFFLINE": "0"}
        endpoint.release_for_spawn()
        events.append("cluster-start")
        return cast(
            XpoolCluster,
            SimpleNamespace(processes=[], close=lambda: events.append("cluster-close"), diagnostics=lambda: ""),
        )

    def server_start(
        cls: type[SglangServerProcess],
        *,
        model: SglangLaunchModel,
        endpoint: SglangEndpointFamilyLease,
        **kwargs: object,
    ) -> SglangServerProcess:
        endpoint.release_tcp_for_spawn()
        events.append(f"start-{model.model_id}")
        if startup != "success" and model.model_id == ModelId("test/b"):
            if startup == "conflict":
                listener = foreign.enter_context(socket.socket())
                listener.bind((endpoint.family.host, endpoint.family.nccl_port))
                listener.listen()
            raise RuntimeError("second server failed")
        owner = SimpleNamespace(name=model.model_id, process=SimpleNamespace(poll=lambda: None))
        return cast(
            SglangServerProcess,
            SimpleNamespace(
                model=model,
                owner=owner,
                close=lambda: events.append(f"close-{model.model_id}"),
                diagnostics=lambda: "",
                url=lambda: f"http://{endpoint.family.host}:{endpoint.family.http_port}",
            ),
        )

    monkeypatch.setattr(XpoolCluster, "start", classmethod(cluster_start))
    monkeypatch.setattr(SglangServerProcess, "start", classmethod(server_start))
    monkeypatch.setattr(XpoolServingSystem, "wait_for_readiness", lambda self, deadline: events.append("ready"))
    with ExitStack() as foreign:
        if startup == "conflict":
            with pytest.raises(TcpEndpointConflict) as error:
                XpoolServingSystem.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert error.value.addresses == ((leases[-1].family.host, leases[-1].family.nccl_port),)
            assert isinstance(error.value.__cause__, RuntimeError)
            assert "second server failed" in str(error.value.__cause__)
            assert events == ["cluster-start", "start-test/a", "start-test/b", "close-test/a", "cluster-close"]
        elif startup == "failure":
            with pytest.raises(RuntimeError, match="second server failed"):
                XpoolServingSystem.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert events == ["cluster-start", "start-test/a", "start-test/b", "close-test/a", "cluster-close"]
        else:
            system = XpoolServingSystem.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert tuple(endpoint.model_id for endpoint in system.endpoints) == (ModelId("test/a"), ModelId("test/b"))
            system.check_alive()
            system.close()
            system.close()
            assert events[:4] == ["cluster-start", "start-test/a", "start-test/b", "ready"]
            assert set(events[4:6]) == {"close-test/a", "close-test/b"}
            assert events[-1] == "cluster-close"
    assert all(lease.closed and all(item.listener is None for item in lease.tcp_reservations) for lease in leases)


def test_startup_allocation_deadline_rolls_back_partial_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    listeners: list[socket.socket] = []
    create_listener = xkit.network.create_qualified_tcp_listener
    monkeypatch.setattr(xkit.network.time, "monotonic", lambda: clock[0])

    def probe(address: tuple[str, int], *, deadline: float | None = None) -> socket.socket:
        assert deadline == 1.0
        if len(listeners) == 2:
            clock[0] = 2.0
            raise TcpEndpointUnreachable(address)
        listener = create_listener(address, deadline=deadline)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(xkit.network, "create_qualified_tcp_listener", probe)
    with pytest.raises(TimeoutError, match="startup deadline"):
        XpoolServingSystem.start(serving_launch(tmp_path), workdir=tmp_path / "run", startup_timeout_seconds=1.0)
    assert len(listeners) == 2
    assert all(listener.fileno() == -1 for listener in listeners)


def serving_launch(tmp_path: Path) -> ServingLaunch:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "test/a"}, {"id": "test/b"}],
        },
        env={},
    )
    return ServingLaunch(
        config,
        {"SGLANG_PLUGINS": "other-plugin", "HF_HUB_OFFLINE": "0"},
        tmp_path,
        tuple(SglangLaunchModel(ModelId(id), SglangGraphMode.EAGER) for id in ("test/a", "test/b")),
    )
