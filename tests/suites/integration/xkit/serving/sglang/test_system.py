from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import httpx
import pytest

import xkit.network
import xkit.serving.cluster
import xkit.serving.sglang.endpoints
import xkit.serving.sglang.system
from xkit.network import TcpEndpointConflict, TcpEndpointReservation, TcpEndpointUnreachable, TcpPortSpace
from xkit.process import OwnedProcessGroup
from xkit.serving.cluster import XpoolCluster
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.config import XpoolConfig
from xpool.model import ModelId
from xpool.utils.sighandler import defer_signal_exceptions


@pytest.mark.parametrize("startup", ["success", "failure", "conflict", "deployment-failure"])
def test_system_complete_startup_or_partial_rollback(
    startup: Literal["success", "failure", "conflict", "deployment-failure"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = serving_launch(tmp_path)
    events: list[str] = []
    leases: list[SglangEndpointFamilyLease] = []
    acquire = SglangEndpointFamilyLease.acquire
    deployment_failure = RuntimeError("deployment failed after verified cleanup")

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

    def cluster_start(cluster: XpoolCluster, endpoint: TcpEndpointReservation, **kwargs: object) -> None:
        cluster_launch = cluster.launch
        assert kwargs["startup_deadline"] is not None
        assert kwargs["enclosing_scope"] is system.task_scope
        assert cluster_launch.config.daemon.port == endpoint.port
        assert cluster_launch.environment["SGLANG_PLUGINS"] == "other-plugin,xpool"
        assert cluster_launch.environment["HF_HUB_OFFLINE"] == "1"
        assert cluster_launch.environment["TRANSFORMERS_OFFLINE"] == "1"
        assert launch.environment == {"SGLANG_PLUGINS": "other-plugin,xpool", "HF_HUB_OFFLINE": "0"}
        endpoint.release_for_spawn()
        events.append("cluster-start")

        def close(**kwargs: object) -> None:
            cluster.client.close()
            cluster.closed = True
            events.append("cluster-close")
            if startup != "success":
                raise deployment_failure

        monkeypatch.setattr(cluster, "close", close)
        monkeypatch.setattr(cluster, "diagnostics", lambda: "")

    def server_start(
        cls: type[SglangServerProcess],
        *,
        model: SglangLaunchModel,
        endpoint: SglangEndpointFamilyLease,
        **kwargs: object,
    ) -> SglangServerProcess:
        endpoint.release_tcp_for_spawn()
        events.append(f"start-{model.model_id}")
        if startup in ("failure", "conflict") and model.model_id == ModelId("test/b"):
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
                close=lambda **kwargs: events.append(f"close-{model.model_id}"),
                diagnostics=lambda: "",
                url=lambda: f"http://{endpoint.family.host}:{endpoint.family.http_port}",
            ),
        )

    monkeypatch.setattr(XpoolCluster, "start", cluster_start)
    monkeypatch.setattr(SglangServerProcess, "start", classmethod(server_start))
    monkeypatch.setattr(XpoolServingSystem, "wait_for_readiness", lambda self, deadline: events.append("ready"))
    with ExitStack() as foreign:
        system = XpoolServingSystem()
        if startup == "conflict":
            with pytest.raises(TcpEndpointConflict) as error:
                system.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert error.value.addresses == ((leases[-1].family.host, leases[-1].family.nccl_port),)
            assert isinstance(error.value.__cause__, RuntimeError)
            assert "second server failed" in str(error.value.__cause__)
            assert events == ["cluster-start", "start-test/a", "start-test/b", "close-test/a", "cluster-close"]
        elif startup == "failure":
            with pytest.raises(RuntimeError, match="second server failed"):
                system.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert events == ["cluster-start", "start-test/a", "start-test/b", "close-test/a", "cluster-close"]
        else:
            system.start(launch, workdir=tmp_path / "run", startup_timeout_seconds=10)
            assert tuple(endpoint.model_id for endpoint in system.endpoints) == (ModelId("test/a"), ModelId("test/b"))
            system.check_alive()
            if startup == "deployment-failure":
                listener = foreign.enter_context(socket.socket())
                listener.bind((leases[-1].family.host, leases[-1].family.nccl_port))
                listener.listen()
                with pytest.raises(RuntimeError) as error:
                    system.close()
                assert error.value is deployment_failure
            else:
                system.close()
                system.close()
            assert events[:4] == ["cluster-start", "start-test/a", "start-test/b", "ready"]
            assert set(events[4:6]) == {"close-test/a", "close-test/b"}
            assert events[-1] == "cluster-close"
        assert system.closed
    assert all(lease.closed and all(item.listener is None for item in lease.tcp_reservations) for lease in leases)


def test_startup_allocation_deadline_rolls_back_partial_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    listeners: list[socket.socket] = []
    create_listener = xkit.network.create_qualified_tcp_listener
    # Keep the real task-control thread and stdlib waits on their actual clock.
    local_clock = SimpleNamespace(monotonic=lambda: clock[0])
    for module in (xkit.network, xkit.serving.sglang.endpoints, xkit.serving.sglang.system):
        monkeypatch.setattr(module, "time", local_clock)

    def probe(address: tuple[str, int], *, deadline: float | None = None) -> socket.socket:
        assert deadline == 1.0
        if len(listeners) == 2:
            clock[0] = 2.0
            raise TcpEndpointUnreachable(address)
        listener = create_listener(address, deadline=deadline)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(xkit.network, "create_qualified_tcp_listener", probe)
    system = XpoolServingSystem()
    with pytest.raises(TimeoutError, match="startup deadline"):
        system.start(serving_launch(tmp_path), workdir=tmp_path / "run", startup_timeout_seconds=1.0)
    assert len(listeners) == 2
    assert all(listener.fileno() == -1 for listener in listeners)


@pytest.mark.parametrize("phase", ["publication", "readiness"])
def test_async_startup_cancellation_joins_created_process_before_releasing_endpoints(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = XpoolServingSystem()
    entered = threading.Event()
    resume = threading.Event()
    ready = tmp_path / "daemon-ready"
    program = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
Path(sys.argv[1]).touch()
while True:
    time.sleep(0.01)
"""

    def spawn(name: str, arguments: list[str], **kwargs: object) -> OwnedProcessGroup:
        assert name == "daemon"
        owner = OwnedProcessGroup.spawn_logged(
            name,
            [sys.executable, "-c", program, str(ready)],
            cwd=tmp_path,
            env={},
            log_path=tmp_path / "daemon.log",
        )
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert owner.process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return owner

    def start(cluster: XpoolCluster, endpoint: TcpEndpointReservation, **kwargs: object) -> None:
        endpoint.release_for_spawn()
        with defer_signal_exceptions():
            cluster.processes.append(spawn("daemon", []))
        entered.set()
        assert resume.wait(5)

    def get(client: httpx.Client, path: str, **kwargs: object) -> httpx.Response:
        assert path == "/health"
        entered.set()
        assert resume.wait(5)
        return httpx.Response(503, request=httpx.Request("GET", f"{client.base_url}health"))

    if phase == "publication":
        monkeypatch.setattr(XpoolCluster, "start", start)
    else:
        monkeypatch.setattr(xkit.serving.cluster, "spawn_process", spawn)
        monkeypatch.setattr(httpx.Client, "get", get)

    async def run() -> None:
        startup = asyncio.create_task(
            system.start_async(serving_launch(tmp_path), workdir=tmp_path / "run", startup_timeout_seconds=10)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            startup.cancel()
            deadline = time.monotonic() + 5
            while system.cleanup_future is None:
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            cleanup = system.cleanup_future
            cleanup_deadline = system.cleanup_deadline
            assert system.startup_future is not None
            assert not system.startup_future.done()
            assert not cleanup.done()
            assert not startup.done()
            assert system.cluster is not None
            process = system.cluster.processes[0].process
            assert process.poll() is None
            assert all(not endpoint.closed for endpoint in system.server_endpoints)
            joiner = asyncio.create_task(system.close_async())
            await asyncio.sleep(0)
            joiner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await joiner
            assert system.cleanup_future is cleanup
            assert system.cleanup_deadline == cleanup_deadline
            assert not cleanup.done()
            resume.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(startup, 5)
            await system.close_async()
            assert system.cleanup_future is cleanup
            assert system.cleanup_deadline == cleanup_deadline
            assert system.cluster.cleanup_deadline == cleanup_deadline
            assert system.closed
            assert process.returncode == 0
            assert all(endpoint.closed for endpoint in system.server_endpoints)
        finally:
            resume.set()
            await system.close_async()
            await asyncio.gather(startup, return_exceptions=True)

    asyncio.run(run())


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
        {"SGLANG_PLUGINS": "other-plugin,xpool", "HF_HUB_OFFLINE": "0"},
        tmp_path,
        tuple(SglangLaunchModel(ModelId(id), SglangGraphMode.EAGER) for id in ("test/a", "test/b")),
    )
