from __future__ import annotations

import errno
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

import tests.harness.native.cluster
from tests.harness.native.cluster import XpoolCluster
from tests.harness.runner.network import TcpEndpointConflict, TcpEndpointReservation, TcpPortSpace
from tests.harness.runner.process import OwnedProcessGroup
from tests.harness.sglang.serving.launch import E2eLaunch
from xpool.config import XpoolConfig


class FakeProcess:
    """Minimal live Popen surface for cluster ownership tests."""

    next_pid = 12000

    def __init__(self) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode: int | None = None
        self.stdout = None
        self.stderr = None

    def poll(self) -> int | None:
        """Return the synthetic process status."""

        return self.returncode


@dataclass(slots=True)
class FakeOwnedProcessGroup(OwnedProcessGroup):
    """Record cleanup ordering without launching a subprocess."""

    events: list[str] = field(default_factory=list)

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.name}")
        cast(FakeProcess, self.process).returncode = 0

    def close(self) -> None:
        self.events.append(f"close:{self.name}")

    def tail(self) -> str:
        return f"tail:{self.name}"


class FakeResponse:
    """Small HTTP response carrying one static payload."""

    def __init__(self, payload: dict[str, object] | None = None, *, success: bool = True) -> None:
        self.payload = payload
        self.is_success = success
        self.status_code = 200 if success else 503
        self.text = "" if payload is None else str(payload)

    def json(self) -> dict[str, object]:
        """Return the configured response payload."""

        if self.payload is None:
            raise ValueError("response has no JSON payload")
        return self.payload


class FakeHttpClient:
    """Serve daemon health and readiness from fixed test data."""

    def __init__(self, readiness: dict[str, object], *, healthy: bool = True) -> None:
        self.readiness = readiness
        self.healthy = healthy
        self.closed = False

    def get(self, path: str) -> FakeResponse:
        """Return one health or readiness response."""

        if path == "/health":
            return FakeResponse(success=self.healthy)
        if path == "/ready":
            return FakeResponse(self.readiness)
        raise AssertionError(f"unexpected path {path}")

    def close(self) -> None:
        """Record client closure."""

        self.closed = True


def test_cluster_starts_complete_agent_set_and_closes_in_role_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    launch = cluster_launch(tmp_path, daemon_port=endpoint.port)
    events: list[str] = []
    client = FakeHttpClient(online_readiness())
    launches: list[tuple[str, list[str]]] = []
    client_options: dict[str, object] = {}

    def spawn(
        cls: type[OwnedProcessGroup],
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> OwnedProcessGroup:
        launches.append((name, command))
        return FakeOwnedProcessGroup(
            name=name,
            process=cast(subprocess.Popen[str], FakeProcess()),
            log_path=log_path,
            events=events,
        )

    def wait(process: subprocess.Popen[str], timeout_seconds: float) -> bool:
        cast(FakeProcess, process).returncode = 0
        return True

    monkeypatch.setattr(OwnedProcessGroup, "spawn_logged", classmethod(spawn))

    def create_client(**kwargs: object) -> FakeHttpClient:
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(tests.harness.native.cluster.httpx, "Client", create_client)
    monkeypatch.setattr(tests.harness.native.cluster, "signal_process_group", lambda *args: None)
    monkeypatch.setattr(tests.harness.native.cluster, "wait_for_process_group", wait)

    cluster = XpoolCluster.start(launch, endpoint)
    cluster.close()

    assert [name for name, command in launches] == ["daemon", "atnagent-0", "ffnagent-1", "ffnagent-2"]
    assert launches[0][1] == ["xpool", "daemon", "serve"]
    assert launches[1][1][-3:] == ["atnagent", "--cuda-device", "0"]
    assert launches[2][1][-3:] == ["ffnagent", "--cuda-device", "1"]
    assert events == ["close:daemon", "close:atnagent-0", "close:ffnagent-1", "close:ffnagent-2"]
    assert cluster.daemon_startup_seconds >= 0
    assert client_options["timeout"] == tests.harness.native.cluster.CONTROL_PLANE_HTTP_TIMEOUT_SECONDS
    assert cast(float, client_options["timeout"]) > tests.harness.native.cluster.POLL_INTERVAL_SECONDS
    assert client.closed
    endpoint.close()


def test_cluster_classifies_port_conflict_only_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    launch = cluster_launch(tmp_path, daemon_port=endpoint.port)
    events: list[str] = []
    client = FakeHttpClient(online_readiness(), healthy=False)

    def spawn(*args: object, **kwargs: object) -> OwnedProcessGroup:
        return FakeOwnedProcessGroup(
            name="daemon",
            process=cast(subprocess.Popen[str], FakeProcess()),
            log_path=tmp_path / "daemon.log",
            events=events,
        )

    def reacquire(reservation: TcpEndpointReservation) -> None:
        assert events == ["terminate:daemon", "close:daemon"]
        raise OSError(errno.EADDRINUSE, "address in use")

    monkeypatch.setattr(OwnedProcessGroup, "spawn_logged", spawn)
    monkeypatch.setattr(tests.harness.native.cluster.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(tests.harness.native.cluster, "DAEMON_STARTUP_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(TcpEndpointReservation, "reacquire", reacquire)

    with pytest.raises(TcpEndpointConflict) as error:
        XpoolCluster.start(launch, endpoint)

    assert error.value.addresses == ((endpoint.host, endpoint.port),)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert client.closed
    endpoint.close()


def test_cluster_preserves_startup_failure_when_endpoint_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the startup failure when endpoint classification itself is unavailable."""

    endpoint = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    launch = cluster_launch(tmp_path, daemon_port=endpoint.port)
    client = FakeHttpClient(online_readiness(), healthy=False)

    def spawn(*args: object, **kwargs: object) -> OwnedProcessGroup:
        return FakeOwnedProcessGroup(
            name="daemon",
            process=cast(subprocess.Popen[str], FakeProcess()),
            log_path=tmp_path / "daemon.log",
        )

    def fail_endpoint_probe(reservation: TcpEndpointReservation) -> None:
        raise OSError(errno.EACCES, "endpoint inspection denied")

    monkeypatch.setattr(OwnedProcessGroup, "spawn_logged", spawn)
    monkeypatch.setattr(tests.harness.native.cluster.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(tests.harness.native.cluster, "DAEMON_STARTUP_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(TcpEndpointReservation, "reacquire", fail_endpoint_probe)

    with pytest.raises(OSError, match="endpoint inspection denied") as error:
        XpoolCluster.start(launch, endpoint)

    assert error.value.errno == errno.EACCES
    assert isinstance(error.value.__cause__, RuntimeError)
    assert client.closed
    endpoint.close()


def cluster_launch(tmp_path: Path, *, daemon_port: int = 19810) -> E2eLaunch:
    """Build one fully resolved launch without model-weight I/O."""

    config_path = tmp_path / "xpool.toml"
    config_path.write_text("", encoding="utf-8")
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": daemon_port},
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [{"id": "organization/model"}],
        },
        env={},
    )
    return E2eLaunch(
        case_id="cluster",
        models=(),
        config=config,
        config_path=config_path,
        environment=MappingProxyType({"XPOOL_CONFIG": str(config_path)}),
        observer_outdir=tmp_path / "observers",
    )


def online_readiness() -> dict[str, object]:
    """Return a valid pre-Instance readiness payload with every Agent online."""

    return {
        "ready": False,
        "generation": None,
        "fabric_phase": None,
        "fabric_invocation_failure": None,
        "fabric_owner_failure": None,
        "fabric_control_failure": None,
        "transport_ready": False,
        "instances_initialized": False,
        "mps_status": "online",
        "cuda_devices": [0, 1, 2],
        "atnagents": [{"pid": 12001, "status": "online", "cuda_device": 0}],
        "ffnagents": [
            {"pid": 12002, "status": "online", "cuda_device": 1},
            {"pid": 12003, "status": "online", "cuda_device": 2},
        ],
        "instances": [],
    }
