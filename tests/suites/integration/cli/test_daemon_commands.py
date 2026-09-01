from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import xpool.cli.subcommands.atnagent
import xpool.cli.subcommands.daemon
from tests.harness.support.config import reset_global_config
from xpool.cli import main
from xpool.fabric import FabricGenerationId
from xpool.service.client import XpoolClientError
from xpool.service.wire import ReadinessSnapshot

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def ready_snapshot(*, ready: bool) -> ReadinessSnapshot:
    instance_status = "online" if ready else "offline"
    generation = FabricGenerationId(high=1, low=1) if ready else None
    return ReadinessSnapshot.model_validate(
        {
            "ready": ready,
            "generation": generation,
            "fabric_phase": "executable" if ready else None,
            "fabric_invocation_failure": None,
            "fabric_owner_failure": None,
            "fabric_control_failure": None,
            "transport_ready": ready,
            "instances_initialized": ready,
            "mps_status": "online" if ready else "offline",
            "cuda_devices": [0, 1],
            "atnagents": [{"pid": 100, "status": "online", "cuda_device": 0}],
            "ffnagents": [{"pid": 300, "status": "online", "cuda_device": 1}],
            "instances": [
                {
                    "pid": 200 if ready else None,
                    "status": instance_status,
                    "instance_id": "test/model",
                    "cuda_device": 0,
                    "rank": 0,
                }
            ],
        }
    )


def client_class(readiness: ReadinessSnapshot) -> type:
    class FakeXpoolClient:
        def close(self) -> None:
            pass

        def readiness(self) -> ReadinessSnapshot:
            return readiness

    return FakeXpoolClient


def test_daemon_check_reports_ready_snapshot(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(xpool.cli.subcommands.daemon, "XpoolClient", client_class(ready_snapshot(ready=True)))

    assert main(["daemon", "check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["error"] is None
    assert payload["daemon"] == {"host": "127.0.0.1", "port": 9810}
    assert payload["readiness"]["cuda_devices"] == [0, 1]
    assert payload["readiness"]["mps_status"] == "online"
    assert payload["readiness"]["atnagents"] == [{"cuda_device": 0, "pid": 100, "status": "online"}]
    assert payload["readiness"]["instances"] == [
        {
            "cuda_device": 0,
            "instance_id": "test/model",
            "pid": 200,
            "rank": 0,
            "status": "online",
        }
    ]


def test_daemon_check_reports_not_ready_snapshot(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(xpool.cli.subcommands.daemon, "XpoolClient", client_class(ready_snapshot(ready=False)))

    assert main(["daemon", "check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["error"] is None
    assert payload["readiness"]["ready"] is False


def test_daemon_check_reports_daemon_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FailingXpoolClient:
        def __init__(self) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr(xpool.cli.subcommands.daemon, "XpoolClient", FailingXpoolClient)

    assert main(["daemon", "check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["readiness"] is None
    assert payload["error"] == "daemon unavailable"
    assert payload["daemon"] == {"host": "127.0.0.1", "port": 9810}


def test_daemon_serve_runs_uvicorn(monkeypatch) -> None:
    failure = xpool.cli.subcommands.daemon.DaemonFailure()
    app = SimpleNamespace(state=SimpleNamespace(daemon_failure=failure))
    calls: list[tuple[object, str, int, object]] = []

    class FakeDaemonServer:
        def __init__(self, config, server_failure) -> None:
            calls.append((config.app, config.host, config.port, server_failure))

        def run(self) -> None:
            pass

    monkeypatch.setattr(xpool.cli.subcommands.daemon, "create_daemon", lambda: app)
    monkeypatch.setattr(xpool.cli.subcommands.daemon, "DaemonServer", FakeDaemonServer)

    assert main(["daemon", "serve", "--config", "configs/xpool.example.toml"]) == 0

    assert calls == [(app, "127.0.0.1", 9810, failure)]


def test_daemon_serve_returns_nonzero_after_watchdog_failure(monkeypatch) -> None:
    failure = xpool.cli.subcommands.daemon.DaemonFailure()
    app = SimpleNamespace(state=SimpleNamespace(daemon_failure=failure))

    class FailingDaemonServer:
        def __init__(self, config, server_failure) -> None:
            assert config.app is app
            assert server_failure is failure

        def run(self) -> None:
            failure.record(RuntimeError("watchdog failed"))

    monkeypatch.setattr(xpool.cli.subcommands.daemon, "create_daemon", lambda: app)
    monkeypatch.setattr(xpool.cli.subcommands.daemon, "DaemonServer", FailingDaemonServer)

    assert main(["daemon", "serve", "--config", "configs/xpool.example.toml"]) == 1


def test_atnagent_run_reports_daemon_transport_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FakeAtnAgent:
        def __init__(self, *, cuda_device: int) -> None:
            return None

        def run(self) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr(xpool.cli.subcommands.atnagent, "AtnAgent", FakeAtnAgent)

    assert main(["atnagent", "--cuda-device", "0"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "daemon unavailable" in captured.err
