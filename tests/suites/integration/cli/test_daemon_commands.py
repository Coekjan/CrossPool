from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import xpool.cli.subcommands.atnagent
import xpool.cli.subcommands.daemon
from xpool.cli import main
from xpool.fabric import FabricGenerationId
from xpool.service.client import XpoolClientError
from xpool.service.daemon.control import ControlPlane
from xpool.service.wire import ReadinessSnapshot
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config

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
            "devices": [0, 1],
            "atnagents": [{"pid": 100, "status": "online", "device": 0}],
            "ffnagents": [{"pid": 300, "status": "online", "device": 1}],
            "instances": [
                {
                    "pid": 200 if ready else None,
                    "status": instance_status,
                    "model_id": str(TEST_MODEL_ID),
                    "device": 0,
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
    assert payload["readiness"]["devices"] == [0, 1]
    assert payload["readiness"]["mps_status"] == "online"
    assert payload["readiness"]["atnagents"] == [{"device": 0, "pid": 100, "status": "online"}]
    assert payload["readiness"]["instances"] == [
        {
            "device": 0,
            "model_id": str(TEST_MODEL_ID),
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
    control = ControlPlane()
    control.close()
    failure = xpool.cli.subcommands.daemon.DaemonFailure(control)
    app = SimpleNamespace(state=SimpleNamespace(daemon_failure=failure, control_plane=control))
    calls: list[tuple[object, str, int, bool, object, object]] = []

    class FakeDaemonServer:
        def __init__(self, config, server_failure, server_control) -> None:
            calls.append((config.app, config.host, config.port, config.access_log, server_failure, server_control))

        def run(self) -> None:
            pass

    monkeypatch.setattr(xpool.cli.subcommands.daemon, "create_daemon", lambda: app)
    monkeypatch.setattr(xpool.cli.subcommands.daemon, "DaemonServer", FakeDaemonServer)

    assert main(["daemon", "serve", "--config", "configs/xpool.example.toml"]) == 0

    assert calls == [(app, "127.0.0.1", 9810, False, failure, control)]


def test_atnagent_run_reports_daemon_transport_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FakeAtnAgent:
        def __init__(self, *, device: int) -> None:
            return None

        def run(self) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr(xpool.cli.subcommands.atnagent, "AtnAgent", FakeAtnAgent)

    assert main(["atnagent", "--device", "0"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "daemon unavailable" in captured.err
