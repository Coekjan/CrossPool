from __future__ import annotations

from tests.harness.cli.common import (
    ReadinessScope,
    ReadinessSnapshot,
    client_class,
    json,
    ready_snapshot,
)
from xpool.cli import main
from xpool.cli.subcommands import atnagent as atnagent_cli
from xpool.cli.subcommands import daemon as daemon_cli
from xpool.service.client import XpoolClientError


def test_daemon_check_reports_ready_snapshot(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(daemon_cli, "XpoolClient", client_class(ready_snapshot(ready=True)))

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
            "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "pid": 200,
            "rank": 0,
            "status": "online",
        }
    ]


def test_daemon_check_forwards_repeated_readiness_scopes(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    requested_scopes: list[tuple[ReadinessScope, ...]] = []

    class RecordingXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def readiness(self, scopes: tuple[ReadinessScope, ...] = ()) -> ReadinessSnapshot:
            requested_scopes.append(scopes)
            return ready_snapshot(ready=True)

    monkeypatch.setattr(daemon_cli, "XpoolClient", RecordingXpoolClient)

    assert main(["daemon", "check", "--scope", "atn"]) == 0
    assert requested_scopes == [(ReadinessScope.ATN,)]
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_daemon_check_reports_not_ready_snapshot(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(daemon_cli, "XpoolClient", client_class(ready_snapshot(ready=False)))

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

    monkeypatch.setattr(daemon_cli, "XpoolClient", FailingXpoolClient)

    assert main(["daemon", "check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["readiness"] is None
    assert payload["error"] == "daemon unavailable"
    assert payload["daemon"] == {"host": "127.0.0.1", "port": 9810}


def test_daemon_serve_runs_uvicorn(monkeypatch) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    app = object()
    calls: list[tuple[object, str, int]] = []

    monkeypatch.setattr(daemon_cli, "create_daemon", lambda: app)
    monkeypatch.setattr(
        daemon_cli.uvicorn,
        "run",
        lambda app_arg, *, host, port: calls.append((app_arg, host, port)),
    )

    assert main(["daemon", "serve"]) == 0

    assert calls == [(app, "127.0.0.1", 9810)]


def test_atnagent_run_reports_daemon_transport_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FakeAtnAgent:
        def __init__(self, *, cuda_device: int) -> None:
            return None

        def run(self) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

    monkeypatch.setattr(atnagent_cli, "AtnAgent", FakeAtnAgent)

    assert main(["atnagent", "--cuda-device", "0"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "daemon unavailable" in captured.err
