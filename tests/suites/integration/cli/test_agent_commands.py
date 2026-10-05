from __future__ import annotations

import pytest

import xpool.cli.subcommands.atnagent
import xpool.cli.subcommands.ffnagent
from xpool.cli import main
from xtest.harness.support.config import reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.mark.parametrize("command", ["atnagent", "ffnagent"])
def test_agent_run_requires_device_device(monkeypatch, capsys, command: str) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    assert main([command]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--device" in captured.err


@pytest.mark.parametrize(
    ("command", "device"),
    [("atnagent", 0), ("ffnagent", 1)],
)
def test_agent_run_dispatches_selected_role(monkeypatch, command: str, device: int) -> None:
    calls: list[int] = []

    class FakeAgent:
        def __init__(self, *, device: int) -> None:
            calls.append(device)

        def run(self) -> None:
            return None

    if command == "atnagent":
        monkeypatch.setattr(xpool.cli.subcommands.atnagent, "AtnAgent", FakeAgent)
    else:
        monkeypatch.setattr(xpool.cli.subcommands.ffnagent, "FfnAgent", FakeAgent)

    assert main([command, "--config", "configs/xpool.example.toml", "--device", str(device)]) == 0

    assert calls == [device]
