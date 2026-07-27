from __future__ import annotations

import pytest

import xpool.cli.subcommands.atnagent
import xpool.cli.subcommands.ffnagent
from tests.harness.pytest_plugin import reset_global_config
from xpool.cli import main

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.mark.parametrize("command", ["atnagent", "ffnagent"])
def test_agent_run_requires_cuda_device(monkeypatch, capsys, command: str) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    assert main([command]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--cuda-device" in captured.err


@pytest.mark.parametrize(
    ("command", "cuda_device"),
    [("atnagent", 0), ("ffnagent", 1)],
)
def test_agent_run_dispatches_selected_role(monkeypatch, command: str, cuda_device: int) -> None:
    calls: list[int] = []

    class FakeAgent:
        def __init__(self, *, cuda_device: int) -> None:
            calls.append(cuda_device)

        def run(self) -> None:
            return None

    if command == "atnagent":
        monkeypatch.setattr(xpool.cli.subcommands.atnagent, "AtnAgent", FakeAgent)
    else:
        monkeypatch.setattr(xpool.cli.subcommands.ffnagent, "FfnAgent", FakeAgent)

    assert main([command, "--config", "configs/xpool.example.toml", "--cuda-device", str(cuda_device)]) == 0

    assert calls == [cuda_device]
