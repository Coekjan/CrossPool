from __future__ import annotations

from xpool.cli import main
from xpool.cli.subcommands import atnagent as atnagent_cli


def test_atnagent_run_requires_cuda_device(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    assert main(["atnagent"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--cuda-device" in captured.err


def test_atnagent_run_dispatches_selected_agent(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FakeAtnAgent:
        def __init__(self, *, cuda_device: int) -> None:
            calls.append(cuda_device)

        def run(self) -> None:
            return None

    monkeypatch.setattr(atnagent_cli, "AtnAgent", FakeAtnAgent)

    assert main(["atnagent", "--cuda-device", "0"]) == 0

    assert calls == [0]
