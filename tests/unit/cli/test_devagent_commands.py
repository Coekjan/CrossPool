from __future__ import annotations

from xpool.cli import main
from xpool.cli.subcommands import devagent as devagent_cli


def test_devagent_run_requires_cuda_device(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    assert main(["devagent"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--cuda-device" in captured.err


def test_devagent_run_dispatches_selected_agent(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    class FakeDevagent:
        def run(self) -> None:
            return None

    def create_fake_devagent(cuda_device: int) -> FakeDevagent:
        calls.append(cuda_device)
        return FakeDevagent()

    monkeypatch.setattr(devagent_cli, "create_devagent", create_fake_devagent)

    assert main(["devagent", "--cuda-device", "0"]) == 0

    assert calls == [0]
