from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

import xpool.config as config_module
from xpool.cli import main
from xpool.runtime.mps import MpsPreflight


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    yield
    monkeypatch.setattr(config_module, "_global_config", None)


def test_daemon_check_uses_cli_override(monkeypatch, capsys) -> None:
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_healthy_mps))

    assert main(["daemon", "--config", "configs/xpool.example.toml", "--daemon-host", "cli-host", "--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["daemon"]["host"] == "cli-host"
    assert payload["mps"]["healthy"] is True


def test_daemon_check_returns_unhealthy_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_unhealthy_mps))

    assert main(["daemon", "--config", "configs/xpool.example.toml", "--check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["mps"]["healthy"] is False


def test_device_agent_check_uses_env_config_path(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_healthy_mps))

    assert main(["device-agent", "--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["device_agents"][0]["id"] == "cuda0"
    assert payload["device_agents"][0]["cuda_device"] == 0
    assert payload["device_agents"][0]["nvshmem_rank"] == 0
    assert payload["device_agents"][1]["id"] == "cuda1"
    assert payload["device_agents"][1]["cuda_device"] == 1
    assert payload["device_agents"][1]["nvshmem_rank"] == 1
    assert payload["mps"]["healthy"] is True


def test_device_agent_check_returns_unhealthy_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_unhealthy_mps))

    assert main(["device-agent", "--check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["mps"]["healthy"] is False


def test_missing_config_path_returns_cli_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_healthy_mps))

    assert main(["daemon", "--check"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "XPOOL_CONFIG" in captured.err


def test_invalid_config_returns_cli_error(tmp_path, capsys) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[devices]
attention_cuda_devices = [0]
ffn_cuda_devices = [0]

[[models]]
id = "m"
path = "/models/m"
""".strip(),
        encoding="utf-8",
    )

    assert main(["daemon", "--config", str(config_path), "--check"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "overlapping devices" in captured.err


def test_missing_config_file_returns_cli_error(tmp_path, capsys) -> None:
    missing_path = tmp_path / "missing.toml"

    assert main(["daemon", "--config", str(missing_path), "--check"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such file" in captured.err
    assert "Traceback" not in captured.err


def test_malformed_config_file_returns_cli_error(tmp_path, capsys) -> None:
    config_path = tmp_path / "malformed.toml"
    config_path.write_text("[daemon\n", encoding="utf-8")

    assert main(["daemon", "--config", str(config_path), "--check"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_config_registry_is_not_public_cli(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["config-registry"])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def _healthy_mps() -> MpsPreflight:
    return MpsPreflight(
        required=True,
        control_binary="/usr/bin/nvidia-cuda-mps-control",
        pipe_directory=None,
        control_binary_found=True,
        control_daemon_reachable=True,
        healthy=True,
        checked_at_unix_s=1.0,
        message="MPS control daemon is reachable",
    )


def _unhealthy_mps() -> MpsPreflight:
    return MpsPreflight(
        required=True,
        control_binary="/usr/bin/nvidia-cuda-mps-control",
        pipe_directory=None,
        control_binary_found=True,
        control_daemon_reachable=False,
        healthy=False,
        checked_at_unix_s=1.0,
        message="MPS control daemon is not reachable",
    )
