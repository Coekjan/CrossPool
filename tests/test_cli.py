from __future__ import annotations

import json

import pytest

from xpool.cli import main
from xpool.runtime.mps import MpsPreflight


def test_daemon_check_uses_cli_override(monkeypatch, capsys) -> None:
    monkeypatch.setattr(MpsPreflight, "detect", staticmethod(_healthy_mps))

    assert main(["daemon", "--config", "configs/xpool.example.toml", "--daemon-host", "cli-host", "--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["daemon"]["host"] == "cli-host"
    assert payload["mps"]["healthy"] is True


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
