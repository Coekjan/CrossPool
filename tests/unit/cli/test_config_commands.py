from __future__ import annotations

from tests.harness.cli.common import (
    config_record,
    json_config_records,
)
from xpool.cli import main


def test_config_dump_uses_cli_override(capsys) -> None:
    assert main(["config", "dump", "--config", "configs/xpool.example.toml", "--daemon-host", "127.0.0.6"]) == 0

    records = json_config_records(capsys)
    assert config_record(records, "daemon.host") == {
        "name": "daemon.host",
        "source": "cli",
        "value": "127.0.0.6",
    }
    assert config_record(records, "daemon.port")["source"] == "config"
    assert config_record(records, "models[0].id")["source"] == "config"
    assert not any(record["name"] in {"config_path", "devices", "models"} for record in records)


def test_config_dump_reports_config_and_sources(capsys) -> None:
    assert (
        main(
            [
                "config",
                "dump",
                "--config",
                "configs/xpool.example.toml",
                "--daemon-host",
                "127.0.0.6",
            ]
        )
        == 0
    )

    records = json_config_records(capsys)
    assert config_record(records, "daemon.host") == {
        "name": "daemon.host",
        "source": "cli",
        "value": "127.0.0.6",
    }
    assert config_record(records, "debug.shim_loopback.enable")["source"] == "default"
    assert config_record(records, "models[0].path")["source"] == "unset"
    assert all(set(record) == {"name", "value", "source"} for record in records)


def test_config_dump_uses_env_config_path(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")

    assert main(["config", "dump"]) == 0

    records = json_config_records(capsys)
    assert config_record(records, "daemon.host")["source"] == "config"
    assert not any(record["name"] == "config_path" for record in records)


def test_config_dump_reports_env_source(monkeypatch, capsys) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    monkeypatch.setenv("XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE", "1")

    assert main(["config", "dump"]) == 0

    records = json_config_records(capsys)
    assert config_record(records, "debug.shim_loopback.enable") == {
        "name": "debug.shim_loopback.enable",
        "source": "env",
        "value": True,
    }


def test_missing_config_path_returns_cli_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    assert main(["config", "dump"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "XPOOL_CONFIG" in captured.err


def test_invalid_config_returns_cli_error(tmp_path, capsys) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[devices]
atn_cuda_devices = [0]
ffn_cuda_devices = [0]

[[models]]
id = "m"
path = "/models/m"
""".strip(),
        encoding="utf-8",
    )

    assert main(["config", "dump", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "overlapping devices" in captured.err


def test_missing_config_file_returns_cli_error(tmp_path, capsys) -> None:
    missing_path = tmp_path / "missing.toml"

    assert main(["config", "dump", "--config", str(missing_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such file" in captured.err
    assert "Traceback" not in captured.err


def test_malformed_config_file_returns_cli_error(tmp_path, capsys) -> None:
    config_path = tmp_path / "malformed.toml"
    config_path.write_text("[daemon\n", encoding="utf-8")

    assert main(["config", "dump", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
