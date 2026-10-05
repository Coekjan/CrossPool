from __future__ import annotations

import json

import pytest

from xpool.cli import main
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def config_record(records: list[dict[str, object]], name: str) -> dict[str, object]:
    for record in records:
        if record["name"] == name:
            return record
    raise AssertionError(f"missing config record for {name}")


def json_config_records(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    assert captured.err == ""
    records = json.loads(captured.out)
    assert isinstance(records, list)
    return [record for record in records if isinstance(record, dict)]


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
                "--ffn-placement-parallelism",
                "7",
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
    assert config_record(records, "debug.transport_observer.record_capacity")["source"] == "default"
    assert config_record(records, "atn.devices")["source"] == "config"
    assert config_record(records, "ffn.placement.parallelism") == {
        "name": "ffn.placement.parallelism",
        "source": "cli",
        "value": 7,
    }
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
    monkeypatch.setenv("XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY", "64")

    assert main(["config", "dump"]) == 0

    records = json_config_records(capsys)
    assert config_record(records, "debug.transport_observer.record_capacity") == {
        "name": "debug.transport_observer.record_capacity",
        "source": "env",
        "value": 64,
    }


def test_missing_config_path_returns_cli_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    assert main(["config", "dump"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "XPOOL_CONFIG" in captured.err


@pytest.mark.parametrize(
    ("failure", "expected_message"),
    [
        ("missing", "No such file"),
        ("malformed", "line 1"),
        ("schema", "immediately follow the attention device block"),
    ],
)
def test_config_dump_reports_configuration_errors_without_traceback(
    tmp_path,
    capsys,
    failure: str,
    expected_message: str,
) -> None:
    config_path = tmp_path / f"{failure}.toml"
    if failure == "malformed":
        config_path.write_text("[daemon\n", encoding="utf-8")
    elif failure == "schema":
        config_path.write_text(
            f"""
[scheduler]
slo = {{ ttft_ms = 1000, tbt_ms = 50 }}

[atn]
devices = [0]

[ffn]
devices = [0]

[[models]]
id = "{TEST_MODEL_ID}"
path = "/models/m"
""".strip(),
            encoding="utf-8",
        )

    assert main(["config", "dump", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected_message in captured.err
    assert "Traceback" not in captured.err
