from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.harness.support.config import reset_global_config, write_minimal_config
from xpool.config import ConfigError, XpoolConfig, init_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "daemon.example"])
def test_daemon_host_must_be_loopback(host: str) -> None:
    with pytest.raises(ValidationError, match=r"daemon\.host must be localhost or a loopback IP address"):
        XpoolConfig.from_mapping(
            {
                "daemon": {"host": host, "port": 9810},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


@pytest.mark.parametrize(
    ("observer_name", "env_prefix", "capacity"),
    [
        ("transport_observer", "XPOOL_DEBUG_TRANSPORT_OBSERVER", 64),
        ("fabric_observer", "XPOOL_DEBUG_FABRIC_OBSERVER", 32),
    ],
)
def test_env_source_parses_native_observer_settings(
    tmp_path: Path,
    observer_name: str,
    env_prefix: str,
    capacity: int,
) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    outdir = tmp_path.resolve()

    enabled = XpoolConfig.from_mapping(
        payload,
        env={
            f"{env_prefix}_ENABLE": "1",
            f"{env_prefix}_OUTDIR": str(outdir),
            f"{env_prefix}_RECORD_CAPACITY": str(capacity),
        },
    )

    observer = getattr(enabled.debug, observer_name)
    assert observer.enable is True
    assert observer.outdir == outdir
    assert observer.record_capacity == capacity


@pytest.mark.parametrize(
    ("observer_name", "outdir_env_var"),
    [
        ("transport_observer", "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR"),
        ("fabric_observer", "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR"),
    ],
)
def test_native_observer_requires_enable_and_outdir_together(
    tmp_path: Path,
    observer_name: str,
    outdir_env_var: str,
) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match=observer_name):
        XpoolConfig.from_mapping(payload, env={outdir_env_var: str(tmp_path)})


@pytest.mark.parametrize(
    ("observer", "field", "value", "setting_name"),
    [
        ("transport_observer", "enable", True, "debug_transport_observer_enable"),
        ("transport_observer", "outdir", "/tmp/transport", "debug_transport_observer_outdir"),
        ("transport_observer", "record_capacity", 16, "debug_transport_observer_record_capacity"),
        ("fabric_observer", "enable", True, "debug_fabric_observer_enable"),
        ("fabric_observer", "outdir", "/tmp/fabric", "debug_fabric_observer_outdir"),
        ("fabric_observer", "record_capacity", 16, "debug_fabric_observer_record_capacity"),
    ],
)
def test_native_observer_settings_cannot_be_set_from_toml(
    observer: str,
    field: str,
    value: bool | str | int,
    setting_name: str,
) -> None:
    with pytest.raises(ConfigError, match=setting_name):
        XpoolConfig.from_mapping(
            {
                "debug": {observer: {field: value}},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


@pytest.mark.parametrize("capacity", [0, -1, 2**63])
@pytest.mark.parametrize(
    "env_var",
    [
        "XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY",
        "XPOOL_DEBUG_FABRIC_OBSERVER_RECORD_CAPACITY",
    ],
)
def test_observer_record_capacity_must_fit_native_range(capacity: int, env_var: str) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="record_capacity"):
        XpoolConfig.from_mapping(payload, env={env_var: str(capacity)})


def test_env_source_parses_graph_observer_settings(tmp_path: Path) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    outdir = tmp_path.resolve()

    enabled = XpoolConfig.from_mapping(
        payload,
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(outdir),
        },
    )

    assert enabled.debug.graph_observer.enable is True
    assert enabled.debug.graph_observer.outdir == outdir


def test_env_source_parses_prefill_logit_observer_settings(tmp_path: Path) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    outdir = tmp_path.resolve()

    enabled = XpoolConfig.from_mapping(
        payload,
        env={
            "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_OUTDIR": str(outdir),
        },
    )

    assert enabled.debug.prefill_logit_observer.enable is True
    assert enabled.debug.prefill_logit_observer.outdir == outdir


def test_graph_observer_outdir_accepts_relative_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    config = XpoolConfig.from_mapping(
        payload,
        env={
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": "relative/events",
        },
    )

    assert config.debug.graph_observer.outdir == (tmp_path / "relative/events").resolve()


@pytest.mark.parametrize(
    "env_var",
    [
        "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE",
        "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE",
        "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE",
        "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE",
    ],
)
def test_env_source_rejects_malformed_debug_boolean(env_var: str) -> None:
    with pytest.raises(ConfigError, match="boolean flag"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={env_var: "true"},
        )


@pytest.mark.parametrize(
    ("debug_payload", "setting_name"),
    [
        ({"graph_observer": {"enable": True}}, "debug_graph_observer_enable"),
        ({"graph_observer": {"outdir": "/tmp/xpool-graph-events"}}, "debug_graph_observer_outdir"),
        ({"prefill_logit_observer": {"enable": True}}, "debug_prefill_logit_observer_enable"),
        (
            {"prefill_logit_observer": {"outdir": "/tmp/xpool-prefill-logits"}},
            "debug_prefill_logit_observer_outdir",
        ),
    ],
)
def test_debug_graph_observer_cannot_be_set_from_toml(
    debug_payload: dict[str, object],
    setting_name: str,
) -> None:
    with pytest.raises(ConfigError, match=setting_name):
        XpoolConfig.from_mapping(
            {
                "debug": debug_payload,
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )


@pytest.mark.parametrize(
    "env",
    [
        {"XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1"},
        {"XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": "events"},
    ],
)
def test_graph_observer_requires_enable_and_outdir_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="must be set or unset together"):
        XpoolConfig.from_mapping(payload, env=env)


@pytest.mark.parametrize(
    "env",
    [
        {"XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE": "1"},
        {"XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_OUTDIR": "events"},
    ],
)
def test_prefill_logit_observer_requires_enable_and_outdir_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="must be set or unset together"):
        XpoolConfig.from_mapping(payload, env=env)


def test_unknown_xpool_env_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        config = XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={
                "XPOOL_UNKNOWN_SETTING": "0",
                "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "0",
                "XPOOL_UNKNOWN": "1",
            },
        )

    assert config.debug.graph_observer.enable is False
    assert "XPOOL_UNKNOWN_SETTING" in caplog.text
    assert "XPOOL_UNKNOWN" in caplog.text


def test_init_global_config_warns_once_for_unknown_xpool_env(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_minimal_config(tmp_path)
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    monkeypatch.setenv("XPOOL_UNKNOWN", "1")

    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        init_global_config(config_path=config_path)

    messages = [record.message for record in caplog.records if "XPOOL_UNKNOWN" in record.message]
    assert messages == ["ignoring unknown xpool environment variables: XPOOL_UNKNOWN"]
