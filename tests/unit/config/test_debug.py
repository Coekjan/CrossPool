from __future__ import annotations

import logging

from pydantic import ValidationError

from tests.harness.config import (
    Path,
    pytest,
    write_minimal_config,
)
from xpool.config import ConfigError, XpoolConfig, init_global_config


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


def test_env_source_parses_debug_loopback_flag() -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    enabled = XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1"})
    disabled = XpoolConfig.from_mapping(payload, env={})
    explicitly_disabled = XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "0"})

    assert enabled.debug.shim_loopback.enable is True
    assert disabled.debug.shim_loopback.enable is False
    assert explicitly_disabled.debug.shim_loopback.enable is False
    assert disabled.debug.transport_loopback.enable is False
    assert disabled.debug.graph_observer.enable is False
    assert disabled.debug.transport_observer.enable is False


def test_env_source_parses_transport_observer_settings(tmp_path: Path) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    outdir = tmp_path.resolve()

    enabled = XpoolConfig.from_mapping(
        payload,
        env={
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(outdir),
        },
    )

    assert enabled.debug.transport_observer.enable is True
    assert enabled.debug.transport_observer.outdir == outdir


def test_transport_observer_requires_enable_and_outdir_together(tmp_path: Path) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="transport_observer"):
        XpoolConfig.from_mapping(
            payload,
            env={"XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(tmp_path)},
        )


def test_env_source_parses_transport_loopback_flag() -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    enabled = XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_TRANSPORT_LOOPBACK_ENABLE": "1"})
    disabled = XpoolConfig.from_mapping(payload, env={})

    assert enabled.debug.transport_loopback.enable is True
    assert disabled.debug.transport_loopback.enable is False


def test_debug_loopback_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={
                "XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1",
                "XPOOL_DEBUG_TRANSPORT_LOOPBACK_ENABLE": "1",
            },
        )


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


def test_env_source_rejects_invalid_debug_loopback_flag() -> None:
    with pytest.raises(ConfigError, match="boolean flag"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "true"},
        )


def test_env_source_rejects_invalid_transport_loopback_flag() -> None:
    with pytest.raises(ConfigError, match="boolean flag"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_TRANSPORT_LOOPBACK_ENABLE": "true"},
        )


def test_env_source_rejects_invalid_graph_observer_flag() -> None:
    with pytest.raises(ConfigError, match="boolean flag"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "true"},
        )


def test_debug_loopback_cannot_be_set_from_toml() -> None:
    with pytest.raises(ConfigError, match="debug_shim_loopback_enable"):
        XpoolConfig.from_mapping(
            {
                "debug": {"shim_loopback": {"enable": True}},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )


def test_debug_transport_loopback_cannot_be_set_from_toml() -> None:
    with pytest.raises(ConfigError, match="debug_transport_loopback_enable"):
        XpoolConfig.from_mapping(
            {
                "debug": {"transport_loopback": {"enable": True}},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )


def test_debug_graph_observer_cannot_be_set_from_toml() -> None:
    for debug_payload, setting_name in (
        ({"graph_observer": {"enable": True}}, "debug_graph_observer_enable"),
        ({"graph_observer": {"outdir": "/tmp/xpool-graph-events"}}, "debug_graph_observer_outdir"),
    ):
        with pytest.raises(ConfigError, match=setting_name):
            XpoolConfig.from_mapping(
                {
                    "debug": debug_payload,
                    "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                    "models": [{"id": "m", "path": "/models/m"}],
                },
            )


def test_graph_observer_requires_outdir_when_enabled() -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="must be set or unset together"):
        XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1"})


def test_graph_observer_rejects_outdir_when_disabled(tmp_path: Path) -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }

    with pytest.raises(ValidationError, match="must be set or unset together"):
        XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(tmp_path)})


def test_unknown_xpool_env_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        config = XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={
                "XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "0",
                "XPOOL_DEBUG_TRANSPORT_LOOPBACK_ENABLE": "0",
                "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "0",
                "XPOOL_UNKNOWN": "1",
            },
        )

    assert config.debug.shim_loopback.enable is False
    assert config.debug.transport_loopback.enable is False
    assert config.debug.graph_observer.enable is False
    assert "XPOOL_UNKNOWN" in caplog.text


def test_init_global_config_warns_once_for_unknown_xpool_env(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = write_minimal_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        init_global_config(
            config_path=config_path,
            env={
                "XPOOL_CONFIG": str(config_path),
                "XPOOL_UNKNOWN": "1",
            },
        )

    messages = [record.message for record in caplog.records if "XPOOL_UNKNOWN" in record.message]
    assert messages == ["Ignoring unknown xpool environment variables: XPOOL_UNKNOWN"]
