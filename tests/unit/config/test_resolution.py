from __future__ import annotations

from copy import deepcopy

import xpool.config as config_module
from tests.harness.config import (
    Path,
    pytest,
    source_record,
    write_minimal_config,
)
from xpool.config import (
    ConfigError,
    ConfigSource,
    MissingRequiredConfig,
    XpoolConfig,
    get_global_config,
    init_global_config,
)


def test_cli_config_default_precedence_without_config_field_env(
    tmp_path: Path,
) -> None:
    config_path = write_minimal_config(
        tmp_path,
        daemon_host="127.0.0.2",
        daemon_port=1000,
        atn_concurrency=1,
        ffn_concurrency=1,
    )

    config = init_global_config(
        config_path=config_path,
        env={"XPOOL_CONFIG": "ignored-when-config-path-is-explicit"},
        cli={
            "daemon_host": "127.0.0.3",
            "scheduler_atn_concurrency": 3,
        },
    )

    assert config.daemon.host == "127.0.0.3"
    assert config.daemon.port == 1000
    assert config.scheduler.atn_concurrency == 3
    assert config.scheduler.ffn_concurrency == 1


def test_defaults_fill_missing_optional_sections() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        cli={},
    )

    assert config.debug.shim_loopback.enable is False
    assert config.debug.transport_loopback.enable is False
    assert config.debug.graph_observer.enable is False
    assert config.debug.graph_observer.outdir is None
    assert config.daemon.host == "127.0.0.1"
    assert config.daemon.port == 9810
    assert config.scheduler.atn_concurrency == 1
    assert config.scheduler.ffn_concurrency == 1
    assert config.vendor.model_base_uri is None


def test_config_resolution_does_not_mutate_caller_mapping() -> None:
    payload: dict[str, object] = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "vendor": {"model_base_uri": "/models"},
        "models": [{"id": "m"}],
    }
    original = deepcopy(payload)

    config = XpoolConfig.from_mapping(payload, cli={"daemon_host": "127.0.0.3"})

    assert config.daemon.host == "127.0.0.3"
    assert config.model_path_of("m") == Path("/models/m")
    assert payload == original


def test_config_path_precedence_uses_cli_before_env(
    tmp_path: Path,
) -> None:
    env_config = write_minimal_config(tmp_path / "env", daemon_host="127.0.0.4")
    cli_config = write_minimal_config(tmp_path / "cli", daemon_host="127.0.0.5")

    config = init_global_config(config_path=cli_config, env={"XPOOL_CONFIG": str(env_config)})

    assert config.daemon.host == "127.0.0.5"


def test_missing_config_path_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="XPOOL_CONFIG"):
        init_global_config(env={})


def test_global_config_access_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "global_config", None)
    with pytest.raises(MissingRequiredConfig, match="global config"):
        get_global_config()

    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    assert init_global_config(config=config) is config
    assert get_global_config() is config


def test_init_global_config_is_idempotent_for_equal_effective_config() -> None:
    payload = {
        "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    first = XpoolConfig.from_mapping(payload)
    equivalent = XpoolConfig.from_mapping(payload)

    assert init_global_config(config=first) is first
    assert init_global_config(config=equivalent) is first
    assert get_global_config() is first


def test_init_global_config_rejects_different_effective_config() -> None:
    first = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    different = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    init_global_config(config=first)
    with pytest.raises(ConfigError, match="already initialized with different values"):
        init_global_config(config=different)


def test_init_global_config_rejects_mixed_injection_inputs() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        init_global_config(config=config, env={})


def test_init_global_config_tracks_effective_sources() -> None:
    config = init_global_config(
        config_path="configs/xpool.example.toml",
        cli={"daemon_host": "127.0.0.6"},
        env={"XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1"},
    )
    report = config.sources

    assert config.daemon.host == "127.0.0.6"
    assert config.debug.shim_loopback.enable is True
    assert "sources" not in config.model_dump(mode="json")
    assert source_record(report, "daemon.host") == {
        "name": "daemon.host",
        "source": ConfigSource.CLI,
        "value": "127.0.0.6",
    }
    assert source_record(report, "debug.shim_loopback.enable") == {
        "name": "debug.shim_loopback.enable",
        "source": ConfigSource.ENV,
        "value": True,
    }
    assert source_record(report, "daemon.port")["source"] == ConfigSource.CONFIG
    assert source_record(report, "devices.atn_cuda_devices")["source"] == ConfigSource.CONFIG
    assert source_record(report, "models[0].id")["source"] == ConfigSource.CONFIG
    assert source_record(report, "models[0].path") == {
        "name": "models[0].path",
        "source": ConfigSource.UNSET,
        "value": None,
    }
    assert not any(record["name"] in {"config_path", "devices", "models"} for record in report)


def test_config_sources_format_multiple_model_indices() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [
                {"id": "model-zero", "path": "/custom/model-zero"},
                {"id": "org/model-one"},
            ],
        }
    )
    report = config.sources

    assert source_record(report, "models[0].id")["source"] == ConfigSource.CONFIG
    assert source_record(report, "models[0].path")["source"] == ConfigSource.CONFIG
    assert source_record(report, "models[1].id")["source"] == ConfigSource.CONFIG
    assert source_record(report, "models[1].path") == {
        "name": "models[1].path",
        "source": ConfigSource.UNSET,
        "value": None,
    }


def test_config_rejects_non_list_models_for_registered_wildcard_settings() -> None:
    with pytest.raises(ConfigError, match="expected list config value"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": {"id": "m", "path": "/models/m"},
            }
        )


def test_init_global_config_uses_env_config_path_without_exposing_bootstrap_source() -> None:
    config = init_global_config(env={"XPOOL_CONFIG": "configs/xpool.example.toml"})

    assert config.daemon.host == "127.0.0.1"
    assert not any(record["name"] == "config_path" for record in config.sources)


def test_int_source_rejects_invalid_integer() -> None:
    with pytest.raises(ConfigError, match="expected integer config value for daemon_port"):
        XpoolConfig.from_mapping(
            {
                "daemon": {"port": "not-an-int"},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )
