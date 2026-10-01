from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

import xpool.config
from xpool.config import (
    ConfigError,
    ConfigSource,
    MissingRequiredConfig,
    XpoolConfig,
    get_global_config,
    init_global_config,
)
from xtest.harness.support.config import (
    TEST_MODEL_ID,
    minimal_config,
    reset_global_config,
    write_minimal_config,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_cli_config_default_precedence_without_config_field_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_minimal_config(
        tmp_path,
        daemon_host="127.0.0.2",
        daemon_port=1000,
        atn_concurrency=1,
        ffn_concurrency=1,
    )

    monkeypatch.setenv("XPOOL_CONFIG", "ignored-when-config-path-is-explicit")
    config = init_global_config(
        config_path=config_path,
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
    config = minimal_config()

    assert config.debug.graph_observer.enable is False
    assert config.debug.graph_observer.outdir is None
    assert config.debug.prefill_logit_observer.enable is False
    assert config.debug.prefill_logit_observer.outdir is None
    assert config.daemon.host == "127.0.0.1"
    assert config.daemon.port == 9810
    assert config.scheduler.atn_concurrency == 1
    assert config.scheduler.ffn_concurrency == 1
    assert config.ffn.loader.parallelism == 4
    assert config.logging.level == "info"
    assert config.logging.color is True
    assert config.ffn.device_memory_calibration is None
    assert config.vendor.model_base_uri is None
    sources = {record["name"]: record["source"] for record in config.sources}
    assert sources["daemon.host"] is ConfigSource.DEFAULT
    assert sources["scheduler.ffn_policy"] is ConfigSource.DEFAULT
    assert sources["vendor.model_base_uri"] is ConfigSource.UNSET
    assert sources["models[0].path"] is ConfigSource.CONFIG


@pytest.mark.parametrize("model_slo", [None, (700, 25)])
def test_fixture_toml_preserves_quoted_paths_and_optional_model_slo(
    tmp_path: Path, model_slo: tuple[int, int] | None
) -> None:
    model_path = tmp_path / 'quoted"model\\weights'
    path = write_minimal_config(tmp_path, model_path=model_path, model_slo=model_slo)
    config = XpoolConfig.from_file(path)

    assert config.model_path_of(TEST_MODEL_ID) == model_path
    if model_slo is None:
        assert config.models[0].slo is None
    else:
        slo = config.models[0].slo
        assert slo is not None
        assert (slo.ttft_ms, slo.tbt_ms) == model_slo


def test_ffn_loader_parallelism_uses_cli_env_config_default_precedence() -> None:
    payload = {
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1], "loader": {"parallelism": 2}},
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }

    assert XpoolConfig.from_mapping(payload).ffn.loader.parallelism == 2
    assert XpoolConfig.from_mapping(payload, env={"XPOOL_FFN_LOADER_PARALLELISM": "3"}).ffn.loader.parallelism == 3
    assert (
        XpoolConfig.from_mapping(
            payload,
            cli={"ffn_loader_parallelism": 5},
            env={"XPOOL_FFN_LOADER_PARALLELISM": "3"},
        ).ffn.loader.parallelism
        == 5
    )


def test_ffn_placement_parallelism_uses_cli_env_config_default_precedence() -> None:
    payload = {
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1], "placement": {"parallelism": 2}},
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }

    assert XpoolConfig.from_mapping(payload).ffn.placement.parallelism == 2
    assert (
        XpoolConfig.from_mapping(payload, env={"XPOOL_FFN_PLACEMENT_PARALLELISM": "3"}).ffn.placement.parallelism == 3
    )
    assert (
        XpoolConfig.from_mapping(
            payload,
            cli={"ffn_placement_parallelism": 5},
            env={"XPOOL_FFN_PLACEMENT_PARALLELISM": "3"},
        ).ffn.placement.parallelism
        == 5
    )


def test_logging_level_uses_cli_env_config_default_precedence() -> None:
    payload = {
        "logging": {"level": "warning"},
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1]},
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }

    assert XpoolConfig.from_mapping(payload).logging.level == "warning"
    assert XpoolConfig.from_mapping(payload, env={"XPOOL_LOG_LEVEL": "debug"}).logging.level == "debug"
    assert (
        XpoolConfig.from_mapping(
            payload,
            cli={"logging_level": "error"},
            env={"XPOOL_LOG_LEVEL": "debug"},
        ).logging.level
        == "error"
    )


def test_logging_color_uses_toml_and_ignores_environment_override(caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "logging": {"color": False},
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1]},
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }

    with caplog.at_level("WARNING", logger="xpool.config"):
        config = XpoolConfig.from_mapping(payload, env={"XPOOL_LOGGING_COLOR": "1"})

    assert config.logging.color is False
    assert "ignoring unknown xpool environment variables" in caplog.text


def test_ffn_device_memory_calibration_uses_env_before_config() -> None:
    payload = {
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1], "device_memory_calibration": "/config/memory.json"},
        "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
    }

    assert XpoolConfig.from_mapping(payload).ffn.device_memory_calibration == Path("/config/memory.json")
    assert XpoolConfig.from_mapping(
        payload,
        env={"XPOOL_FFN_DEVICE_MEMORY_CALIBRATION": "/env/memory.json"},
    ).ffn.device_memory_calibration == Path("/env/memory.json")


def test_ffn_device_memory_calibration_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match=r"ffn\.device_memory_calibration must be absolute"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1], "device_memory_calibration": "relative.json"},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )


def test_ffn_device_memory_calibration_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1], "device_memory_calibration": "~/memory.json"},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )

    assert config.ffn.device_memory_calibration == tmp_path / "memory.json"


def test_config_resolution_does_not_mutate_caller_mapping() -> None:
    payload: dict[str, object] = {
        "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
        "atn": {"devices": [0]},
        "ffn": {"devices": [1]},
        "vendor": {"model_base_uri": "/models"},
        "models": [{"id": str(TEST_MODEL_ID)}],
    }
    original = deepcopy(payload)

    config = XpoolConfig.from_mapping(payload, cli={"daemon_host": "127.0.0.3"})

    assert config.daemon.host == "127.0.0.3"
    assert config.model_path_of(TEST_MODEL_ID) == Path("/models") / TEST_MODEL_ID.relative_path
    assert payload == original


def test_config_path_precedence_uses_cli_before_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_config = write_minimal_config(tmp_path / "env", daemon_host="127.0.0.4")
    cli_config = write_minimal_config(tmp_path / "cli", daemon_host="127.0.0.5")

    monkeypatch.setenv("XPOOL_CONFIG", str(env_config))
    config = init_global_config(config_path=cli_config)

    assert config.daemon.host == "127.0.0.5"


def test_missing_config_path_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    with pytest.raises(MissingRequiredConfig, match="XPOOL_CONFIG"):
        init_global_config()


def test_global_config_access_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.config, "global_config", None)
    with pytest.raises(MissingRequiredConfig, match="global config"):
        get_global_config()

    config_path = write_minimal_config(tmp_path)

    config = init_global_config(config_path=config_path)
    assert get_global_config() is config


def test_init_global_config_is_idempotent_for_equal_effective_config(tmp_path: Path) -> None:
    config_path = write_minimal_config(tmp_path)
    first = init_global_config(config_path=config_path)

    assert init_global_config(config_path=config_path) is first
    assert get_global_config() is first


def test_init_global_config_rejects_different_effective_config(tmp_path: Path) -> None:
    first_path = write_minimal_config(tmp_path / "first", ffn_cuda_devices=(2,))
    different_path = write_minimal_config(tmp_path / "different", ffn_cuda_devices=(3,))

    init_global_config(config_path=first_path)
    with pytest.raises(ConfigError, match="already initialized with different values"):
        init_global_config(config_path=different_path)


def test_init_global_config_tracks_effective_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY", "64")
    config = init_global_config(
        config_path="configs/xpool.example.toml",
        cli={"daemon_host": "127.0.0.6"},
    )
    sources = {record["name"]: record for record in config.sources}

    assert config.daemon.host == "127.0.0.6"
    assert "sources" not in config.model_dump(mode="json")
    assert sources["daemon.host"] == {
        "name": "daemon.host",
        "source": ConfigSource.CLI,
        "value": "127.0.0.6",
    }
    assert sources["debug.transport_observer.record_capacity"] == {
        "name": "debug.transport_observer.record_capacity",
        "source": ConfigSource.ENV,
        "value": 64,
    }
    assert sources["debug.fabric_observer.record_capacity"] == {
        "name": "debug.fabric_observer.record_capacity",
        "source": ConfigSource.DEFAULT,
        "value": 8192,
    }
    assert sources["daemon.port"]["source"] == ConfigSource.CONFIG
    assert sources["atn.devices"]["source"] == ConfigSource.CONFIG
    assert sources["models[0].id"]["source"] == ConfigSource.CONFIG
    assert sources["models[0].path"] == {
        "name": "models[0].path",
        "source": ConfigSource.UNSET,
        "value": None,
    }
    assert not {"config_path", "models"} & sources.keys()


def test_config_sources_format_multiple_model_indices() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": "test/model-zero", "path": "/custom/model-zero"},
                {"id": "org/model-one"},
            ],
        }
    )
    sources = {record["name"]: record for record in config.sources}

    assert sources["models[0].id"]["source"] == ConfigSource.CONFIG
    assert sources["models[0].path"]["source"] == ConfigSource.CONFIG
    assert sources["models[1].id"]["source"] == ConfigSource.CONFIG
    assert sources["models[1].path"] == {
        "name": "models[1].path",
        "source": ConfigSource.UNSET,
        "value": None,
    }


def test_config_rejects_non_list_models_for_registered_wildcard_settings() -> None:
    with pytest.raises(ConfigError, match="expected list config value"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": {"id": str(TEST_MODEL_ID), "path": "/models/m"},
            }
        )


def test_init_global_config_uses_env_config_path_without_exposing_bootstrap_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XPOOL_CONFIG", "configs/xpool.example.toml")
    config = init_global_config()

    assert config.daemon.host == "127.0.0.1"
    assert not any(record["name"] == "config_path" for record in config.sources)


def test_int_source_rejects_invalid_integer() -> None:
    with pytest.raises(ConfigError, match="expected integer config value for daemon_port"):
        XpoolConfig.from_mapping(
            {
                "daemon": {"port": "not-an-int"},
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            },
        )
