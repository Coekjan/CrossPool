from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import xpool.config as config_module
from xpool.config import (
    CONFIG_REGISTRY,
    ConfigError,
    ConfigSource,
    DeviceRole,
    MissingRequiredConfig,
    XpoolConfig,
    get_global_config,
    init_global_config,
)


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config_module, "_global_config", None)
    yield
    monkeypatch.setattr(config_module, "_global_config", None)


def test_example_config_loads_schema_only() -> None:
    config = XpoolConfig.from_file(Path("configs/xpool.example.toml"))

    assert config.debug.enable_shim_loopback is False
    assert config.scheduler.attention_concurrency == 1
    assert config.scheduler.transport_concurrency == 1
    assert config.vendor.model_base_uri == Path("/absolute/path/to/models")
    assert config.devices.attention_cuda_devices == [0]
    assert config.devices.ffn_cuda_devices == [1]
    assert [(agent.id, agent.cuda_device, agent.nvshmem_rank, agent.role) for agent in config.device_agents] == [
        ("cuda0", 0, 0, DeviceRole.ATTENTION),
        ("cuda1", 1, 1, DeviceRole.FFN),
    ]
    assert config.models[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    with pytest.raises(AttributeError):
        _ = config.models[0].path
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/absolute/path/to/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    assert config.serving_instances[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert config.serving_instances[0].ffn_agent_ids == ["cuda1"]


def test_example_config_covers_required_schema_paths() -> None:
    with Path("configs/xpool.example.toml").open("rb") as config_file:
        payload = tomllib.load(config_file)

    missing = [
        setting.name
        for setting in CONFIG_REGISTRY
        if setting.required
        and ConfigSource.CONFIG in setting.allowed_sources
        and setting.path is not None
        and not _has_required_path(payload, setting.path)
    ]

    assert missing == []


def test_old_explicit_topology_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "device_agents": [{"id": "gpu0", "cuda_device": 0, "nvshmem_rank": 0, "roles": ["attention"]}],
                "models": [{"id": "m", "path": "/models/m"}],
                "sglang_instances": [{"id": "m", "model": "m"}],
            },
            cli_overrides={},
        )


def test_model_metadata_fields_are_rejected_from_toml() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m", "family": "dense", "hidden_size": 1}],
            },
            cli_overrides={},
        )


def test_duplicate_and_overlapping_devices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="devices.attention_cuda_devices must be unique"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0, 0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli_overrides={},
        )

    with pytest.raises(ValidationError, match="overlapping devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli_overrides={},
        )


def test_duplicate_model_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="model paths must be unique"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [
                    {"id": "m0", "path": "/models/m"},
                    {"id": "m1", "path": "/models/m"},
                ],
            },
            cli_overrides={},
        )


def test_cli_config_default_precedence_without_config_field_env(
    tmp_path: Path,
) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        daemon_host="from-config",
        daemon_port=1000,
        attention_concurrency=1,
        transport_concurrency=1,
    )

    config = init_global_config(
        config_path=config_path,
        env={"XPOOL_CONFIG": "ignored-when-config-path-is-explicit"},
        cli_overrides={
            "daemon_host": "from-cli",
            "scheduler_attention_concurrency": 3,
        },
    )

    assert config.daemon.host == "from-cli"
    assert config.daemon.port == 1000
    assert config.scheduler.attention_concurrency == 3
    assert config.scheduler.transport_concurrency == 1


def test_defaults_fill_missing_optional_sections() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        cli_overrides={},
    )

    assert config.debug.enable_shim_loopback is False
    assert config.daemon.host == "127.0.0.1"
    assert config.daemon.port == 9810
    assert config.scheduler.attention_concurrency == 1
    assert config.scheduler.transport_concurrency == 1
    assert config.vendor.model_base_uri is None


def test_config_resolution_does_not_mutate_caller_mapping() -> None:
    payload: dict[str, object] = {
        "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "vendor": {"model_base_uri": "/models"},
        "models": [{"id": "m"}],
    }
    original = deepcopy(payload)

    config = XpoolConfig.from_mapping(payload, cli_overrides={"daemon_host": "from-cli"})

    assert config.daemon.host == "from-cli"
    assert config.model_path_of("m") == Path("/models/m")
    assert payload == original


def test_missing_required_top_level_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="devices"):
        XpoolConfig.from_mapping({"models": []}, cli_overrides={})


def test_missing_required_nested_field_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="ffn_cuda_devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli_overrides={},
        )


def test_config_path_precedence_uses_cli_before_env(
    tmp_path: Path,
) -> None:
    env_config = _write_minimal_config(tmp_path / "env", daemon_host="from-env-config")
    cli_config = _write_minimal_config(tmp_path / "cli", daemon_host="from-cli-config")

    config = init_global_config(config_path=cli_config, env={"XPOOL_CONFIG": str(env_config)})

    assert config.daemon.host == "from-cli-config"


def test_missing_config_path_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="XPOOL_CONFIG"):
        init_global_config(env={})


def test_global_config_access_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_global_config", None)
    with pytest.raises(MissingRequiredConfig, match="global config"):
        get_global_config()

    config = XpoolConfig.from_mapping(
        {
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    assert init_global_config(config=config) is config
    assert get_global_config() is config


def test_init_global_config_rejects_mixed_injection_inputs() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        init_global_config(config=config, env={})


def test_registry_declares_process_env_settings() -> None:
    registry = {setting.name: setting for setting in CONFIG_REGISTRY}

    assert "sglang_plugins" not in registry
    assert "preflight_require_mps" not in registry
    assert "capture_decode_buckets" not in registry
    assert registry["config_path"].env_var == "XPOOL_CONFIG"
    assert registry["config_path"].allowed_sources == (ConfigSource.CLI, ConfigSource.ENV)
    assert registry["config_path"].path is None
    assert registry["debug_enable_shim_loopback"].env_var == "XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK"
    assert registry["debug_enable_shim_loopback"].allowed_sources == (ConfigSource.ENV, ConfigSource.DEFAULT)
    assert registry["debug_enable_shim_loopback"].path == ("debug", "enable_shim_loopback")
    assert registry["vendor_model_base_uri"].env_var is None
    assert registry["vendor_model_base_uri"].allowed_sources == (ConfigSource.CONFIG,)
    assert registry["vendor_model_base_uri"].path == ("vendor", "model_base_uri")


def test_env_source_parses_debug_loopback_flag() -> None:
    payload = {
        "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m"}],
    }
    enabled = XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "1"})
    disabled = XpoolConfig.from_mapping(payload, env={})
    explicitly_disabled = XpoolConfig.from_mapping(payload, env={"XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "0"})

    assert enabled.debug.enable_shim_loopback is True
    assert disabled.debug.enable_shim_loopback is False
    assert explicitly_disabled.debug.enable_shim_loopback is False


def test_env_source_rejects_invalid_debug_loopback_flag() -> None:
    with pytest.raises(ConfigError, match="boolean flag"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={"XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "true"},
        )


def test_int_source_rejects_invalid_integer() -> None:
    with pytest.raises(ConfigError, match="expected integer config value for daemon_port"):
        XpoolConfig.from_mapping(
            {
                "daemon": {"port": "not-an-int"},
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )


def test_debug_loopback_cannot_be_set_from_toml() -> None:
    with pytest.raises(ConfigError, match="debug_enable_shim_loopback"):
        XpoolConfig.from_mapping(
            {
                "debug": {"enable_shim_loopback": True},
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
        )


def test_unknown_xpool_env_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        config = XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            env={
                "XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK": "0",
                "XPOOL_UNKNOWN": "1",
            },
        )

    assert config.debug.enable_shim_loopback is False
    assert "XPOOL_UNKNOWN" in caplog.text


def test_vendor_model_base_uri_from_config_derives_model_path() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat"}],
        }
    )

    assert config.vendor.model_base_uri == Path("/models")
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )


def test_vendor_model_base_uri_is_config_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        config = XpoolConfig.from_mapping(
            {
                "vendor": {"model_base_uri": "/models-from-config"},
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat"}],
            },
            env={"XPOOL_VENDOR_MODEL_BASE_URI": "/models-from-env"},
        )

    assert "XPOOL_VENDOR_MODEL_BASE_URI" in caplog.text
    assert config.vendor.model_base_uri == Path("/models-from-config")
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/models-from-config/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )


def test_explicit_model_path_overrides_vendor_model_base_uri() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "path": "/custom/deepseek"}],
        }
    )

    with pytest.raises(AttributeError):
        _ = config.models[0].path
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path("/custom/deepseek")


def test_model_path_lookup_rejects_unknown_model_id() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat"}],
        }
    )

    with pytest.raises(MissingRequiredConfig, match="unknown configured model id"):
        config.model_path_of("deepseek-ai/Unknown")


def test_model_path_or_vendor_model_base_uri_is_required() -> None:
    with pytest.raises(ValidationError, match="vendor.model_base_uri"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat"}],
            }
        )


def test_vendor_model_base_uri_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="vendor.model_base_uri must be absolute"):
        XpoolConfig.from_mapping(
            {
                "vendor": {"model_base_uri": "relative/models"},
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "deepseek-ai/DeepSeek-V2-Lite-Chat"}],
            }
        )


def test_init_global_config_warns_once_for_unknown_xpool_env(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = _write_minimal_config(tmp_path)

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


def test_model_path_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="path must be absolute"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "relative/model"}],
            },
            cli_overrides={},
        )


def test_model_tp_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tp"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1, 2]},
                "models": [{"id": "m", "path": "/models/m", "tp": 4}],
            },
            cli_overrides={},
        )


def _write_minimal_config(
    path: Path,
    *,
    daemon_host: str = "127.0.0.1",
    daemon_port: int = 9810,
    attention_concurrency: int = 1,
    transport_concurrency: int = 1,
    model_path: Path | None = None,
    attention_cuda_devices: tuple[int, ...] = (0, 1),
    ffn_cuda_devices: tuple[int, ...] = (2, 3, 4),
) -> Path:
    if path.suffix != ".toml":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "xpool.toml"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = model_path or Path("/models/deepseek-ai/DeepSeek-V2-Lite-Chat")
    attention_devices = ", ".join(str(device) for device in attention_cuda_devices)
    ffn_devices = ", ".join(str(device) for device in ffn_cuda_devices)
    path.write_text(
        f"""
[daemon]
host = "{daemon_host}"
port = {daemon_port}

[scheduler]
attention_concurrency = {attention_concurrency}
transport_concurrency = {transport_concurrency}

[devices]
attention_cuda_devices = [{attention_devices}]
ffn_cuda_devices = [{ffn_devices}]

[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    return path


def _has_required_path(payload: Mapping[str, object], path: tuple[str, ...]) -> bool:
    if "*" not in path:
        found, value = _get_path(payload, path)
        return found and not _is_empty(value)

    wildcard_index = path.index("*")
    collection_path = path[:wildcard_index]
    item_path = path[wildcard_index + 1 :]
    found, collection = _get_path(payload, collection_path)
    if not found or not isinstance(collection, list) or not collection:
        return False
    for item in collection:
        if not isinstance(item, Mapping):
            return False
        found, value = _get_path(cast(Mapping[str, object], item), item_path)
        if not found or _is_empty(value):
            return False
    return True


def _get_path(payload: Mapping[str, object], path: tuple[str, ...]) -> tuple[bool, object]:
    cursor: object = payload
    for key in path:
        if not isinstance(cursor, Mapping):
            return False, None
        mapping = cast(Mapping[str, object], cursor)
        if key not in mapping:
            return False, None
        cursor = mapping[key]
    return True, cursor


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []
