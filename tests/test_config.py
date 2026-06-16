from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import xpool.config as config_module
from xpool.config import (
    CONFIG_REGISTRY,
    AttentionKind,
    ConfigError,
    ConfigSource,
    DeviceRole,
    MissingRequiredConfig,
    SglangModelMetadata,
    TopologyError,
    XpoolConfig,
    config_registry_as_dict,
    load_config,
)


def test_example_config_loads_schema_only() -> None:
    config = XpoolConfig.from_file(Path("configs/xpool.example.toml"))

    assert config.scheduler.attention_concurrency == 1
    assert config.scheduler.transport_concurrency == 1
    assert config.devices.attention_cuda_devices == [0]
    assert config.devices.ffn_cuda_devices == [1]
    assert [(agent.id, agent.cuda_device, agent.nvshmem_rank, agent.role) for agent in config.device_agents] == [
        ("cuda0", 0, 0, DeviceRole.ATTENTION),
        ("cuda1", 1, 1, DeviceRole.FFN),
    ]
    assert config.models[0].id == "deepseek-v2-lite-chat"
    assert config.models[0].path.is_absolute()
    assert config.sglang_instances[0].id == "deepseek-v2-lite-chat"
    assert config.sglang_instances[0].ffn_agent_ids == ["cuda1"]


def test_example_config_covers_required_schema_paths() -> None:
    with Path("configs/xpool.example.toml").open("rb") as config_file:
        payload = tomllib.load(config_file)

    missing = [
        setting.name
        for setting in CONFIG_REGISTRY
        if setting.required
        and ConfigSource.CONFIG in setting.allowed_sources
        and setting.config_path is not None
        and not _has_required_path(payload, setting.config_path)
    ]

    assert missing == []


def test_old_explicit_topology_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "device_agents": [{"id": "gpu0", "cuda_device": 0, "nvshmem_rank": 0, "roles": ["attention"]}],
                "models": [{"id": "m", "path": "/models/m", "tp": 1}],
                "sglang_instances": [{"id": "m", "model": "m"}],
            },
            cli_overrides={},
        )


def test_model_metadata_fields_are_rejected_from_toml() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m", "tp": 1, "family": "dense", "hidden_size": 1}],
            },
            cli_overrides={},
        )


def test_duplicate_and_overlapping_devices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="devices.attention_cuda_devices must be unique"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0, 0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m", "tp": 1}],
            },
            cli_overrides={},
        )

    with pytest.raises(ValidationError, match="overlapping devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m", "tp": 1}],
            },
            cli_overrides={},
        )


def test_cli_config_default_precedence_without_config_field_env(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        daemon_host="from-config",
        daemon_port=1000,
        attention_concurrency=1,
        transport_concurrency=1,
    )

    config = load_config(
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
            "models": [{"id": "m", "path": "/models/m", "tp": 1}],
        },
        cli_overrides={},
    )

    assert config.daemon.host == "127.0.0.1"
    assert config.daemon.port == 9810
    assert config.scheduler.attention_concurrency == 1
    assert config.scheduler.transport_concurrency == 1


def test_config_resolution_does_not_mutate_caller_mapping() -> None:
    payload: dict[str, object] = {
        "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
        "models": [{"id": "m", "path": "/models/m", "tp": 1}],
    }
    original = deepcopy(payload)

    config = XpoolConfig.from_mapping(payload, cli_overrides={"daemon_host": "from-cli"})

    assert config.daemon.host == "from-cli"
    assert payload == original


def test_missing_required_top_level_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="devices"):
        XpoolConfig.from_mapping({"models": []}, cli_overrides={})


def test_missing_required_nested_field_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="ffn_cuda_devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m", "tp": 1}],
            },
            cli_overrides={},
        )


def test_config_path_precedence_uses_cli_before_env(tmp_path: Path) -> None:
    env_config = _write_minimal_config(tmp_path / "env", daemon_host="from-env-config")
    cli_config = _write_minimal_config(tmp_path / "cli", daemon_host="from-cli-config")

    config = load_config(config_path=cli_config, env={"XPOOL_CONFIG": str(env_config)})

    assert config.daemon.host == "from-cli-config"


def test_missing_config_path_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="config_path"):
        load_config(env={})


def test_registry_has_no_env_only_settings() -> None:
    registry = {item["name"]: item for item in config_registry_as_dict()}

    assert "config_path" not in registry
    assert "sglang_plugins" not in registry
    assert "preflight_require_mps" not in registry
    assert "capture_decode_buckets" not in registry


def test_model_path_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="path must be absolute"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "relative/model", "tp": 1}],
            },
            cli_overrides={},
        )


def test_model_tp_may_select_ffn_agent_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v2",
        hidden_size=2048,
        attention_heads=16,
        kv_heads=16,
        attention_kind=AttentionKind.MLA,
        physical_kv_lanes=1,
    )
    model_dir = _write_model_config(
        tmp_path / "deepseek-v2-lite-chat",
        {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 11008,
            "moe_intermediate_size": 1408,
        },
    )
    config = XpoolConfig.from_file(_write_minimal_config(tmp_path / "config", model_path=model_dir, tp=2))

    runtime = config.resolve_runtime()
    model = runtime.models[0]

    assert [agent.id for agent in runtime.device_agents] == ["cuda0", "cuda1", "cuda2", "cuda3", "cuda4"]
    assert model.spec.family == "deepseek_v2"
    assert model.spec.hidden_size == 2048
    assert model.spec.attention_kind is AttentionKind.MLA
    assert model.parallel_policy.sglang_tp_size == 2
    assert model.parallel_policy.attention_tp_size == 1
    assert model.parallel_policy.attention_dp_size == 2
    assert model.parallel_policy.enable_dp_attention is True
    assert model.parallel_policy.ffn_tp_size == 2
    assert model.ffn_agent_ids == ["cuda2", "cuda3"]


def test_gqa_attention_policy_is_derived_from_sglang_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=4,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=4,
    )
    model_dir = _write_model_config(
        tmp_path / "qwen",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 11008,
        },
    )
    config = XpoolConfig.from_file(_write_minimal_config(tmp_path / "config", model_path=model_dir, tp=1))

    policy = config.resolve_runtime().models[0].parallel_policy

    assert policy.attention_kind is AttentionKind.GQA
    assert policy.sglang_tp_size == 2
    assert policy.sglang_dp_size == 1
    assert policy.attention_tp_size == 2
    assert policy.attention_dp_size == 1


def test_deepseek_v4_is_regular_mqa_when_sglang_reports_non_mla(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v4",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=1,
        attention_kind=AttentionKind.MQA,
        physical_kv_lanes=1,
    )
    model_dir = _write_model_config(
        tmp_path / "deepseek-v4",
        {
            "model_type": "deepseek_v4",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "qk_rope_head_dim": 64,
            "intermediate_size": 11008,
        },
    )
    config = XpoolConfig.from_file(_write_minimal_config(tmp_path / "config", model_path=model_dir, tp=1))

    model = config.resolve_runtime().models[0]

    assert model.spec.attention_kind is AttentionKind.MQA
    assert model.spec.physical_kv_lanes == 1
    assert model.parallel_policy.attention_tp_size == 1


def test_zero_kv_heads_from_sglang_metadata_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sglang_metadata(
        monkeypatch,
        family="bad",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=0,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=0,
    )
    model_dir = _write_model_config(
        tmp_path / "bad",
        {
            "model_type": "bad",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 0,
            "intermediate_size": 11008,
        },
    )
    config = XpoolConfig.from_file(_write_minimal_config(tmp_path / "config", model_path=model_dir, tp=1))

    with pytest.raises(ConfigError, match="num_key_value_heads"):
        config.resolve_runtime()


def test_model_tp_must_fit_configured_ffn_devices() -> None:
    with pytest.raises(ValidationError, match="requests tp=4"):
        XpoolConfig.from_mapping(
            {
                "devices": {"attention_cuda_devices": [0], "ffn_cuda_devices": [1, 2]},
                "models": [{"id": "m", "path": "/models/m", "tp": 4}],
            },
            cli_overrides={},
        )


def test_ffn_tp_divisibility_is_checked_during_runtime_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=4,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=4,
    )
    model_dir = _write_model_config(
        tmp_path / "bad-ffn",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 3,
        },
    )
    config = XpoolConfig.from_file(_write_minimal_config(tmp_path / "config", model_path=model_dir, tp=2))

    with pytest.raises(TopologyError, match="does not divide dense intermediate size"):
        config.resolve_runtime()


def _write_minimal_config(
    path: Path,
    *,
    daemon_host: str = "127.0.0.1",
    daemon_port: int = 9810,
    attention_concurrency: int = 1,
    transport_concurrency: int = 1,
    model_path: Path | None = None,
    tp: int = 1,
) -> Path:
    if path.suffix != ".toml":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "xpool.toml"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = model_path or Path("/models/deepseek-v2-lite-chat")
    path.write_text(
        f"""
[daemon]
host = "{daemon_host}"
port = {daemon_port}

[scheduler]
attention_concurrency = {attention_concurrency}
transport_concurrency = {transport_concurrency}

[devices]
attention_cuda_devices = [0, 1]
ffn_cuda_devices = [2, 3, 4]

[[models]]
id = "deepseek-v2-lite-chat"
path = "{resolved_model_path}"
tp = {tp}
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_model_config(path: Path, payload: Mapping[str, object]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_sglang_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: str,
    hidden_size: int,
    attention_heads: int,
    kv_heads: int,
    attention_kind: AttentionKind,
    physical_kv_lanes: int,
) -> None:
    def load_metadata(_config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=family or model_id,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            attention_kind=attention_kind,
            physical_kv_lanes=physical_kv_lanes,
        )

    monkeypatch.setattr(config_module, "_load_sglang_model_metadata", load_metadata)


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
