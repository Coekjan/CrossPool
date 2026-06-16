"""Configuration schema, source registry, and derived topology for xpool."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigSource(StrEnum):
    CLI = "cli"
    CONFIG = "config"
    DEFAULT = "default"


ParserName = Literal["int", "raw", "str"]


class ConfigError(ValueError):
    """Base error for xpool config resolution failures."""


class MissingRequiredConfig(ConfigError):
    """Raised when a required setting has no value from any allowed source."""


class TopologyError(ConfigError):
    """Raised when model or device topology cannot be derived safely."""


class DeviceRole(StrEnum):
    ATTENTION = "attention"
    FFN = "ffn"


class AttentionKind(StrEnum):
    MLA = "mla"
    GQA = "gqa"
    MHA = "mha"
    MQA = "mqa"


@dataclass(frozen=True, slots=True)
class SglangModelMetadata:
    """Model shape facts derived through SGLang's config resolution path."""

    family: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_kind: AttentionKind
    physical_kv_lanes: int


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """Registry entry for one TOML, CLI, or defaulted setting."""

    name: str
    config_path: tuple[str, ...] | None
    parser: ParserName
    allowed_sources: tuple[ConfigSource, ...]
    description: str
    default: object = None
    required: bool = False
    cli: str | None = None

    @property
    def has_default(self) -> bool:
        return ConfigSource.DEFAULT in self.allowed_sources

    @property
    def is_config_field(self) -> bool:
        return self.config_path is not None and "*" not in self.config_path

    @property
    def is_required_config_path(self) -> bool:
        return self.config_path is not None and self.required


def parse_int(value: object) -> int:
    return int(str(value).strip())


def parse_str(value: object) -> str:
    return str(value)


def parse_raw(value: object) -> object:
    return value


TOP_LEVEL_SOURCES = (
    ConfigSource.CLI,
    ConfigSource.CONFIG,
    ConfigSource.DEFAULT,
)
CONFIG_REQUIRED = (ConfigSource.CONFIG,)

CONFIG_REGISTRY: tuple[ConfigSetting, ...] = (
    ConfigSetting(
        name="daemon_host",
        config_path=("daemon", "host"),
        parser="str",
        allowed_sources=TOP_LEVEL_SOURCES,
        default="127.0.0.1",
        cli="--daemon-host",
        description="Daemon control-plane bind host.",
    ),
    ConfigSetting(
        name="daemon_port",
        config_path=("daemon", "port"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=9810,
        cli="--daemon-port",
        description="Daemon control-plane bind port.",
    ),
    ConfigSetting(
        name="scheduler_attention_concurrency",
        config_path=("scheduler", "attention_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--attention-concurrency",
        description="Maximum concurrent attention owners per attention CUDA device.",
    ),
    ConfigSetting(
        name="scheduler_transport_concurrency",
        config_path=("scheduler", "transport_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--transport-concurrency",
        description="Maximum concurrent transport-slot owners per attention CUDA device.",
    ),
    ConfigSetting(
        name="devices",
        config_path=("devices",),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Role-local CUDA device lists. Device agents and NVSHMEM ranks are derived from this section.",
    ),
    ConfigSetting(
        name="attention_cuda_devices",
        config_path=("devices", "attention_cuda_devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host SGLang attention and attention-side xpool device agents.",
    ),
    ConfigSetting(
        name="ffn_cuda_devices",
        config_path=("devices", "ffn_cuda_devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host FFN-side xpool device agents.",
    ),
    ConfigSetting(
        name="models",
        config_path=("models",),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Served model registry. Each model derives one SGLang instance.",
    ),
    ConfigSetting(
        name="model_id",
        config_path=("models", "*", "id"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Full model instance id, for example deepseek-v2-lite-chat.",
    ),
    ConfigSetting(
        name="model_path",
        config_path=("models", "*", "path"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Absolute local model path containing config.json.",
    ),
    ConfigSetting(
        name="model_tp",
        config_path=("models", "*", "tp"),
        parser="int",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="FFN tensor-parallel degree requested by the model.",
    ),
)


class DaemonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(ge=1, le=65535)


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attention_concurrency: int = Field(ge=1)
    transport_concurrency: int = Field(ge=1)


class DevicesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attention_cuda_devices: list[int] = Field(min_length=1)
    ffn_cuda_devices: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_devices(self) -> "DevicesConfig":
        _validate_cuda_device_list(self.attention_cuda_devices, "devices.attention_cuda_devices")
        _validate_cuda_device_list(self.ffn_cuda_devices, "devices.ffn_cuda_devices")
        overlap = sorted(set(self.attention_cuda_devices) & set(self.ffn_cuda_devices))
        if overlap:
            raise ValueError(f"CUDA devices may host only one xpool role; overlapping devices: {overlap}")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    path: Path
    tp: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_model_path(self) -> "ModelConfig":
        path = self.path.expanduser()
        if not path.is_absolute():
            raise ValueError(f"models[{self.id}].path must be absolute: {self.path}")
        self.path = path
        return self


class DeviceAgentConfig(BaseModel):
    """Derived device-agent placement.

    This is not a TOML schema item. It is derived from the role-local CUDA
    device lists, with one unique xpool agent per participating CUDA device.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    cuda_device: int = Field(ge=0)
    nvshmem_rank: int = Field(ge=0)
    role: DeviceRole

    @property
    def roles(self) -> list[DeviceRole]:
        return [self.role]


class ModelInstanceConfig(BaseModel):
    """Derived SGLang instance placement for one configured model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model_id: str
    attention_cuda_devices: list[int] = Field(min_length=1)
    ffn_agent_ids: list[str] = Field(min_length=1)
    ffn_tp_size: int = Field(ge=1)
    sglang_tp_size: int = Field(ge=1)


class ModelSpec(BaseModel):
    """Model metadata resolved from a local config.json through SGLang."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    family: str
    hidden_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_key_value_heads: int = Field(ge=1)
    attention_kind: AttentionKind
    physical_kv_lanes: int = Field(ge=1)
    dense_intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    num_experts: int | None = None
    raw_config_path: Path


class ParallelPolicy(BaseModel):
    """Model-derived role-local policy for one xpool model instance."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    attention_kind: AttentionKind
    sglang_tp_size: int = Field(ge=1)
    sglang_dp_size: int = Field(ge=1)
    attention_tp_size: int = Field(ge=1)
    attention_dp_size: int = Field(ge=1)
    ffn_tp_size: int = Field(ge=1)
    physical_kv_lanes: int = Field(ge=1)
    enable_dp_attention: bool


class ResolvedModelConfig(BaseModel):
    """Configured model plus metadata and derived runtime policy."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path
    ffn_agent_ids: list[str]
    spec: ModelSpec
    parallel_policy: ParallelPolicy


class ResolvedRuntimeConfig(BaseModel):
    """Runtime view derived from TOML and SGLang-resolved model metadata."""

    model_config = ConfigDict(extra="forbid")

    daemon: DaemonConfig
    scheduler: SchedulerConfig
    device_agents: list[DeviceAgentConfig]
    sglang_instances: list[ModelInstanceConfig]
    models: list[ResolvedModelConfig]


class XpoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daemon: DaemonConfig
    scheduler: SchedulerConfig
    devices: DevicesConfig
    models: list[ModelConfig] = Field(min_length=1)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cli_overrides: Mapping[str, object] | None = None,
    ) -> "XpoolConfig":
        config_path = Path(path)
        with config_path.open("rb") as config_file:
            payload = tomllib.load(config_file)
        return cls.from_mapping(payload, cli_overrides=cli_overrides)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        cli_overrides: Mapping[str, object] | None = None,
    ) -> "XpoolConfig":
        resolved = resolve_config_payload(payload, cli_overrides=cli_overrides)
        return cls.model_validate(resolved)

    @property
    def device_agents(self) -> list[DeviceAgentConfig]:
        return derive_device_agents(self.devices)

    @property
    def sglang_instances(self) -> list[ModelInstanceConfig]:
        return derive_model_instances(self)

    @model_validator(mode="after")
    def validate_references(self) -> "XpoolConfig":
        _require_unique([model.id for model in self.models], "model ids")
        ffn_device_count = len(self.devices.ffn_cuda_devices)
        for model in self.models:
            if model.tp > ffn_device_count:
                raise ValueError(
                    f"model {model.id} requests tp={model.tp}, but only {ffn_device_count} FFN devices are configured"
                )
        return self

    def resolve_runtime(self) -> ResolvedRuntimeConfig:
        device_agents = self.device_agents
        instances = self.sglang_instances
        models: list[ResolvedModelConfig] = []
        for model, instance in zip(self.models, instances, strict=True):
            spec = load_model_spec(model.path, model_id=model.id)
            policy = derive_parallel_policy(
                spec,
                attention_device_count=len(self.devices.attention_cuda_devices),
                ffn_tp_size=model.tp,
            )
            models.append(
                ResolvedModelConfig(
                    id=model.id,
                    path=model.path,
                    ffn_agent_ids=instance.ffn_agent_ids,
                    spec=spec,
                    parallel_policy=policy,
                )
            )
        return ResolvedRuntimeConfig(
            daemon=self.daemon,
            scheduler=self.scheduler,
            device_agents=device_agents,
            sglang_instances=instances,
            models=models,
        )


def load_config(
    *,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> XpoolConfig:
    effective_env = os.environ if env is None else env
    effective_path = config_path or effective_env.get("XPOOL_CONFIG")
    if effective_path is None:
        raise MissingRequiredConfig("missing required config setting: config_path")
    return XpoolConfig.from_file(effective_path, cli_overrides=cli_overrides)


def load_model_spec(model_path: str | Path, *, model_id: str) -> ModelSpec:
    path = Path(model_path).expanduser()
    config_path = path if path.is_file() else path / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = cast(Mapping[str, object], json.load(config_file))
    return parse_model_config(raw, model_id=model_id, config_path=config_path.resolve())


def parse_model_config(raw: Mapping[str, object], *, model_id: str, config_path: Path) -> ModelSpec:
    sglang = _validate_sglang_model_metadata(
        _load_sglang_model_metadata(config_path, model_id=model_id), model_id=model_id
    )
    return ModelSpec(
        model_id=model_id,
        family=sglang.family,
        hidden_size=sglang.hidden_size,
        num_attention_heads=sglang.num_attention_heads,
        num_key_value_heads=sglang.num_key_value_heads,
        attention_kind=sglang.attention_kind,
        physical_kv_lanes=sglang.physical_kv_lanes,
        dense_intermediate_size=_optional_int(raw, "intermediate_size"),
        moe_intermediate_size=_optional_int(raw, "moe_intermediate_size"),
        num_experts=_optional_int(raw, "num_experts") or _optional_int(raw, "n_routed_experts"),
        raw_config_path=config_path,
    )


def derive_parallel_policy(
    spec: ModelSpec,
    *,
    attention_device_count: int,
    ffn_tp_size: int,
) -> ParallelPolicy:
    if attention_device_count <= 0:
        raise TopologyError("attention_device_count must be positive")
    if ffn_tp_size <= 0:
        raise TopologyError("ffn_tp_size must be positive")

    if spec.attention_kind is AttentionKind.MLA:
        attention_tp_size = 1
    else:
        attention_tp_size = min(spec.num_key_value_heads, attention_device_count)
        if spec.num_attention_heads % attention_tp_size != 0:
            raise TopologyError(
                f"{spec.model_id}: attention TP {attention_tp_size} does not divide query heads "
                f"{spec.num_attention_heads}"
            )
        if spec.physical_kv_lanes % attention_tp_size != 0:
            raise TopologyError(
                f"{spec.model_id}: attention TP {attention_tp_size} does not divide KV lanes {spec.physical_kv_lanes}"
            )

    if attention_device_count % attention_tp_size != 0:
        raise TopologyError(
            f"{spec.model_id}: attention TP {attention_tp_size} does not divide attention device count "
            f"{attention_device_count}"
        )
    _validate_ffn_tp_divisibility(spec, ffn_tp_size)
    attention_dp_size = attention_device_count // attention_tp_size
    return ParallelPolicy(
        model_id=spec.model_id,
        attention_kind=spec.attention_kind,
        sglang_tp_size=attention_device_count,
        sglang_dp_size=attention_dp_size,
        attention_tp_size=attention_tp_size,
        attention_dp_size=attention_dp_size,
        ffn_tp_size=ffn_tp_size,
        physical_kv_lanes=spec.physical_kv_lanes,
        enable_dp_attention=attention_dp_size > 1,
    )


def derive_device_agents(devices: DevicesConfig) -> list[DeviceAgentConfig]:
    agents: list[DeviceAgentConfig] = []
    for nvshmem_rank, cuda_device in enumerate(devices.attention_cuda_devices + devices.ffn_cuda_devices):
        role = DeviceRole.ATTENTION if nvshmem_rank < len(devices.attention_cuda_devices) else DeviceRole.FFN
        agents.append(
            DeviceAgentConfig(
                id=f"cuda{cuda_device}",
                cuda_device=cuda_device,
                nvshmem_rank=nvshmem_rank,
                role=role,
            )
        )
    return agents


def derive_model_instances(config: XpoolConfig) -> list[ModelInstanceConfig]:
    ffn_agents = [agent for agent in config.device_agents if agent.role is DeviceRole.FFN]
    return [
        ModelInstanceConfig(
            id=model.id,
            model_id=model.id,
            attention_cuda_devices=list(config.devices.attention_cuda_devices),
            ffn_agent_ids=[agent.id for agent in ffn_agents[: model.tp]],
            ffn_tp_size=model.tp,
            sglang_tp_size=len(config.devices.attention_cuda_devices),
        )
        for model in config.models
    ]


def resolve_config_payload(
    payload: Mapping[str, object],
    *,
    cli_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved: dict[str, object] = deepcopy(dict(payload))
    effective_cli = cli_overrides or {}
    for setting in CONFIG_REGISTRY:
        if not setting.is_config_field:
            continue
        value, source = _resolve_setting(setting, resolved, effective_cli)
        if source is not None:
            _set_nested(resolved, setting.config_path or (), value)

    _validate_required_config_paths(resolved)
    return resolved


def config_registry_as_dict() -> list[dict[str, object]]:
    return [
        {
            "name": setting.name,
            "config_path": ".".join(setting.config_path) if setting.config_path else None,
            "default": setting.default if setting.has_default else None,
            "has_default": setting.has_default,
            "required": setting.required,
            "parser": setting.parser,
            "allowed_sources": [source.value for source in setting.allowed_sources],
            "cli": setting.cli,
            "description": setting.description,
        }
        for setting in CONFIG_REGISTRY
    ]


def _resolve_setting(
    setting: ConfigSetting,
    payload: Mapping[str, object],
    cli_overrides: Mapping[str, object],
) -> tuple[object, ConfigSource | None]:
    if ConfigSource.CLI in setting.allowed_sources and setting.name in cli_overrides:
        return _parse_setting(setting, cli_overrides[setting.name]), ConfigSource.CLI
    if ConfigSource.CONFIG in setting.allowed_sources:
        found, config_value = _get_nested(payload, setting.config_path or ())
        if found:
            return _parse_setting(setting, config_value), ConfigSource.CONFIG
    if setting.has_default:
        return _parse_setting(setting, setting.default), ConfigSource.DEFAULT
    if setting.required:
        raise MissingRequiredConfig(f"missing required config setting: {setting.name}")
    return None, None


def _parse_setting(setting: ConfigSetting, value: object) -> object:
    if setting.parser == "int":
        return parse_int(value)
    if setting.parser == "raw":
        return parse_raw(value)
    if setting.parser == "str":
        return parse_str(value)
    raise ConfigError(f"unknown parser for {setting.name}: {setting.parser}")


def _validate_required_config_paths(payload: Mapping[str, object]) -> None:
    for setting in CONFIG_REGISTRY:
        if not setting.is_required_config_path:
            continue
        path = setting.config_path or ()
        if "*" in path:
            _validate_required_wildcard_path(payload, setting)
            continue
        found, value = _get_nested(payload, path)
        if not found or _is_empty_required_value(value):
            raise MissingRequiredConfig(f"missing required config setting: {setting.name}")


def _validate_required_wildcard_path(payload: Mapping[str, object], setting: ConfigSetting) -> None:
    path = setting.config_path or ()
    wildcard_index = path.index("*")
    collection_path = path[:wildcard_index]
    item_path = path[wildcard_index + 1 :]
    found, collection = _get_nested(payload, collection_path)
    if not found or not isinstance(collection, list):
        raise MissingRequiredConfig(f"missing required config setting: {'.'.join(collection_path)}")
    for index, item in enumerate(collection):
        if not isinstance(item, Mapping):
            raise MissingRequiredConfig(f"missing required config setting: {'.'.join(collection_path)}[{index}]")
        found, value = _get_nested(cast(Mapping[str, object], item), item_path)
        if not found or _is_empty_required_value(value):
            raise MissingRequiredConfig(
                f"missing required config setting: {'.'.join(collection_path)}[{index}].{'.'.join(item_path)}"
            )


def _is_empty_required_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value:
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def _get_nested(payload: Mapping[str, object], path: tuple[str, ...]) -> tuple[bool, object]:
    cursor: object = payload
    for key in path:
        if not isinstance(cursor, Mapping):
            return False, None
        mapping = cast(Mapping[str, object], cursor)
        if key not in mapping:
            return False, None
        cursor = mapping[key]
    return True, cursor


def _set_nested(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    if not path:
        raise ConfigError("cannot set empty config path")
    cursor = payload
    for key in path[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"cannot override nested config path {'.'.join(path)}")
        cursor = cast(dict[str, object], child)
    cursor[path[-1]] = value


def _validate_cuda_device_list(values: Sequence[int], label: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f"{label} must contain non-negative CUDA device indices")
    _require_unique(values, label)


def _require_unique(values: Sequence[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    return _coerce_config_int(value, key)


def _coerce_config_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"model config.json field {key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"model config.json field {key} must be an integer") from exc
    raise ConfigError(f"model config.json field {key} must be an integer")


def _load_sglang_model_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
    model_path = config_path.parent if config_path.name == "config.json" else config_path
    try:
        from sglang.srt.configs.model_config import AttentionArch as SglangAttentionArch
        from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig
    except Exception as exc:  # pragma: no cover - exercised only when runtime deps are missing
        raise ConfigError("SGLang is required to derive xpool model topology") from exc

    try:
        sglang_config = SglangModelConfig(str(model_path), trust_remote_code=True)
    except Exception as exc:
        raise ConfigError(f"{model_id}: SGLang failed to resolve model config at {model_path}") from exc

    family = str(
        getattr(sglang_config.hf_text_config, "model_type", None)
        or getattr(sglang_config.hf_config, "model_type", None)
        or model_id
    )
    hidden_size = _positive_runtime_int(getattr(sglang_config, "hidden_size", None), "hidden_size", model_id)
    attention_heads = _positive_runtime_int(
        sglang_config.get_total_num_attention_heads(),
        "num_attention_heads",
        model_id,
    )
    kv_heads = _positive_runtime_int(sglang_config.get_total_num_kv_heads(), "num_key_value_heads", model_id)

    if sglang_config.attention_arch == SglangAttentionArch.MLA:
        _validate_sglang_mla_dimensions(sglang_config, model_id=model_id)
        return SglangModelMetadata(
            family=family,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            attention_kind=AttentionKind.MLA,
            physical_kv_lanes=1,
        )

    attention_kind = _derive_regular_attention_kind(kv_heads=kv_heads, attention_heads=attention_heads)
    return SglangModelMetadata(
        family=family,
        hidden_size=hidden_size,
        num_attention_heads=attention_heads,
        num_key_value_heads=kv_heads,
        attention_kind=attention_kind,
        physical_kv_lanes=kv_heads,
    )


def _validate_sglang_model_metadata(metadata: SglangModelMetadata, *, model_id: str) -> SglangModelMetadata:
    for label, value in (
        ("hidden_size", metadata.hidden_size),
        ("num_attention_heads", metadata.num_attention_heads),
        ("num_key_value_heads", metadata.num_key_value_heads),
        ("physical_kv_lanes", metadata.physical_kv_lanes),
    ):
        if value <= 0:
            raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    if metadata.attention_kind is AttentionKind.MLA and metadata.physical_kv_lanes != 1:
        raise ConfigError(f"{model_id}: SGLang-derived MLA physical_kv_lanes must be 1")
    if metadata.attention_kind is not AttentionKind.MLA and metadata.physical_kv_lanes != metadata.num_key_value_heads:
        raise ConfigError(f"{model_id}: SGLang-derived regular physical_kv_lanes must match num_key_value_heads")
    return metadata


def _derive_regular_attention_kind(*, kv_heads: int, attention_heads: int) -> AttentionKind:
    if kv_heads == 1:
        return AttentionKind.MQA
    if kv_heads == attention_heads:
        return AttentionKind.MHA
    return AttentionKind.GQA


def _validate_sglang_mla_dimensions(sglang_config: object, *, model_id: str) -> None:
    for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        _positive_runtime_int(getattr(sglang_config, name, None), name, model_id)


def _positive_runtime_int(value: object, label: str, model_id: str) -> int:
    if value is None:
        raise ConfigError(f"{model_id}: SGLang-derived {label} is missing")
    parsed = _coerce_config_int(value, label)
    if parsed <= 0:
        raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    return parsed


def _validate_ffn_tp_divisibility(spec: ModelSpec, ffn_tp_size: int) -> None:
    for label, width in (
        ("dense intermediate size", spec.dense_intermediate_size),
        ("MoE intermediate size", spec.moe_intermediate_size),
    ):
        if width is not None and width % ffn_tp_size != 0:
            raise TopologyError(f"{spec.model_id}: FFN TP {ffn_tp_size} does not divide {label} {width}")
