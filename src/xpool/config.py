"""Configuration schema, source registry, and static placement for xpool."""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class ConfigSource(StrEnum):
    """Configuration value source used by the xpool setting registry.

    Attributes:
        CLI: Value came from an explicit xpool CLI override.
        ENV: Value came from an allowlisted process environment variable.
        CONFIG: Value came from the TOML config file.
        DEFAULT: Value came from a registry default.
    """

    CLI = "cli"
    ENV = "env"
    CONFIG = "config"
    DEFAULT = "default"


ParserName = Literal["bool", "int", "raw", "str"]


class ConfigError(ValueError):
    """Base error for xpool config resolution failures."""


class MissingRequiredConfig(ConfigError):
    """Raised when a required setting has no value from any allowed source."""


class TopologyError(ConfigError):
    """Raised when model or device topology cannot be derived safely."""


class DeviceRole(StrEnum):
    """Exclusive role assigned to one CUDA device agent.

    Attributes:
        ATTENTION: Device hosts attention execution and the attention-side shim agent.
        FFN: Device hosts xpool FFN execution.
    """

    ATTENTION = "attention"
    FFN = "ffn"


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """Registry entry for one TOML, CLI, or defaulted setting.

    Attributes:
        name: Stable registry key used by CLI overrides and diagnostics.
        path: Canonical ``XpoolConfig`` field path, or ``None`` for bootstrap-only settings.
        parser: Parser name used to normalize raw source values.
        allowed_sources: Sources allowed to provide this setting.
        description: Human-readable setting purpose for generated registry output.
        default: Default value used when ``DEFAULT`` is an allowed source.
        required: Whether missing values are configuration errors.
        cli: CLI flag name when the setting is overrideable from the command line.
        env_var: Environment variable name when the setting is process-env backed.
    """

    name: str
    path: tuple[str, ...] | None
    parser: ParserName
    allowed_sources: tuple[ConfigSource, ...]
    description: str
    default: object = None
    required: bool = False
    cli: str | None = None
    env_var: str | None = None


TOP_LEVEL_SOURCES = (
    ConfigSource.CLI,
    ConfigSource.CONFIG,
    ConfigSource.DEFAULT,
)
CONFIG_REQUIRED = (ConfigSource.CONFIG,)

CONFIG_REGISTRY: tuple[ConfigSetting, ...] = (
    ConfigSetting(
        name="config_path",
        path=None,
        parser="str",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV),
        cli="--config",
        env_var="XPOOL_CONFIG",
        description="Bootstrap TOML config path used before repository config can be loaded.",
    ),
    ConfigSetting(
        name="debug_enable_shim_loopback",
        path=("debug", "enable_shim_loopback"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_ENABLE_SHIM_LOOPBACK",
        description="Development-only switch that routes FFN shim calls to the debug loopback op.",
    ),
    ConfigSetting(
        name="daemon_host",
        path=("daemon", "host"),
        parser="str",
        allowed_sources=TOP_LEVEL_SOURCES,
        default="127.0.0.1",
        cli="--daemon-host",
        description="Daemon control-plane bind host.",
    ),
    ConfigSetting(
        name="daemon_port",
        path=("daemon", "port"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=9810,
        cli="--daemon-port",
        description="Daemon control-plane bind port.",
    ),
    ConfigSetting(
        name="scheduler_attention_concurrency",
        path=("scheduler", "attention_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--attention-concurrency",
        description="Maximum concurrent attention owners per attention CUDA device.",
    ),
    ConfigSetting(
        name="scheduler_transport_concurrency",
        path=("scheduler", "transport_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--transport-concurrency",
        description="Maximum concurrent transport-slot owners per attention CUDA device.",
    ),
    ConfigSetting(
        name="devices",
        path=("devices",),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description=(
            "Role-local CUDA device lists. Device agents and communication ranks are derived from this section."
        ),
    ),
    ConfigSetting(
        name="attention_cuda_devices",
        path=("devices", "attention_cuda_devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host attention execution and attention-side xpool device agents.",
    ),
    ConfigSetting(
        name="ffn_cuda_devices",
        path=("devices", "ffn_cuda_devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host FFN-side xpool device agents.",
    ),
    ConfigSetting(
        name="models",
        path=("models",),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Served model registry. Each model derives one serving instance.",
    ),
    ConfigSetting(
        name="model_id",
        path=("models", "*", "id"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Full model instance id, for example deepseek-v2-lite-chat.",
    ),
    ConfigSetting(
        name="model_path",
        path=("models", "*", "path"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Absolute local model path containing config.json.",
    ),
)


class DaemonConfig(BaseModel):
    """Daemon control-plane bind settings from config, CLI, or defaults."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(description="Host or interface address used by the daemon HTTP control plane.")
    port: int = Field(ge=1, le=65535, description="TCP port used by the daemon HTTP control plane.")


class SchedulerConfig(BaseModel):
    """Conservative resource-concurrency limits enforced by device agents."""

    model_config = ConfigDict(extra="forbid")

    attention_concurrency: int = Field(
        ge=1,
        description="Maximum number of concurrent attention owners per attention CUDA device.",
    )
    transport_concurrency: int = Field(
        ge=1,
        description="Maximum number of concurrent communication-slot owners per attention CUDA device.",
    )


class DevicesConfig(BaseModel):
    """Role-local CUDA device lists supplied by TOML config."""

    model_config = ConfigDict(extra="forbid")

    attention_cuda_devices: list[int] = Field(
        min_length=1,
        description="CUDA device indices that host attention execution and attention-side xpool agents.",
    )
    ffn_cuda_devices: list[int] = Field(
        min_length=1,
        description="CUDA device indices that host xpool FFN execution agents.",
    )

    @model_validator(mode="after")
    def validate_devices(self) -> "DevicesConfig":
        """Reject duplicate or role-overlapping CUDA device lists.

        Returns:
            The validated device config.

        Raises:
            ValueError: If a CUDA device is duplicated or assigned to both roles.
        """

        for label, values in (
            ("devices.attention_cuda_devices", self.attention_cuda_devices),
            ("devices.ffn_cuda_devices", self.ffn_cuda_devices),
        ):
            if any(value < 0 for value in values):
                raise ValueError(f"{label} must contain non-negative CUDA device indices")
            _require_unique(values, label)
        overlap = sorted(set(self.attention_cuda_devices) & set(self.ffn_cuda_devices))
        if overlap:
            raise ValueError(f"CUDA devices may host only one xpool role; overlapping devices: {overlap}")
        return self


class ModelConfig(BaseModel):
    """User-declared model served by one xpool-managed serving instance."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Full model instance id, for example deepseek-v2-lite-chat.")
    path: Path = Field(description="Absolute local model path containing config.json.")

    @model_validator(mode="after")
    def validate_model_path(self) -> "ModelConfig":
        """Normalize and validate the configured model path.

        Returns:
            The validated model config with ``~`` expanded.

        Raises:
            ValueError: If the configured path is not absolute.
        """

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

    id: str = Field(description="Stable device-agent id derived from the CUDA device index.")
    cuda_device: int = Field(ge=0, description="CUDA device index owned by this device agent.")
    nvshmem_rank: int = Field(ge=0, description="NVSHMEM rank assigned by role-local device declaration order.")
    role: DeviceRole = Field(description="Exclusive runtime role hosted by this CUDA device.")

    @property
    def roles(self) -> list[DeviceRole]:
        """Return this agent's role as a list for launch-plan compatibility.

        Returns:
            Single-item role list; one CUDA device may host only one xpool role.
        """

        return [self.role]


class ServingInstanceConfig(BaseModel):
    """Derived serving-instance placement for one configured model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Serving instance id, equal to the configured model id in the first topology.")
    model_id: str = Field(description="Configured model id served by this instance.")
    attention_cuda_devices: list[int] = Field(
        min_length=1,
        description="Attention CUDA devices assigned to this serving instance.",
    )
    ffn_agent_ids: list[str] = Field(min_length=1, description="FFN device-agent ids used by this model.")
    ffn_tp_size: int = Field(ge=1, description="FFN tensor-parallel degree derived from FFN device count.")
    attention_world_size: int = Field(ge=1, description="Attention-side world size derived from attention devices.")
    instance_index: int = Field(ge=0, description="Integer serving-instance index fed to the native shim ABI.")
    model_index: int = Field(ge=0, description="Integer model index fed to the native shim ABI.")


class DebugConfig(BaseModel):
    """Debug-only runtime switches resolved through the config registry."""

    model_config = ConfigDict(extra="forbid")

    enable_shim_loopback: bool = Field(
        default=False,
        description="Whether FFN shim modules should call the debug loopback native op instead of production shim.",
    )


class XpoolConfig(BaseModel):
    """Validated xpool TOML config plus derived runtime views."""

    model_config = ConfigDict(extra="forbid")

    daemon: DaemonConfig = Field(description="Daemon control-plane config.")
    scheduler: SchedulerConfig = Field(description="Scheduler resource-concurrency config.")
    debug: DebugConfig = Field(description="Debug-only runtime switches.")
    devices: DevicesConfig = Field(description="Role-local CUDA device config.")
    models: list[ModelConfig] = Field(min_length=1, description="Configured served model list.")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cli_overrides: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "XpoolConfig":
        """Load and validate an xpool TOML file.

        Args:
            path: Path to the TOML config file.
            cli_overrides: Optional CLI-derived setting overrides that take
                precedence over TOML values.
            env: Optional allowlisted environment settings. Only registry
                entries with ``ENV`` in ``allowed_sources`` may read it.

        Returns:
            Validated config object with defaults and CLI overrides resolved.

        Raises:
            OSError: If the file cannot be opened.
            tomllib.TOMLDecodeError: If the file is not valid TOML.
            ConfigError: If registry resolution fails.
            pydantic.ValidationError: If schema validation fails.
        """

        config_path = Path(path)
        with config_path.open("rb") as config_file:
            payload = tomllib.load(config_file)
        return cls.from_mapping(payload, cli_overrides=cli_overrides, env=env)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        cli_overrides: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "XpoolConfig":
        """Validate an in-memory config mapping.

        Args:
            payload: TOML-like mapping to validate. The mapping is deep-copied
                before defaults or overrides are applied.
            cli_overrides: Optional CLI-derived setting overrides that take
                precedence over mapping values.
            env: Optional allowlisted environment settings. Only registry
                entries with ``ENV`` in ``allowed_sources`` may read it.

        Returns:
            Validated config object.

        Raises:
            ConfigError: If registry resolution fails.
            pydantic.ValidationError: If schema validation fails.

        Side Effects:
            Does not mutate ``payload``.
        """

        resolved: dict[str, object] = deepcopy(dict(payload))
        effective_cli = cli_overrides or {}
        effective_env = env or {}
        if env is not None:
            allowed_env_vars = frozenset(setting.env_var for setting in CONFIG_REGISTRY if setting.env_var is not None)
            unknown_env_vars = tuple(
                sorted(name for name in effective_env if name.startswith("XPOOL_") and name not in allowed_env_vars)
            )
            if unknown_env_vars:
                logger.warning("Ignoring unknown xpool environment variables: %s", ", ".join(unknown_env_vars))
        for setting in CONFIG_REGISTRY:
            if setting.path is None or ConfigSource.CONFIG in setting.allowed_sources:
                continue
            found, _value = _get_nested(resolved, setting.path)
            if found:
                raise ConfigError(f"config setting {setting.name} does not allow TOML source: {'.'.join(setting.path)}")
        for setting in CONFIG_REGISTRY:
            if setting.path is None or "*" in setting.path:
                continue
            value, source = _resolve_setting(setting, resolved, effective_cli, effective_env)
            if source is not None:
                _set_nested(resolved, setting.path, value)

        _validate_required_config_paths(resolved)
        return cls.model_validate(resolved)

    @property
    def device_agents(self) -> list[DeviceAgentConfig]:
        """Derive one device agent for every configured CUDA device.

        Returns:
            Device-agent placement list ordered by NVSHMEM rank.
        """

        agents: list[DeviceAgentConfig] = []
        for nvshmem_rank, cuda_device in enumerate(self.devices.attention_cuda_devices + self.devices.ffn_cuda_devices):
            role = DeviceRole.ATTENTION if nvshmem_rank < len(self.devices.attention_cuda_devices) else DeviceRole.FFN
            agents.append(
                DeviceAgentConfig(
                    id=f"cuda{cuda_device}",
                    cuda_device=cuda_device,
                    nvshmem_rank=nvshmem_rank,
                    role=role,
                )
            )
        return agents

    @property
    def serving_instances(self) -> list[ServingInstanceConfig]:
        """Derive one serving instance for every configured model.

        Returns:
            Instance placement list in model declaration order.
        """

        ffn_agent_ids = [f"cuda{cuda_device}" for cuda_device in self.devices.ffn_cuda_devices]
        ffn_tp_size = len(ffn_agent_ids)
        return [
            ServingInstanceConfig(
                id=model.id,
                model_id=model.id,
                attention_cuda_devices=list(self.devices.attention_cuda_devices),
                ffn_agent_ids=ffn_agent_ids,
                ffn_tp_size=ffn_tp_size,
                attention_world_size=len(self.devices.attention_cuda_devices),
                instance_index=index,
                model_index=index,
            )
            for index, model in enumerate(self.models)
        ]

    @property
    def model_index_by_path(self) -> dict[Path, int]:
        """Map each model's resolved absolute path to its declaration-order index."""

        return {model.path.resolve(): index for index, model in enumerate(self.models)}

    @model_validator(mode="after")
    def validate_references(self) -> "XpoolConfig":
        """Reject duplicate model identities and paths.

        Returns:
            The validated config object.

        Raises:
            ValueError: If model ids or resolved model paths are duplicated.
        """

        _require_unique([model.id for model in self.models], "model ids")
        _require_unique([model.path.resolve() for model in self.models], "model paths")
        return self


_global_config: XpoolConfig | None = None
_global_config_lock = Lock()


def init_global_config(
    *,
    config: XpoolConfig | None = None,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> XpoolConfig:
    """Initialize the process-global xpool config.

    Args:
        config: Optional already validated config to install directly. This is
            intended for tests and narrow internal setup paths.
        config_path: Explicit TOML config path. When provided, it takes
            precedence over ``env["XPOOL_CONFIG"]``.
        env: Environment mapping used by the registry. Defaults to ``os.environ``.
        cli_overrides: Optional CLI-derived setting overrides.

    Returns:
        The config object now returned by :func:`get_global_config`.

    Raises:
        ConfigError: If direct ``config`` injection is mixed with file/env/CLI
            inputs, or if registry resolution fails.
        MissingRequiredConfig: If no config path is available.
        OSError: If the config file cannot be opened.
        tomllib.TOMLDecodeError: If the config file is not valid TOML.
        pydantic.ValidationError: If the resolved payload violates the xpool
            configuration schema.

    Side Effects:
        Replaces the process-global config held by this module.
    """

    if config is not None and (config_path is not None or env is not None or cli_overrides is not None):
        raise ConfigError("init_global_config(config=...) cannot be combined with config_path, env, or cli_overrides")
    if config is not None:
        resolved = config
    else:
        effective_env = os.environ if env is None else env
        effective_cli: dict[str, object] = dict(cli_overrides or {})
        if config_path is not None:
            effective_cli["config_path"] = str(config_path)

        config_path_setting = next(setting for setting in CONFIG_REGISTRY if setting.name == "config_path")
        effective_path_value, effective_path_source = _resolve_setting(
            config_path_setting, {}, effective_cli, effective_env
        )
        if effective_path_source is None:
            raise MissingRequiredConfig(
                "xpool config path is required: set the XPOOL_CONFIG environment variable "
                "(or pass --config). Serving instance identity is derived from the "
                "one-model-one-serving-instance mapping in this config."
            )
        resolved = XpoolConfig.from_file(
            cast(str, effective_path_value),
            cli_overrides=effective_cli,
            env=effective_env,
        )

    global _global_config
    with _global_config_lock:
        _global_config = resolved
    return resolved


def get_global_config() -> XpoolConfig:
    """Return the process-global xpool config.

    Returns:
        Config previously installed by :func:`init_global_config`.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
    """

    config = _global_config
    if config is None:
        raise MissingRequiredConfig("xpool global config has not been loaded")
    return config


def _resolve_setting(
    setting: ConfigSetting,
    payload: Mapping[str, object],
    cli_overrides: Mapping[str, object],
    env: Mapping[str, str],
) -> tuple[object, ConfigSource | None]:
    if ConfigSource.CLI in setting.allowed_sources and setting.name in cli_overrides:
        return _parse_setting(setting, cli_overrides[setting.name]), ConfigSource.CLI
    if ConfigSource.ENV in setting.allowed_sources and setting.env_var is not None and setting.env_var in env:
        return _parse_setting(setting, env[setting.env_var]), ConfigSource.ENV
    if ConfigSource.CONFIG in setting.allowed_sources:
        found, config_value = _get_nested(payload, setting.path or ())
        if found:
            return _parse_setting(setting, config_value), ConfigSource.CONFIG
    if ConfigSource.DEFAULT in setting.allowed_sources:
        return _parse_setting(setting, setting.default), ConfigSource.DEFAULT
    if setting.required:
        raise MissingRequiredConfig(f"missing required config setting: {setting.name}")
    return None, None


def _parse_setting(setting: ConfigSetting, value: object) -> object:
    match setting.parser:
        case "bool":
            if isinstance(value, bool):
                return value
            normalized = str(value).strip()
            if normalized == "1":
                return True
            if normalized == "0":
                return False
            raise ConfigError(f"expected boolean flag value '0' or '1', got {value!r}")
        case "int":
            try:
                return int(str(value).strip())
            except ValueError as exc:
                raise ConfigError(f"expected integer config value for {setting.name}, got {value!r}") from exc
        case "raw":
            return value
        case "str":
            return str(value)
        case _:
            raise ConfigError(f"unknown parser for {setting.name}: {setting.parser}")


def _validate_required_config_paths(payload: Mapping[str, object]) -> None:
    for setting in CONFIG_REGISTRY:
        if setting.path is None or not setting.required:
            continue
        path = setting.path
        if "*" in path:
            _validate_required_wildcard_path(payload, setting)
            continue
        found, value = _get_nested(payload, path)
        if not found or value is None or value == "" or value == []:
            raise MissingRequiredConfig(f"missing required config setting: {setting.name}")


def _validate_required_wildcard_path(payload: Mapping[str, object], setting: ConfigSetting) -> None:
    path = setting.path or ()
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
        if not found or value is None or value == "" or value == []:
            raise MissingRequiredConfig(
                f"missing required config setting: {'.'.join(collection_path)}[{index}].{'.'.join(item_path)}"
            )


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


def _require_unique(values: Sequence[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
