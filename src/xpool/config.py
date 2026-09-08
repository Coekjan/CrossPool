"""Configuration schema, source registry, and static placement for CrossPool."""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

import xpool.native

__all__ = [
    "CONFIG_REGISTRY",
    "AtnAgentConfig",
    "AtnConfig",
    "ConfigError",
    "ConfigSetting",
    "ConfigSource",
    "ConfigSourceRecord",
    "DebugConfig",
    "FabricObserverDebugConfig",
    "FfnAgentConfig",
    "FfnConfig",
    "FfnLoaderConfig",
    "FfnPlacementConfig",
    "FfnRoutingObserverDebugConfig",
    "FfnSchedulingPolicy",
    "GraphObserverDebugConfig",
    "InstanceConfig",
    "LoggingConfig",
    "MissingRequiredConfig",
    "ModelConfig",
    "PrefillLogitObserverDebugConfig",
    "SchedulerConfig",
    "TopologyError",
    "VendorConfig",
    "XpoolConfig",
    "XpoolDaemonConfig",
    "get_global_config",
    "init_global_config",
]

logger = logging.getLogger(__name__)


class ConfigSource(StrEnum):
    """Configuration value source used by the CrossPool setting registry.

    Attributes:
        CLI: Value came from an explicit CrossPool CLI override.
        ENV: Value came from an allowlisted process environment variable.
        CONFIG: Value came from the TOML config file.
        DEFAULT: Value came from a registry default.
        UNSET: Optional value has no direct CLI, environment, TOML, or default source.
    """

    CLI = "cli"
    ENV = "env"
    CONFIG = "config"
    DEFAULT = "default"
    UNSET = "unset"


type ParserName = Literal["bool", "int", "raw", "str"]


class ConfigSourceRecord(TypedDict):
    """Resolved config value and source provenance."""

    name: str
    value: object
    source: ConfigSource


class ConfigError(ValueError):
    """Base error for CrossPool config resolution failures."""


class MissingRequiredConfig(ConfigError):
    """Raised when a required setting has no value from any allowed source."""


class TopologyError(ConfigError):
    """Raised when model or device topology cannot be derived safely."""


class FfnSchedulingPolicy(StrEnum):
    """Device-side policy used to admit ready FFN steps to executors.

    Attributes:
        FIFO: Admit the ready invocation with the smallest monotonic ticket.
        RANDOM: Select a ready invocation with the generation's deterministic
            random seed.
    """

    FIFO = "fifo"
    RANDOM = "random"


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

    def parse(self, value: object) -> object:
        """Parse one raw value using this setting's declared parser.

        Args:
            value: Raw selected value.

        Returns:
            Parsed config value.

        Raises:
            ConfigError: If the parser rejects the value or is unknown.
        """

        match self.parser:
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
                    raise ConfigError(f"expected integer config value for {self.name}, got {value!r}") from exc
            case "raw":
                return value
            case "str":
                return str(value)
            case _:
                raise ConfigError(f"unknown parser for {self.name}: {self.parser}")

    def resolve(
        self,
        payload: Mapping[str, object],
        cli_overrides: Mapping[str, object],
        env: Mapping[str, str],
    ) -> tuple[object, ConfigSource | None]:
        """Resolve this setting according to CrossPool source precedence.

        Args:
            payload: Config-file payload.
            cli_overrides: Explicit CLI values keyed by setting name.
            env: Allowlisted environment values.

        Returns:
            Resolved value and source, or ``(None, None)`` when optional and unset.

        Raises:
            MissingRequiredConfig: If this required setting has no value.
            ConfigError: If the selected value cannot be parsed.
        """

        if ConfigSource.CLI in self.allowed_sources and self.name in cli_overrides:
            return self.parse(cli_overrides[self.name]), ConfigSource.CLI
        if ConfigSource.ENV in self.allowed_sources and self.env_var is not None and self.env_var in env:
            return self.parse(env[self.env_var]), ConfigSource.ENV
        if ConfigSource.CONFIG in self.allowed_sources:
            found, config_value = get_nested(payload, self.path or ())
            if found:
                return self.parse(config_value), ConfigSource.CONFIG
        if ConfigSource.DEFAULT in self.allowed_sources:
            return self.parse(self.default), ConfigSource.DEFAULT
        if self.required:
            raise MissingRequiredConfig(f"missing required config setting: {self.name}")
        return None, None

    def source_records(self, payload: Mapping[str, object]) -> list[ConfigSourceRecord]:
        """Expand this setting's wildcard path into concrete source records.

        Args:
            payload: Original config mapping before overrides are applied.

        Returns:
            Source records for every concrete wildcard path.

        Raises:
            ConfigError: If the payload shape does not match the wildcard path.
        """

        path = self.path or ()
        records: list[ConfigSourceRecord] = []

        def walk(value: object, remaining_path: tuple[str, ...], concrete_path: tuple[str | int, ...]) -> None:
            if not remaining_path:
                records.append(
                    {
                        "name": format_source_record_name(concrete_path),
                        "value": value,
                        "source": ConfigSource.CONFIG,
                    }
                )
                return

            segment = remaining_path[0]
            rest = remaining_path[1:]
            if segment == "*":
                if not isinstance(value, list):
                    raise ConfigError(
                        f"expected list config value at {format_source_record_name(concrete_path)} "
                        f"for wildcard setting {self.name}"
                    )
                for index, item in enumerate(value):
                    walk(item, rest, (*concrete_path, index))
                return

            if not isinstance(value, Mapping):
                raise ConfigError(f"expected mapping config value at {format_source_record_name(concrete_path)}")
            mapping = cast(Mapping[str, object], value)
            if segment not in mapping:
                if "*" in rest:
                    return
                records.append(
                    {
                        "name": format_source_record_name((*concrete_path, segment, *rest)),
                        "value": None,
                        "source": ConfigSource.UNSET,
                    }
                )
                return
            walk(mapping[segment], rest, (*concrete_path, segment))

        walk(payload, path, ())
        return records


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
        name="logging_level",
        path=("logging", "level"),
        parser="str",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default="info",
        cli="--log-level",
        env_var="XPOOL_LOG_LEVEL",
        description="Minimum level emitted by CrossPool runtime loggers.",
    ),
    ConfigSetting(
        name="logging_color",
        path=("logging", "color"),
        parser="bool",
        allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=True,
        description="Whether CrossPool runtime log levels use ANSI color on TTY stderr.",
    ),
    ConfigSetting(
        name="debug_graph_observer_enable",
        path=("debug", "graph_observer", "enable"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE",
        description="Development-only switch for SGLang graph events and native FFN Graph snapshots.",
    ),
    ConfigSetting(
        name="debug_graph_observer_outdir",
        path=("debug", "graph_observer", "outdir"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR",
        description="Directory for SGLang graph event files and native FFN Graph snapshots.",
    ),
    ConfigSetting(
        name="debug_prefill_logit_observer_enable",
        path=("debug", "prefill_logit_observer", "enable"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE",
        description="Development-only switch that records first-prefill logits from SGLang.",
    ),
    ConfigSetting(
        name="debug_prefill_logit_observer_outdir",
        path=("debug", "prefill_logit_observer", "outdir"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_OUTDIR",
        description="Directory used by the prefill-logit observer for Safetensors output.",
    ),
    ConfigSetting(
        name="debug_transport_observer_enable",
        path=("debug", "transport_observer", "enable"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE",
        description="Development-only switch that records native transport device-phase timings.",
    ),
    ConfigSetting(
        name="debug_transport_observer_outdir",
        path=("debug", "transport_observer", "outdir"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR",
        description="Directory used by the native transport observer for JSON output.",
    ),
    ConfigSetting(
        name="debug_transport_observer_record_capacity",
        path=("debug", "transport_observer", "record_capacity"),
        parser="int",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=8192,
        env_var="XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY",
        description="Positive number of native transport trace records retained per arena.",
    ),
    ConfigSetting(
        name="debug_fabric_observer_enable",
        path=("debug", "fabric_observer", "enable"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE",
        description="Development-only switch that records cross-Agent Fabric phases.",
    ),
    ConfigSetting(
        name="debug_fabric_observer_outdir",
        path=("debug", "fabric_observer", "outdir"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR",
        description="Directory used by the fabric observer for structured snapshots.",
    ),
    ConfigSetting(
        name="debug_fabric_observer_record_capacity",
        path=("debug", "fabric_observer", "record_capacity"),
        parser="int",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=8192,
        env_var="XPOOL_DEBUG_FABRIC_OBSERVER_RECORD_CAPACITY",
        description="Positive number of native fabric trace records retained per PE.",
    ),
    ConfigSetting(
        name="debug_ffn_routing_observer_enable",
        path=("debug", "ffn_routing_observer", "enable"),
        parser="bool",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=False,
        env_var="XPOOL_DEBUG_FFN_ROUTING_OBSERVER_ENABLE",
        description="Development-only switch that records real-FFN routing tensors.",
    ),
    ConfigSetting(
        name="debug_ffn_routing_observer_outdir",
        path=("debug", "ffn_routing_observer", "outdir"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_DEBUG_FFN_ROUTING_OBSERVER_OUTDIR",
        description="Directory used by the FFN routing observer for Safetensors output.",
    ),
    ConfigSetting(
        name="debug_ffn_routing_observer_record_capacity",
        path=("debug", "ffn_routing_observer", "record_capacity"),
        parser="int",
        allowed_sources=(ConfigSource.ENV, ConfigSource.DEFAULT),
        default=8,
        env_var="XPOOL_DEBUG_FFN_ROUTING_OBSERVER_RECORD_CAPACITY",
        description="Positive number of routing records retained per FfnAgent.",
    ),
    ConfigSetting(
        name="vendor_model_base_uri",
        path=("vendor", "model_base_uri"),
        parser="str",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Absolute local model-cache root used to resolve model ids such as org/name into weight paths.",
    ),
    ConfigSetting(
        name="atn_devices",
        path=("atn", "devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host attention execution and attention-side CrossPool agents.",
    ),
    ConfigSetting(
        name="ffn_devices",
        path=("ffn", "devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="CUDA devices that host FFN-side CrossPool agents.",
    ),
    ConfigSetting(
        name="ffn_device_memory_calibration",
        path=("ffn", "device_memory_calibration"),
        parser="raw",
        allowed_sources=(ConfigSource.ENV, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=None,
        env_var="XPOOL_FFN_DEVICE_MEMORY_CALIBRATION",
        description="Absolute path to one environment-qualified CrossPool memory calibration Profile.",
    ),
    ConfigSetting(
        name="ffn_device_memory_extra_margin_bytes",
        path=("ffn", "device_memory_extra_margin_bytes"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=0,
        description="Explicit extra device-memory safety margin added after estimation.",
    ),
    ConfigSetting(
        name="ffn_loader_parallelism",
        path=("ffn", "loader", "parallelism"),
        parser="int",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=4,
        cli="--ffn-loader-parallelism",
        env_var="XPOOL_FFN_LOADER_PARALLELISM",
        description="Number of bounded Host readers used by one FfnAgent checkpoint materialization.",
    ),
    ConfigSetting(
        name="ffn_placement_parallelism",
        path=("ffn", "placement", "parallelism"),
        parser="int",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=4,
        cli="--ffn-placement-parallelism",
        env_var="XPOOL_FFN_PLACEMENT_PARALLELISM",
        description="Worker count for FFN Placement objective passes.",
    ),
    ConfigSetting(
        name="ffn_placement_timeout_seconds",
        path=("ffn", "placement", "timeout_seconds"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=60,
        description="Shared non-renewable FFN Placement solve deadline in seconds.",
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
        name="scheduler_atn_concurrency",
        path=("scheduler", "atn_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--atn-concurrency",
        description="Reserved attention-side concurrency setting for future KV-pool admission.",
    ),
    ConfigSetting(
        name="scheduler_ffn_concurrency",
        path=("scheduler", "ffn_concurrency"),
        parser="int",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=1,
        cli="--ffn-concurrency",
        description="Number of FFN Executor Lanes instantiated per Fabric generation.",
    ),
    ConfigSetting(
        name="scheduler_ffn_policy",
        path=("scheduler", "ffn_policy"),
        parser="str",
        allowed_sources=TOP_LEVEL_SOURCES,
        default=FfnSchedulingPolicy.FIFO.value,
        cli="--ffn-policy",
        description="Device-side policy used to admit ready FFN steps to distributed executors.",
    ),
    ConfigSetting(
        name="scheduler_ffn_random_seed",
        path=("scheduler", "ffn_random_seed"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Optional nonzero uint64 seed used only by the random FFN scheduler.",
    ),
    ConfigSetting(
        name="models",
        path=("models",),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Model registry. Each model derives one instance.",
    ),
    ConfigSetting(
        name="model_id",
        path=("models", "*", "id"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Full model id, for example deepseek-ai/DeepSeek-V2-Lite-Chat.",
    ),
    ConfigSetting(
        name="model_path",
        path=("models", "*", "path"),
        parser="str",
        allowed_sources=CONFIG_REQUIRED,
        description="Optional absolute local model path override containing config.json.",
    ),
    ConfigSetting(
        name="model_ffn_tp_size",
        path=("models", "*", "ffn_tp_size"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Optional fixed FFN tensor-parallel width for this model.",
    ),
)


class XpoolDaemonConfig(BaseModel):
    """Daemon control-plane bind settings from config, CLI, or defaults."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(description="Host or interface address used by the daemon HTTP control plane.")
    port: int = Field(ge=1, le=65535, description="TCP port used by the daemon HTTP control plane.")

    @model_validator(mode="after")
    def validate_loopback_host(self) -> XpoolDaemonConfig:
        """Require the unauthenticated daemon control plane to stay host-local."""

        if self.host == "localhost":
            return self
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("daemon.host must be localhost or a loopback IP address") from exc
        if not address.is_loopback:
            raise ValueError("daemon.host must be localhost or a loopback IP address")
        return self


class LoggingConfig(BaseModel):
    """Process-local CrossPool runtime logging policy."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Minimum level emitted by CrossPool runtime loggers.",
    )
    color: bool = Field(
        default=True,
        description="Whether CrossPool runtime log levels use ANSI color on TTY stderr.",
    )


class SchedulerConfig(BaseModel):
    """FFN lane scheduling and reserved attention admission settings."""

    model_config = ConfigDict(extra="forbid")

    atn_concurrency: int = Field(
        ge=1,
        description="Reserved attention-side concurrency setting for future KV-pool admission.",
    )
    ffn_concurrency: int = Field(
        ge=1,
        description="Number of FFN Executor Lanes instantiated per Fabric generation.",
    )
    ffn_policy: FfnSchedulingPolicy = Field(
        description="Device-side policy used to admit ready FFN steps to executors.",
    )
    ffn_random_seed: int | None = Field(
        default=None,
        ge=1,
        le=2**64 - 1,
        description="Explicit generation seed for the random FFN scheduler.",
    )

    @model_validator(mode="after")
    def validate_ffn_scheduler(self) -> SchedulerConfig:
        """Reject policy-specific state on the FIFO scheduler."""

        if self.ffn_policy is FfnSchedulingPolicy.FIFO and self.ffn_random_seed is not None:
            raise ValueError("scheduler.ffn_random_seed is valid only when scheduler.ffn_policy is random")
        return self


class FfnLoaderConfig(BaseModel):
    """Host checkpoint-loading concurrency for one FfnAgent."""

    model_config = ConfigDict(extra="forbid")

    parallelism: int = Field(
        default=4,
        ge=1,
        description="Number of bounded exact-key Safetensors readers.",
    )


class FfnPlacementConfig(BaseModel):
    """FFN Placement resource-admission policy."""

    model_config = ConfigDict(extra="forbid")

    parallelism: int = Field(
        default=4,
        ge=1,
        description="Number of placement-solver workers.",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Whole placement-solver deadline in seconds.",
    )


def validate_device_sequence(devices: list[int]) -> list[int]:
    """Validate one role's rank-ordered CUDA device sequence."""

    if any(device < 0 for device in devices):
        raise ValueError("devices must contain non-negative CUDA device indices")
    if len(devices) != len(set(devices)):
        raise ValueError("devices must be unique")
    if devices != sorted(devices):
        raise ValueError("devices must be sorted in ascending order")
    return devices


class AtnConfig(BaseModel):
    """ATN-owned device assignment."""

    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(
        min_length=1,
        description="CUDA device indices that host attention execution and attention-side CrossPool agents.",
    )

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, devices: list[int]) -> list[int]:
        """Validate the rank-ordered AtnAgent device sequence."""

        return validate_device_sequence(devices)


class FfnConfig(BaseModel):
    """FFN-owned device assignment and startup policy."""

    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(
        min_length=1,
        description="CUDA device indices that host CrossPool FFN execution agents.",
    )
    device_memory_calibration: Path | None = Field(
        default=None,
        description="Absolute path to one CrossPool Memory Calibration Profile.",
    )
    device_memory_extra_margin_bytes: int = Field(
        default=0,
        ge=0,
        description="Operator-requested bytes reserved beyond the estimated device-memory envelope.",
    )
    loader: FfnLoaderConfig = Field(
        default_factory=FfnLoaderConfig,
        description="Checkpoint-loading settings.",
    )
    placement: FfnPlacementConfig = Field(
        default_factory=FfnPlacementConfig,
        description="FFN placement optimizer settings.",
    )

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, devices: list[int]) -> list[int]:
        """Validate the rank-ordered FfnAgent device sequence."""

        return validate_device_sequence(devices)

    @model_validator(mode="after")
    def validate_device_memory_calibration(self) -> FfnConfig:
        """Normalize and validate the optional calibration Profile path."""

        if self.device_memory_calibration is None:
            return self
        path = self.device_memory_calibration.expanduser()
        if not path.is_absolute():
            raise ValueError(f"ffn.device_memory_calibration must be absolute: {self.device_memory_calibration}")
        self.device_memory_calibration = path
        return self


class ModelConfig(BaseModel):
    """User-declared model served by one CrossPool-managed instance."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Full model id, for example deepseek-ai/DeepSeek-V2-Lite-Chat.")
    path: Path | None = Field(
        default=None,
        description="Optional absolute local model path override containing config.json.",
    )
    ffn_tp_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional fixed FFN tensor-parallel width; omission resolves to the FfnAgent Fleet width.",
    )

    @model_validator(mode="after")
    def validate_model_path(self) -> ModelConfig:
        """Normalize and validate the optional configured model path.

        Returns:
            The validated model config with ``~`` expanded.

        Raises:
            ValueError: If the configured path is not absolute.
        """

        model_path = self.path
        if model_path is None:
            return self
        path = model_path.expanduser()
        if not path.is_absolute():
            raise ValueError(f"models[{self.id}].path must be absolute: {model_path}")
        self.path = path
        return self


class AtnAgentConfig(BaseModel):
    """Derived placement for one configured AtnAgent."""

    model_config = ConfigDict(extra="forbid")

    cuda_device: int = Field(ge=0, description="CUDA device index owned by this AtnAgent.")
    rank: int = Field(ge=0, description="Rank in the configured attention-device list.")


class FfnAgentConfig(BaseModel):
    """Derived placement for one configured FfnAgent."""

    model_config = ConfigDict(extra="forbid")

    cuda_device: int = Field(ge=0, description="CUDA device index owned by this FfnAgent.")
    rank: int = Field(ge=0, description="Rank in the configured FFN-device list.")


class InstanceConfig(BaseModel):
    """Derived instance placement for one configured model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Instance id, equal to the configured model id in the current topology.")
    instance_index: int = Field(ge=0, description="Integer instance index fed to the native shim ABI.")


class GraphObserverDebugConfig(BaseModel):
    """Debug-only CUDA Graph observer settings shared by graph producers."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description="Whether Devkit should observe SGLang graph events and native FFN Graph structure.",
    )
    outdir: Path | None = Field(
        default=None,
        description="Directory for per-process SGLang graph events and native FFN Graph snapshots.",
    )

    @model_validator(mode="after")
    def validate_graph_observer(self) -> GraphObserverDebugConfig:
        """Normalize and validate debug graph observer output settings.

        Returns:
            The validated debug config.

        Raises:
            ValueError: If graph observation enablement and output directory
                presence are not configured together.
        """

        if self.enable != (self.outdir is not None):
            raise ValueError(
                "debug.graph_observer.enable and debug.graph_observer.outdir must be set or unset together"
            )
        if self.outdir is None:
            return self
        outdir = self.outdir.expanduser()
        self.outdir = outdir.resolve() if outdir.is_absolute() else (Path.cwd() / outdir).resolve()
        return self


class PrefillLogitObserverDebugConfig(BaseModel):
    """Debug-only first-prefill logit observer settings."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description="Whether the SGLang plugin should record first-prefill next-token logits.",
    )
    outdir: Path | None = Field(
        default=None,
        description="Directory where the observer writes per-process Safetensors files.",
    )

    @model_validator(mode="after")
    def validate_prefill_logit_observer(self) -> PrefillLogitObserverDebugConfig:
        """Normalize and validate prefill-logit observer output settings."""

        if self.enable != (self.outdir is not None):
            raise ValueError(
                "debug.prefill_logit_observer.enable and debug.prefill_logit_observer.outdir "
                "must be set or unset together"
            )
        if self.outdir is not None:
            outdir = self.outdir.expanduser()
            self.outdir = outdir.resolve() if outdir.is_absolute() else (Path.cwd() / outdir).resolve()
        return self


class TransportObserverDebugConfig(BaseModel):
    """Debug-only native transport device-phase observer settings."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=False, description="Whether native transport device-phase timing is enabled.")
    outdir: Path | None = Field(default=None, description="Directory where transport timing snapshots are written.")
    record_capacity: int = Field(
        default=8192,
        gt=0,
        le=2**63 - 1,
        description="Maximum transport trace records retained in each native arena.",
    )

    @model_validator(mode="after")
    def validate_transport_observer(self) -> TransportObserverDebugConfig:
        """Normalize and validate transport observer output settings.

        Returns:
            The validated debug config.

        Raises:
            ValueError: If enablement and output directory presence differ.
        """

        if self.enable != (self.outdir is not None):
            raise ValueError(
                "debug.transport_observer.enable and debug.transport_observer.outdir must be set or unset together"
            )
        if self.outdir is not None:
            outdir = self.outdir.expanduser()
            self.outdir = outdir.resolve() if outdir.is_absolute() else (Path.cwd() / outdir).resolve()
        return self


class FabricObserverDebugConfig(BaseModel):
    """Debug-only cross-Agent Fabric observer settings."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=False, description="Whether cross-Agent Fabric timing is enabled.")
    outdir: Path | None = Field(default=None, description="Directory where fabric snapshots are written.")
    record_capacity: int = Field(
        default=8192,
        gt=0,
        le=2**63 - 1,
        description="Maximum fabric trace records retained by each native PE.",
    )

    @model_validator(mode="after")
    def validate_fabric_observer(self) -> FabricObserverDebugConfig:
        """Normalize and validate fabric observer output settings.

        Returns:
            The validated debug config.

        Raises:
            ValueError: If enablement and output directory presence differ.
        """

        if self.enable != (self.outdir is not None):
            raise ValueError(
                "debug.fabric_observer.enable and debug.fabric_observer.outdir must be set or unset together"
            )
        if self.outdir is not None:
            outdir = self.outdir.expanduser()
            self.outdir = outdir.resolve() if outdir.is_absolute() else (Path.cwd() / outdir).resolve()
        return self


class FfnRoutingObserverDebugConfig(BaseModel):
    """Debug-only real-FFN routing tensor observer settings."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=False, description="Whether to capture real-FFN routing records.")
    outdir: Path | None = Field(default=None, description="Directory that receives routing-observer artifacts.")
    record_capacity: int = Field(
        default=8,
        ge=1,
        le=2**63 - 1,
        description="Maximum routing records retained per process.",
    )

    @model_validator(mode="after")
    def validate_routing_observer(self) -> FfnRoutingObserverDebugConfig:
        """Normalize and validate routing-observer output settings."""

        if self.enable != (self.outdir is not None):
            raise ValueError(
                "debug.ffn_routing_observer.enable and debug.ffn_routing_observer.outdir must be set or unset together"
            )
        if self.outdir is not None:
            outdir = self.outdir.expanduser()
            self.outdir = outdir.resolve() if outdir.is_absolute() else (Path.cwd() / outdir).resolve()
        return self


class DebugConfig(BaseModel):
    """Debug-only runtime switches resolved through the config registry."""

    model_config = ConfigDict(extra="forbid")

    graph_observer: GraphObserverDebugConfig = Field(
        default_factory=GraphObserverDebugConfig,
        description="SGLang and native FFN CUDA Graph observer debug settings.",
    )
    prefill_logit_observer: PrefillLogitObserverDebugConfig = Field(
        default_factory=PrefillLogitObserverDebugConfig,
        description="SGLang first-prefill logit observer debug settings.",
    )
    transport_observer: TransportObserverDebugConfig = Field(
        default_factory=TransportObserverDebugConfig,
        description="Native transport device-phase observer settings.",
    )
    fabric_observer: FabricObserverDebugConfig = Field(
        default_factory=FabricObserverDebugConfig,
        description="Cross-Agent Fabric observer settings.",
    )
    ffn_routing_observer: FfnRoutingObserverDebugConfig = Field(
        default_factory=FfnRoutingObserverDebugConfig,
        description="Real-FFN routing tensor observer settings.",
    )

    def native_options(self) -> xpool.native.debug.Options:
        """Project the exact device-runtime debug options to native values."""

        return xpool.native.debug.Options(
            transport_observer=xpool.native.debug.TraceObserverOptions(
                enable=self.transport_observer.enable,
                record_capacity=self.transport_observer.record_capacity,
            ),
            fabric_observer=xpool.native.debug.TraceObserverOptions(
                enable=self.fabric_observer.enable,
                record_capacity=self.fabric_observer.record_capacity,
            ),
            graph_observer=xpool.native.debug.GraphObserverOptions(enable=self.graph_observer.enable),
            ffn_routing_observer=xpool.native.debug.FfnRoutingObserverOptions(
                enable=self.ffn_routing_observer.enable,
                record_capacity=self.ffn_routing_observer.record_capacity,
            ),
        )


class VendorConfig(BaseModel):
    """Vendor model-root settings used to resolve configured model ids."""

    model_config = ConfigDict(extra="forbid")

    model_base_uri: Path | None = Field(
        default=None,
        description="Absolute local model-cache root prepended to model ids when models[].path is omitted.",
    )

    @model_validator(mode="after")
    def validate_model_base_uri(self) -> VendorConfig:
        """Normalize and validate the optional vendor model-cache root.

        Returns:
            The validated vendor config.

        Raises:
            ValueError: If the configured model-cache root is a relative local path.
        """

        if self.model_base_uri is None:
            return self
        model_base_uri = self.model_base_uri.expanduser()
        if not model_base_uri.is_absolute():
            raise ValueError(f"vendor.model_base_uri must be absolute: {self.model_base_uri}")
        self.model_base_uri = model_base_uri
        return self


class XpoolConfig(BaseModel):
    """Validated CrossPool TOML config plus derived runtime views."""

    model_config = ConfigDict(extra="forbid")
    _sources: tuple[ConfigSourceRecord, ...] = PrivateAttr(default_factory=tuple)

    daemon: XpoolDaemonConfig = Field(description="Daemon control-plane config.")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Runtime logging policy.")
    scheduler: SchedulerConfig = Field(description="Scheduler resource-concurrency config.")
    atn: AtnConfig = Field(description="ATN device assignment.")
    ffn: FfnConfig = Field(description="FFN device assignment and startup policy.")
    debug: DebugConfig = Field(description="Debug-only runtime switches.")
    vendor: VendorConfig = Field(default_factory=VendorConfig, description="Vendor model-root settings.")
    models: list[ModelConfig] = Field(min_length=1, description="Configured model list.")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add registry-declared config override flags to an argparse parser.

        Args:
            parser: Subcommand parser that should accept CrossPool config override
                flags.

        Side Effects:
            Mutates ``parser`` by adding every setting whose registry entry
            allows CLI input.
        """

        for setting in CONFIG_REGISTRY:
            if setting.cli is None or ConfigSource.CLI not in setting.allowed_sources:
                continue
            parser.add_argument(
                setting.cli,
                dest=setting.name,
                type=int if setting.parser == "int" else str,
                help=setting.description,
            )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cli: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> XpoolConfig:
        """Load and validate a CrossPool TOML file.

        Args:
            path: Path to the TOML config file.
            cli: Optional CLI-derived setting overrides that take
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
        return cls.from_mapping(payload, cli=cli, env=env)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        cli: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> XpoolConfig:
        """Validate an in-memory config mapping.

        Args:
            payload: TOML-like mapping to validate. The mapping is deep-copied
                before defaults or overrides are applied.
            cli: Optional CLI-derived setting overrides that take
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

        source_payload = deepcopy(dict(payload))
        resolved: dict[str, object] = deepcopy(dict(payload))
        effective_cli = cli or {}
        effective_env = env or {}
        if env is not None:
            allowed_env_vars = frozenset(setting.env_var for setting in CONFIG_REGISTRY if setting.env_var is not None)
            unknown_env_vars = tuple(
                sorted(name for name in effective_env if name.startswith("XPOOL_") and name not in allowed_env_vars)
            )
            if unknown_env_vars:
                logger.warning("ignoring unknown xpool environment variables: %s", ", ".join(unknown_env_vars))

        for setting in CONFIG_REGISTRY:
            if setting.path is None or ConfigSource.CONFIG in setting.allowed_sources:
                continue
            if get_nested(source_payload, setting.path)[0]:
                raise ConfigError(f"config setting {setting.name} does not allow TOML source: {'.'.join(setting.path)}")

        sources: list[ConfigSourceRecord] = []
        for setting in CONFIG_REGISTRY:
            if setting.path is not None and "*" in setting.path:
                sources.extend(setting.source_records(source_payload))
                continue
            value, source = setting.resolve(source_payload, effective_cli, effective_env)
            if setting.path is not None and setting.name != "models":
                sources.append(
                    {
                        "name": format_source_record_name(setting.path),
                        "value": value,
                        "source": ConfigSource.UNSET if source is None else source,
                    }
                )
            if setting.path is not None and source is not None:
                set_nested(resolved, setting.path, value)

        config = cls.model_validate(resolved)
        config._sources = tuple(sources)
        return config

    @property
    def sources(self) -> tuple[ConfigSourceRecord, ...]:
        """Return immutable source records for resolved leaf config values."""

        return self._sources

    @cached_property
    def cuda_devices(self) -> tuple[int, ...]:
        """Return all CUDA devices managed by CrossPool, ordered by CUDA device index."""

        return tuple(sorted({*self.atn.devices, *self.ffn.devices}))

    @cached_property
    def atnagents(self) -> tuple[AtnAgentConfig, ...]:
        """Return AtnAgent placements in attention-rank order.

        Returns:
            Immutable AtnAgent placement tuple.
        """

        return tuple(
            AtnAgentConfig(cuda_device=cuda_device, rank=rank) for rank, cuda_device in enumerate(self.atn.devices)
        )

    @cached_property
    def atnagent_by_cuda_device(self) -> Mapping[int, AtnAgentConfig]:
        """Return AtnAgent placements keyed by CUDA device index."""

        return MappingProxyType({agent.cuda_device: agent for agent in self.atnagents})

    @cached_property
    def ffnagents(self) -> tuple[FfnAgentConfig, ...]:
        """Return FfnAgent placements in FFN-rank order.

        Returns:
            Immutable FfnAgent placement tuple.
        """

        return tuple(
            FfnAgentConfig(cuda_device=cuda_device, rank=rank) for rank, cuda_device in enumerate(self.ffn.devices)
        )

    @cached_property
    def ffnagent_by_cuda_device(self) -> Mapping[int, FfnAgentConfig]:
        """Return FfnAgent placements keyed by CUDA device index."""

        return MappingProxyType({agent.cuda_device: agent for agent in self.ffnagents})

    @cached_property
    def instances(self) -> tuple[InstanceConfig, ...]:
        """Derive one instance for every configured model.

        Returns:
            Immutable instance placement tuple in model declaration order.
        """

        return tuple(InstanceConfig(id=model.id, instance_index=index) for index, model in enumerate(self.models))

    @cached_property
    def instance_by_id(self) -> Mapping[str, InstanceConfig]:
        """Return derived instance placement keyed by configured instance id."""

        return MappingProxyType({instance.id: instance for instance in self.instances})

    @property
    def atn_world_size(self) -> int:
        """Return the number of attention-side instance ranks per model."""

        return len(self.atn.devices)

    @model_validator(mode="after")
    def validate_device_assignments(self) -> XpoolConfig:
        """Reject CUDA devices assigned to both ATN and FFN roles."""

        overlap = sorted(set(self.atn.devices) & set(self.ffn.devices))
        if overlap:
            raise ValueError(f"CUDA devices may host only one xpool role; overlapping devices: {overlap}")
        return self

    def model_path_of(self, model_id: str) -> Path:
        """Return the resolved absolute local path for a configured model.

        Args:
            model_id: Full configured model id, for example
                ``deepseek-ai/DeepSeek-V2-Lite-Chat``.

        Returns:
            Explicit ``models[].path`` when present; otherwise
            ``vendor.model_base_uri / model_id``.

        Raises:
            MissingRequiredConfig: If ``model_id`` is not configured, or if the
                model has no explicit path and no vendor model base URI.
        """

        for model in self.models:
            if model.id != model_id:
                continue
            model_path = model.path
            if model_path is None:
                if self.vendor.model_base_uri is None:
                    raise MissingRequiredConfig(
                        f"missing required model path for {model.id}: set models[].path or vendor.model_base_uri"
                    )
                model_path = self.vendor.model_base_uri / model.id
            return model_path
        raise MissingRequiredConfig(f"unknown configured model id: {model_id}")

    @model_validator(mode="after")
    def validate_references(self) -> XpoolConfig:
        """Reject duplicate model identities and paths.

        Returns:
            The validated config object.

        Raises:
            ValueError: If model ids or resolved model paths are duplicated.
        """

        model_path_by_id: dict[str, Path] = {}
        for model in self.models:
            if model.id in model_path_by_id:
                raise ValueError("model ids must be unique")
            model_path = model.path
            if model_path is None:
                if self.vendor.model_base_uri is None:
                    raise MissingRequiredConfig(
                        f"missing required model path for {model.id}: set models[].path or vendor.model_base_uri"
                    )
                model_path = self.vendor.model_base_uri / model.id
            model_path_by_id[model.id] = model_path
        model_paths = [model_path.resolve() for model_path in model_path_by_id.values()]
        if len(model_paths) != len(set(model_paths)):
            raise ValueError("model paths must be unique")

        return self


global_config: XpoolConfig | None = None
global_config_lock = Lock()


def init_global_config(
    *,
    config_path: str | Path | None = None,
    cli: Mapping[str, object] | None = None,
) -> XpoolConfig:
    """Initialize the process-global CrossPool config.

    Args:
        config_path: Explicit TOML config path. When provided, it takes
            precedence over ``XPOOL_CONFIG`` in the process environment.
        cli: Optional CLI-derived setting overrides.

    Returns:
        The config object now returned by :func:`get_global_config`.

    Raises:
        ConfigError: If registry resolution fails or a different effective
            config was already installed in this process.
        MissingRequiredConfig: If no config path is available.
        OSError: If the config file cannot be opened.
        tomllib.TOMLDecodeError: If the config file is not valid TOML.
        pydantic.ValidationError: If the resolved payload violates the CrossPool
            configuration schema.

    Side Effects:
        Installs the process-global config once. Repeated initialization with
        equal effective values returns the first installed object unchanged.
    """

    effective_cli: dict[str, object] = dict(cli or {})
    if config_path is not None:
        effective_cli["config_path"] = str(config_path)

    config_path_setting = next(setting for setting in CONFIG_REGISTRY if setting.name == "config_path")
    effective_path_value, effective_path_source = config_path_setting.resolve({}, effective_cli, os.environ)
    if effective_path_source is None:
        raise MissingRequiredConfig(
            "xpool config path is required: set the XPOOL_CONFIG environment variable "
            "(or pass --config). Instance identity is derived from the "
            "one-model-one-instance mapping in this config."
        )
    resolved = XpoolConfig.from_file(cast(str, effective_path_value), cli=effective_cli, env=os.environ)

    global global_config
    with global_config_lock:
        if global_config is not None:
            if global_config.model_dump(mode="json") != resolved.model_dump(mode="json"):
                raise ConfigError("xpool global config is already initialized with different values")
            return global_config
        global_config = resolved
    return resolved


def get_global_config() -> XpoolConfig:
    """Return the process-global CrossPool config.

    Returns:
        Config previously installed by :func:`init_global_config`.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
    """

    config = global_config
    if config is None:
        raise MissingRequiredConfig("xpool global config has not been loaded")
    return config


def format_source_record_name(path: tuple[str | int, ...]) -> str:
    """Format a nested config path for source-report diagnostics.

    Args:
        path: Config path segments, including integer list indexes.

    Returns:
        Dot-and-bracket notation for the path.
    """

    return "".join(
        f"[{segment}]" if isinstance(segment, int) else f"{'.' if index else ''}{segment}"
        for index, segment in enumerate(path)
    )


def get_nested(payload: Mapping[str, object], path: tuple[str, ...]) -> tuple[bool, object]:
    """Read a nested mapping path without conflating missing and null values.

    Args:
        payload: Mapping to traverse.
        path: String path segments.

    Returns:
        Whether the path exists and its value when present.
    """

    cursor: object = payload
    for key in path:
        if not isinstance(cursor, Mapping):
            return False, None
        mapping = cast(Mapping[str, object], cursor)
        if key not in mapping:
            return False, None
        cursor = mapping[key]
    return True, cursor


def set_nested(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    """Set a nested config path, creating missing mappings.

    Args:
        payload: Mutable config mapping.
        path: Non-empty path to update.
        value: Resolved value to install.

    Raises:
        ConfigError: If the path is empty or crosses a non-mapping value.

    Side Effects:
        Mutates ``payload`` in place.
    """

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
