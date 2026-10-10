"""Configuration schema, source registry, and static placement for CrossPool."""

from __future__ import annotations

import ipaddress
import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

import xpool.native
from xpool.model import ModelId
from xpool.utils.config import (
    ConfigError,
    ConfigModel,
    ConfigSetting,
    ConfigSource,
    ConfigSourceRecord,
    MissingRequiredConfig,
)

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
    "LatencySloConfig",
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
    "validate_device_layout",
]


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


TOP_LEVEL_SOURCES = (
    ConfigSource.CLI,
    ConfigSource.CONFIG,
    ConfigSource.DEFAULT,
)
CONFIG_REQUIRED = (ConfigSource.CONFIG,)

CONFIG_REGISTRY: tuple[ConfigSetting, ...] = (
    ConfigSetting(
        name="cache_root",
        path=("cache_root",),
        parser="path",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=".xpool-cache",
        cli="--cache-root",
        env_var="XPOOL_CACHE_ROOT",
        description="Shared project artifact and profiling workspace root.",
    ),
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
        description="devices that host attention execution and attention-side CrossPool agents.",
    ),
    ConfigSetting(
        name="atn_device_memory_utilization",
        path=("atn", "device_memory_utilization"),
        parser="raw",
        allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=0.95,
        description="Fraction of post-capture device memory available to the elastic KV capacity pool.",
    ),
    ConfigSetting(
        name="ffn_devices",
        path=("ffn", "devices"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="devices that host FFN-side CrossPool agents.",
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
        description="Reserved attention-side concurrency setting for future attention compute admission.",
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
        name="scheduler_slo",
        path=("scheduler", "slo"),
        parser="raw",
        allowed_sources=CONFIG_REQUIRED,
        required=True,
        description="Scheduler-local TTFT and TBT objectives used for elastic KV capacity arbitration.",
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
        name="model_atn_tp_size",
        path=("models", "*", "atn_tp_size"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Optional attention tensor-parallel width; omission divides the attention world by DP.",
    ),
    ConfigSetting(
        name="model_atn_dp_size",
        path=("models", "*", "atn_dp_size"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=1,
        description="Attention data-parallel width; values above one require DP attention.",
    ),
    ConfigSetting(
        name="model_ffn_tp_size",
        path=("models", "*", "ffn_tp_size"),
        parser="int",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Optional fixed FFN tensor-parallel width for this model.",
    ),
    ConfigSetting(
        name="model_slo",
        path=("models", "*", "slo"),
        parser="raw",
        allowed_sources=(ConfigSource.CONFIG,),
        description="Optional complete model override of scheduler-local TTFT and TBT objectives.",
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


class LatencySloConfig(BaseModel):
    """Scheduler-local latency objectives for elastic KV arbitration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ttft_ms: FiniteFloat = Field(gt=0, description="Scheduler-local time-to-first-token objective in milliseconds.")
    tbt_ms: FiniteFloat = Field(gt=0, description="Time-between-tokens objective in milliseconds.")


class SchedulerConfig(BaseModel):
    """FFN lane scheduling and reserved attention admission settings."""

    model_config = ConfigDict(extra="forbid")

    atn_concurrency: int = Field(
        ge=1,
        description="Reserved attention-side concurrency setting for future attention compute admission.",
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
    slo: LatencySloConfig = Field(description="Default latency objectives for elastic KV capacity arbitration.")

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


class AtnConfig(BaseModel):
    """ATN-owned device assignment and memory policy."""

    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(
        min_length=1,
        description="device indices that host attention execution and attention-side CrossPool agents.",
    )
    device_memory_utilization: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Fraction of post-capture device memory available to the elastic KV capacity pool.",
    )

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, devices: list[int]) -> list[int]:
        """Validate the rank-ordered AtnAgent device sequence."""

        if devices != list(range(len(devices))):
            raise ValueError("attention devices must be consecutive indices starting at zero")
        return devices


class FfnConfig(BaseModel):
    """FFN-owned device assignment and startup policy."""

    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(
        min_length=1,
        description="device indices that host CrossPool FFN execution agents.",
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

        if devices[0] < 0 or devices != list(range(devices[0], devices[0] + len(devices))):
            raise ValueError("FFN devices must be nonnegative consecutive indices")
        return devices

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


def validate_device_layout(atn: AtnConfig, ffn: FfnConfig) -> None:
    """Require validated role blocks to form one attention-first device layout.

    Raises ValueError when the FFN block does not immediately follow attention.
    Role configuration types own their individual sequence constraints.
    """

    if ffn.devices[0] != len(atn.devices):
        raise ValueError("FFN devices must immediately follow the attention device block")


class ModelConfig(BaseModel):
    """User-declared model served by one CrossPool-managed instance."""

    model_config = ConfigDict(extra="forbid")

    id: ModelId = Field(description="Canonical model identity, for example deepseek-ai/DeepSeek-V2-Lite-Chat.")
    path: Path | None = Field(
        default=None,
        description="Optional absolute local model path override containing config.json.",
    )
    atn_tp_size: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Optional attention TP width; omission divides the AtnAgent Fleet width by attention DP.",
    )
    atn_dp_size: int = Field(
        default=1,
        ge=1,
        strict=True,
        description="Attention DP width; values above one enable DP attention.",
    )
    ffn_tp_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional fixed FFN tensor-parallel width; omission resolves to the FfnAgent Fleet width.",
    )
    slo: LatencySloConfig | None = Field(
        default=None,
        description="Optional complete override of scheduler latency objectives for this model.",
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

    device: int = Field(ge=0, description="device index owned by this AtnAgent.")
    rank: int = Field(ge=0, description="Rank in the configured attention-device list.")


class FfnAgentConfig(BaseModel):
    """Derived placement for one configured FfnAgent."""

    model_config = ConfigDict(extra="forbid")

    device: int = Field(ge=0, description="device index owned by this FfnAgent.")
    rank: int = Field(ge=0, description="Rank in the configured FFN-device list.")


class InstanceConfig(BaseModel):
    """Derived instance placement for one configured model."""

    model_config = ConfigDict(extra="forbid")

    model_id: ModelId = Field(description="Model ID identifying this configured Instance.")
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


class XpoolConfig(ConfigModel):
    """Validated CrossPool TOML config plus derived runtime views."""

    registry: ClassVar[tuple[ConfigSetting, ...]] = CONFIG_REGISTRY
    env_prefix: ClassVar[str] = "XPOOL_"

    cache_root: Path = Field(description="Shared project artifact and profiling workspace root.")

    daemon: XpoolDaemonConfig = Field(description="Daemon control-plane config.")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Runtime logging policy.")
    scheduler: SchedulerConfig = Field(description="Scheduler resource-concurrency config.")
    atn: AtnConfig = Field(description="ATN device assignment.")
    ffn: FfnConfig = Field(description="FFN device assignment and startup policy.")
    debug: DebugConfig = Field(description="Debug-only runtime switches.")
    vendor: VendorConfig = Field(default_factory=VendorConfig, description="Vendor model-root settings.")
    models: list[ModelConfig] = Field(min_length=1, description="Configured model list.")

    @classmethod
    def resolve_cache_root(
        cls,
        *,
        config_path: str | Path | None = None,
        cli: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> tuple[Path, ConfigSourceRecord]:
        """Resolve cache storage without validating or installing a deployment.

        Explicit runtime files precede XPOOL_CONFIG. Selected files must be
        readable TOML even with a cache override; unrelated fields are ignored.
        TOML cache paths use the file directory, other sources use cwd.
        Environment defaults to this process; pass an empty mapping to omit it.
        """

        environment = os.environ if env is None else env
        path_setting = next(setting for setting in cls.registry if setting.name == "config_path")
        path_inputs = {} if config_path is None else {"config_path": str(config_path)}
        path_value, path_source = path_setting.resolve({}, path_inputs, environment)
        payload: dict[str, object] = {}
        origin = None
        if path_source is not None:
            path = Path(cast(str, path_value)).expanduser().resolve()
            with path.open("rb") as config_file:
                payload = tomllib.load(config_file)
            origin = path.parent
        cache_setting = next(setting for setting in cls.registry if setting.name == "cache_root")
        value, source = cache_setting.resolve(payload, cli or {}, environment, origin=origin)
        return cast(Path, value), {
            "name": "cache_root",
            "value": value,
            "source": ConfigSource.UNSET if source is None else source,
        }

    @cached_property
    def devices(self) -> tuple[int, ...]:
        """Return all devices managed by CrossPool, ordered by device index."""

        return tuple(self.atn.devices + self.ffn.devices)

    @cached_property
    def atnagents(self) -> tuple[AtnAgentConfig, ...]:
        """Return AtnAgent placements in attention-rank order.

        Returns:
            Immutable AtnAgent placement tuple.
        """

        return tuple(AtnAgentConfig(device=device, rank=rank) for rank, device in enumerate(self.atn.devices))

    @cached_property
    def atnagent_by_device(self) -> Mapping[int, AtnAgentConfig]:
        """Return AtnAgent placements keyed by device index."""

        return MappingProxyType({agent.device: agent for agent in self.atnagents})

    @cached_property
    def ffnagents(self) -> tuple[FfnAgentConfig, ...]:
        """Return FfnAgent placements in FFN-rank order.

        Returns:
            Immutable FfnAgent placement tuple.
        """

        return tuple(FfnAgentConfig(device=device, rank=rank) for rank, device in enumerate(self.ffn.devices))

    @cached_property
    def ffnagent_by_device(self) -> Mapping[int, FfnAgentConfig]:
        """Return FfnAgent placements keyed by device index."""

        return MappingProxyType({agent.device: agent for agent in self.ffnagents})

    @cached_property
    def instances(self) -> tuple[InstanceConfig, ...]:
        """Derive one instance for every configured model.

        Returns:
            Immutable instance placement tuple in model declaration order.
        """

        return tuple(InstanceConfig(model_id=model.id, instance_index=index) for index, model in enumerate(self.models))

    @cached_property
    def instance_by_model_id(self) -> Mapping[ModelId, InstanceConfig]:
        """Return derived Instance placements keyed by Model ID."""

        return MappingProxyType({instance.model_id: instance for instance in self.instances})

    @property
    def model_by_id(self) -> Mapping[ModelId, ModelConfig]:
        """Return model declarations keyed by canonical identity."""

        return MappingProxyType({model.id: model for model in self.models})

    @property
    def atn_world_size(self) -> int:
        """Return the physical AtnAgent Fleet width."""

        return len(self.atn.devices)

    def atn_tp_size_of(self, model_id: ModelId) -> int:
        """Resolve declared or omitted attention TP against the complete World.

        Model declarations retain omitted TP as None. The complete configuration
        validates divisibility and the TP-by-DP product before runtime consumers
        use this concrete width.
        """

        model = self.model_by_id[model_id]
        return model.atn_tp_size if model.atn_tp_size is not None else self.atn_world_size // model.atn_dp_size

    @model_validator(mode="after")
    def validate_device_assignments(self) -> XpoolConfig:
        """Validate role separation and each model's complete attention World."""

        validate_device_layout(self.atn, self.ffn)
        for model in self.models:
            if self.atn_world_size % model.atn_dp_size != 0:
                raise ValueError(f"{model.id}: attention DP must divide the configured attention World")
            if self.atn_tp_size_of(model.id) * model.atn_dp_size != self.atn_world_size:
                raise ValueError(f"{model.id}: attention TP times DP must equal the configured attention World")
        return self

    def model_path_of(self, model_id: ModelId) -> Path:
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
                model_path = self.vendor.model_base_uri / model.id.relative_path
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

        model_path_by_id: dict[ModelId, Path] = {}
        for model in self.models:
            if model.id in model_path_by_id:
                raise ValueError("model ids must be unique")
            model_path = model.path
            if model_path is None:
                if self.vendor.model_base_uri is None:
                    raise MissingRequiredConfig(
                        f"missing required model path for {model.id}: set models[].path or vendor.model_base_uri"
                    )
                model_path = self.vendor.model_base_uri / model.id.relative_path
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
