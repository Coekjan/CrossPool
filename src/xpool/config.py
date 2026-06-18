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
    """Configuration value source used by the xpool setting registry.

    Attributes:
        CLI: Value came from an explicit xpool CLI override.
        CONFIG: Value came from the TOML config file.
        DEFAULT: Value came from a registry default.
    """

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
    """Exclusive role assigned to one CUDA device agent.

    Attributes:
        ATTENTION: Device hosts SGLang attention and the attention-side shim agent.
        FFN: Device hosts xpool FFN execution.
    """

    ATTENTION = "attention"
    FFN = "ffn"


class AttentionKind(StrEnum):
    """Attention topology kind derived from SGLang model metadata.

    Attributes:
        MLA: Multi-head latent attention, represented as one physical KV lane.
        GQA: Grouped-query attention with fewer KV heads than query heads.
        MHA: Multi-head attention with equal query and KV head counts.
        MQA: Multi-query attention with one KV head.
    """

    MLA = "mla"
    GQA = "gqa"
    MHA = "mha"
    MQA = "mqa"


@dataclass(frozen=True, slots=True)
class SglangModelMetadata:
    """Model shape facts derived through SGLang's config resolution path.

    Attributes:
        family: Hugging Face or SGLang model type used for diagnostics.
        hidden_size: Hidden-state width consumed by each FFN shim.
        num_attention_heads: Total query-head count before attention-side TP.
        num_key_value_heads: Total KV-head count reported by SGLang.
        attention_kind: Derived attention topology family.
        physical_kv_lanes: Physical KV lanes used for xpool attention policy.
    """

    family: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_kind: AttentionKind
    physical_kv_lanes: int


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """Registry entry for one TOML, CLI, or defaulted setting.

    Attributes:
        name: Stable registry key used by CLI overrides and diagnostics.
        config_path: TOML path for config-backed settings, or ``None`` for virtual settings.
        parser: Parser name used to normalize raw source values.
        allowed_sources: Sources allowed to provide this setting.
        description: Human-readable setting purpose for generated registry output.
        default: Default value used when ``DEFAULT`` is an allowed source.
        required: Whether missing values are configuration errors.
        cli: CLI flag name when the setting is overrideable from the command line.
    """

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
        """Return whether this setting may use its registry default.

        Returns:
            ``True`` when ``ConfigSource.DEFAULT`` is allowed for this setting.
        """

        return ConfigSource.DEFAULT in self.allowed_sources

    @property
    def is_config_field(self) -> bool:
        """Return whether this setting maps to a concrete TOML field.

        Returns:
            ``True`` when the setting has a non-wildcard config path that can be
            populated into the resolved TOML payload.
        """

        return self.config_path is not None and "*" not in self.config_path

    @property
    def is_required_config_path(self) -> bool:
        """Return whether the setting's TOML path must be present.

        Returns:
            ``True`` when this setting is required and has a config path.
        """

        return self.config_path is not None and self.required


def parse_int(value: object) -> int:
    """Parse a registry value as an integer.

    Args:
        value: Raw value from CLI, TOML, or the registry default.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value cannot be parsed by ``int``.
    """

    return int(str(value).strip())


def parse_str(value: object) -> str:
    """Parse a registry value as a string.

    Args:
        value: Raw value from CLI, TOML, or the registry default.

    Returns:
        String representation of the value.
    """

    return str(value)


def parse_raw(value: object) -> object:
    """Return a registry value without coercion.

    Args:
        value: Raw value from CLI, TOML, or the registry default.

    Returns:
        The original value object.
    """

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
        description="CUDA device indices that host SGLang attention and attention-side xpool agents.",
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

        _validate_cuda_device_list(self.attention_cuda_devices, "devices.attention_cuda_devices")
        _validate_cuda_device_list(self.ffn_cuda_devices, "devices.ffn_cuda_devices")
        overlap = sorted(set(self.attention_cuda_devices) & set(self.ffn_cuda_devices))
        if overlap:
            raise ValueError(f"CUDA devices may host only one xpool role; overlapping devices: {overlap}")
        return self


class ModelConfig(BaseModel):
    """User-declared model served by one xpool-managed SGLang instance."""

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


class ModelInstanceConfig(BaseModel):
    """Derived SGLang instance placement for one configured model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="SGLang instance id, equal to the configured model id in the first topology.")
    model_id: str = Field(description="Configured model id served by this SGLang instance.")
    attention_cuda_devices: list[int] = Field(
        min_length=1,
        description="Attention CUDA devices assigned to this SGLang instance.",
    )
    ffn_agent_ids: list[str] = Field(min_length=1, description="FFN device-agent ids used by this model.")
    ffn_tp_size: int = Field(ge=1, description="FFN tensor-parallel degree derived from FFN device count.")
    sglang_tp_size: int = Field(ge=1, description="SGLang attention tensor-parallel degree.")
    instance_index: int = Field(ge=0, description="Integer SGLang instance index fed to the native shim ABI.")
    model_index: int = Field(ge=0, description="Integer model index fed to the native shim ABI.")


class ModelSpec(BaseModel):
    """Model metadata resolved from a local config.json through SGLang."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id whose config.json produced this metadata.")
    family: str = Field(description="SGLang/Hugging Face model family string used for diagnostics.")
    hidden_size: int = Field(ge=1, description="Hidden-state width consumed by each FFN shim call.")
    num_attention_heads: int = Field(ge=1, description="Total query-head count reported by SGLang.")
    num_key_value_heads: int = Field(ge=1, description="Total KV-head count reported by SGLang.")
    attention_kind: AttentionKind = Field(description="Attention topology derived from SGLang metadata.")
    physical_kv_lanes: int = Field(ge=1, description="Physical KV lanes used for attention policy derivation.")
    dense_intermediate_size: int | None = Field(
        default=None,
        description="Dense FFN intermediate width from config.json, when present.",
    )
    moe_intermediate_size: int | None = Field(
        default=None,
        description="MoE expert intermediate width from config.json, when present.",
    )
    num_experts: int | None = Field(default=None, description="MoE routed expert count from config.json, if present.")
    raw_config_path: Path = Field(description="Resolved config.json path used to derive this metadata.")


class ParallelPolicy(BaseModel):
    """Model-derived role-local policy for one xpool model instance."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id this policy applies to.")
    attention_kind: AttentionKind = Field(description="Attention topology used by this policy.")
    sglang_tp_size: int = Field(ge=1, description="SGLang attention tensor-parallel degree.")
    sglang_dp_size: int = Field(ge=1, description="SGLang attention data-parallel degree.")
    attention_tp_size: int = Field(ge=1, description="xpool attention tensor-parallel degree.")
    attention_dp_size: int = Field(ge=1, description="xpool attention data-parallel degree.")
    ffn_tp_size: int = Field(ge=1, description="FFN tensor-parallel degree derived from FFN devices.")
    physical_kv_lanes: int = Field(ge=1, description="Physical KV lanes used to validate attention TP.")
    enable_dp_attention: bool = Field(description="Whether attention data parallelism is active.")


class ResolvedModelConfig(BaseModel):
    """Configured model plus metadata and derived runtime policy."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Configured model id.")
    path: Path = Field(description="Absolute local model path.")
    ffn_agent_ids: list[str] = Field(description="FFN device-agent ids assigned to this model.")
    spec: ModelSpec = Field(description="SGLang-derived model metadata.")
    parallel_policy: ParallelPolicy = Field(description="Derived attention and FFN parallelism policy.")


class ResolvedRuntimeConfig(BaseModel):
    """Runtime view derived from TOML and SGLang-resolved model metadata."""

    model_config = ConfigDict(extra="forbid")

    daemon: DaemonConfig = Field(description="Resolved daemon control-plane settings.")
    scheduler: SchedulerConfig = Field(description="Resolved scheduler resource policy.")
    device_agents: list[DeviceAgentConfig] = Field(description="Derived one-agent-per-CUDA-device launch view.")
    sglang_instances: list[ModelInstanceConfig] = Field(description="Derived one-instance-per-model SGLang view.")
    models: list[ResolvedModelConfig] = Field(description="Resolved models with metadata and parallel policy.")


class XpoolConfig(BaseModel):
    """Validated xpool TOML config plus derived runtime views."""

    model_config = ConfigDict(extra="forbid")

    daemon: DaemonConfig = Field(description="Daemon control-plane config.")
    scheduler: SchedulerConfig = Field(description="Scheduler resource-concurrency config.")
    devices: DevicesConfig = Field(description="Role-local CUDA device config.")
    models: list[ModelConfig] = Field(min_length=1, description="Configured served model list.")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cli_overrides: Mapping[str, object] | None = None,
    ) -> "XpoolConfig":
        """Load and validate an xpool TOML file.

        Args:
            path: Path to the TOML config file.
            cli_overrides: Optional CLI-derived setting overrides that take
                precedence over TOML values.

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
        return cls.from_mapping(payload, cli_overrides=cli_overrides)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        cli_overrides: Mapping[str, object] | None = None,
    ) -> "XpoolConfig":
        """Validate an in-memory config mapping.

        Args:
            payload: TOML-like mapping to validate. The mapping is deep-copied
                before defaults or overrides are applied.
            cli_overrides: Optional CLI-derived setting overrides that take
                precedence over mapping values.

        Returns:
            Validated config object.

        Raises:
            ConfigError: If registry resolution fails.
            pydantic.ValidationError: If schema validation fails.

        Side Effects:
            Does not mutate ``payload``.
        """

        resolved = resolve_config_payload(payload, cli_overrides=cli_overrides)
        return cls.model_validate(resolved)

    @property
    def device_agents(self) -> list[DeviceAgentConfig]:
        """Derive one device agent for every configured CUDA device.

        Returns:
            Device-agent placement list ordered by NVSHMEM rank.
        """

        return derive_device_agents(self.devices)

    @property
    def sglang_instances(self) -> list[ModelInstanceConfig]:
        """Derive one SGLang instance for every configured model.

        Returns:
            Instance placement list in model declaration order.
        """

        return derive_model_instances(self)

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

    def resolve_runtime(self) -> ResolvedRuntimeConfig:
        """Build the full runtime view, including SGLang-derived model metadata.

        Returns:
            Runtime config with device agents, SGLang instances, model specs, and
            derived parallel policies.

        Raises:
            ConfigError: If SGLang cannot resolve model metadata.
            TopologyError: If derived attention or FFN parallelism is invalid.

        Side Effects:
            Reads each model's local ``config.json`` through SGLang's config path.
        """

        device_agents = self.device_agents
        instances = self.sglang_instances
        models: list[ResolvedModelConfig] = []
        for model, instance in zip(self.models, instances, strict=True):
            spec = load_model_spec(model.path, model_id=model.id)
            policy = derive_parallel_policy(
                spec,
                attention_device_count=len(self.devices.attention_cuda_devices),
                ffn_tp_size=len(self.devices.ffn_cuda_devices),
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
    """Load xpool config from an explicit path or ``XPOOL_CONFIG``.

    Args:
        config_path: Explicit TOML config path. When provided, it takes
            precedence over ``env["XPOOL_CONFIG"]``.
        env: Environment mapping used to read ``XPOOL_CONFIG``. Defaults to
            ``os.environ``.
        cli_overrides: Optional CLI-derived setting overrides.

    Returns:
        Validated xpool config.

    Raises:
        MissingRequiredConfig: If no config path is provided by argument or env.
        ConfigError: If registry resolution fails.
        OSError: If the config file cannot be opened.
    """

    effective_env = os.environ if env is None else env
    effective_path = config_path or effective_env.get("XPOOL_CONFIG")
    if effective_path is None:
        raise MissingRequiredConfig(
            "xpool config path is required: set the XPOOL_CONFIG environment variable "
            "(or pass --config). The SGLang plugin resolves its instance id from the "
            "one-model-one-SGLang-instance mapping in this config."
        )
    return XpoolConfig.from_file(effective_path, cli_overrides=cli_overrides)


def load_model_spec(model_path: str | Path, *, model_id: str) -> ModelSpec:
    """Load and derive model metadata for one configured model.

    Args:
        model_path: Absolute model directory or direct ``config.json`` path.
        model_id: Configured model id used in diagnostics.

    Returns:
        Model metadata resolved through SGLang plus FFN shape hints from
        ``config.json``.

    Raises:
        OSError: If ``config.json`` cannot be opened.
        ConfigError: If model metadata cannot be derived safely.
    """

    path = Path(model_path).expanduser()
    config_path = path if path.is_file() else path / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = cast(Mapping[str, object], json.load(config_file))
    return parse_model_config(raw, model_id=model_id, config_path=config_path.resolve())


def parse_model_config(raw: Mapping[str, object], *, model_id: str, config_path: Path) -> ModelSpec:
    """Derive xpool model metadata from raw config.json content.

    Args:
        raw: Parsed JSON mapping from a model ``config.json``.
        model_id: Configured model id used in diagnostics.
        config_path: Resolved path to the JSON file that produced ``raw``.

    Returns:
        xpool model metadata and FFN width hints.

    Raises:
        ConfigError: If SGLang metadata or raw integer fields are invalid.
    """

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
    """Derive attention and FFN parallelism policy for one model.

    Args:
        spec: SGLang-derived model metadata.
        attention_device_count: Number of attention CUDA devices in config.
        ffn_tp_size: FFN tensor-parallel degree derived from FFN device count.

    Returns:
        Derived role-local parallelism policy.

    Raises:
        TopologyError: If attention or FFN topology cannot be divided safely.
    """

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
    if attention_dp_size > 1:
        raise TopologyError(
            f"{spec.model_id}: attention data parallelism is not supported by the first xpool shim ABI; "
            f"reduce attention CUDA devices to {attention_tp_size} or add DP-aware shim support"
        )
    return ParallelPolicy(
        model_id=spec.model_id,
        attention_kind=spec.attention_kind,
        sglang_tp_size=attention_device_count,
        sglang_dp_size=attention_dp_size,
        attention_tp_size=attention_tp_size,
        attention_dp_size=attention_dp_size,
        ffn_tp_size=ffn_tp_size,
        physical_kv_lanes=spec.physical_kv_lanes,
        enable_dp_attention=False,
    )


def derive_device_agents(devices: DevicesConfig) -> list[DeviceAgentConfig]:
    """Derive one xpool device agent for every configured CUDA device.

    Args:
        devices: Role-local CUDA device config.

    Returns:
        Device agents ordered by NVSHMEM rank, with all attention devices first.
    """

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
    """Derive one SGLang instance per configured model.

    Args:
        config: Validated xpool config whose models and device lists should be
            converted into launch-time SGLang instance metadata.

    Returns:
        SGLang instance placement list in model declaration order.

    Preconditions:
        ``config`` has already passed xpool schema validation, including
        non-empty model and FFN device lists.

    FFN device assignment is intentionally shared across models: every instance uses
    the full FFN device-agent pool, so the FFN TP degree is derived from
    ``devices.ffn_cuda_devices`` rather than from per-model placement knobs. FFN device
    multiplexing is arbitrated at runtime by the device-agent scheduler, not by static
    partitioning here. ``instance_index``/``model_index`` are the integer identities
    (model declaration order) fed to the xpool FFN shim ABI; under the
    one-model-one-SGLang-instance mapping they are equal, but kept distinct so a future
    one-model-multi-instance topology can renumber instances independently.
    """

    ffn_agents = [agent for agent in config.device_agents if agent.role is DeviceRole.FFN]
    return [
        ModelInstanceConfig(
            id=model.id,
            model_id=model.id,
            attention_cuda_devices=list(config.devices.attention_cuda_devices),
            ffn_agent_ids=[agent.id for agent in ffn_agents],
            ffn_tp_size=len(ffn_agents),
            sglang_tp_size=len(config.devices.attention_cuda_devices),
            instance_index=index,
            model_index=index,
        )
        for index, model in enumerate(config.models)
    ]


def resolve_config_payload(
    payload: Mapping[str, object],
    *,
    cli_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply registry defaults and CLI overrides to a TOML-like payload.

    Args:
        payload: Config mapping parsed from TOML.
        cli_overrides: Optional CLI-derived setting overrides.

    Returns:
        Deep-copied resolved payload ready for Pydantic validation.

    Raises:
        ConfigError: If a setting cannot be parsed or a nested override is invalid.
        MissingRequiredConfig: If a required config path is absent or empty.

    Side Effects:
        Does not mutate ``payload``.
    """

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
    """Render the config registry as JSON-serializable dictionaries.

    Returns:
        Registry entries with source, default, CLI, and description metadata.
    """

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
