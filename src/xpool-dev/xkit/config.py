"""Assemble catalogue scenes through the existing runtime configuration owner."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PrivateAttr, TypeAdapter, field_validator

import xpool.config
from xpool.config import (
    AtnConfig,
    FfnConfig,
    LatencySloConfig,
    MissingRequiredConfig,
    ModelConfig,
    SchedulerConfig,
    XpoolConfig,
    validate_device_layout,
)
from xpool.model import ModelId
from xpool.utils.config import ConfigError, ConfigModel, ConfigSetting, ConfigSource, ConfigSourceRecord

__all__ = [
    "CONFIG_REGISTRY",
    "DeploymentConfig",
    "FigureConfig",
    "LegendConfig",
    "ToolConfigRecord",
    "XbenchConfig",
    "XbenchReportConfig",
    "XpoolDevConfig",
    "XtestConfig",
    "assemble_config",
    "get_global_config",
    "init_global_config",
    "merge_config",
    "resolve_model_weights",
    "select_models",
]


class LegendConfig(BaseModel):
    """Legend presentation; text comes from retained Model IDs and metric series."""

    model_config = ConfigDict(extra="forbid")
    visible: bool
    location: Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ]
    columns: int = Field(strict=True, gt=0)


class FigureConfig(BaseModel):
    """Exported canvas dimensions in inches and internal subplot arrangement."""

    model_config = ConfigDict(extra="forbid")
    width_inches: FiniteFloat | None = Field(gt=0)
    height_inches: FiniteFloat | None = Field(gt=0)
    columns: int = Field(strict=True, gt=0)
    caption: str | None


class XbenchReportConfig(BaseModel):
    """Presentation settings independent of measurement and original verdict."""

    model_config = ConfigDict(extra="forbid")
    title: str | None
    layout: Literal["single", "double", "half"]
    formats: tuple[Literal["pdf", "svg", "png"], ...] = Field(min_length=1)
    ppi: int = Field(strict=True, gt=0)
    legend: LegendConfig
    ttft: FigureConfig
    itl: FigureConfig
    throughput: FigureConfig

    @field_validator("formats")
    @classmethod
    def validate_formats(
        cls, value: tuple[Literal["pdf", "svg", "png"], ...]
    ) -> tuple[Literal["pdf", "svg", "png"], ...]:
        """Reject duplicate outputs at the configuration boundary."""
        if len(set(value)) != len(value):
            raise ValueError("report formats must be unique")
        return value


class XtestConfig(BaseModel):
    """Test catalogue and default suite/requirement policy, not pytest actions."""

    model_config = ConfigDict(extra="forbid")
    catalog: Path
    suites: tuple[str, ...] = Field(min_length=1)
    strict_requirements: bool


class XbenchConfig(BaseModel):
    """Benchmark catalogue, repetition count and presentation, not case selection."""

    model_config = ConfigDict(extra="forbid")
    catalog: Path
    repetitions: int = Field(strict=True, gt=0)
    report: XbenchReportConfig


CONFIG_REGISTRY: tuple[ConfigSetting, ...] = (
    ConfigSetting(
        name="config_path",
        path=None,
        parser="str",
        allowed_sources=(ConfigSource.CLI, ConfigSource.ENV),
        cli="--config",
        env_var="XKIT_CONFIG",
        description="Development configuration file; runtime selection uses XPOOL_CONFIG.",
    ),
    ConfigSetting(
        name="keep_runs",
        path=("keep_runs",),
        parser="int",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=20,
        cli="--keep",
        description="Number of inactive runs retained by cleanup.",
    ),
    ConfigSetting(
        name="xtest_catalog",
        path=("xtest", "catalog"),
        parser="path",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default="tests/tests.toml",
        cli="--catalog",
        description="Test catalogue.",
    ),
    ConfigSetting(
        name="xtest_suites",
        path=("xtest", "suites"),
        parser="raw",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=("cext", "unit", "integration", "e2e"),
        cli="--suite",
        cli_action="append",
        description="Test suite or full Model ID; repeat to select multiple suites.",
    ),
    ConfigSetting(
        name="xtest_strict_requirements",
        path=("xtest", "strict_requirements"),
        parser="bool",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=False,
        cli="--strict-requirements",
        description="Treat unavailable declared test resources as failures.",
    ),
    ConfigSetting(
        name="xbench_catalog",
        path=("xbench", "catalog"),
        parser="path",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default="benches/benches.toml",
        cli="--catalog",
        description="Benchmark catalogue.",
    ),
    ConfigSetting(
        name="xbench_repetitions",
        path=("xbench", "repetitions"),
        parser="int",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=1,
        cli="--repetitions",
        description="Executions per selected benchmark case; each reuses the prepared workload.",
    ),
    ConfigSetting(
        name="xbench_report_layout",
        path=("xbench", "report", "layout"),
        parser="str",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default="single",
        cli="--layout",
        cli_choices=("single", "double", "half"),
        description="Default canvas-width preset.",
    ),
    ConfigSetting(
        name="xbench_report_formats",
        path=("xbench", "report", "formats"),
        parser="raw",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=("pdf", "svg", "png"),
        cli="--format",
        cli_action="append",
        cli_choices=("pdf", "svg", "png"),
        description="Figure output format; repeat to select multiple formats.",
    ),
    ConfigSetting(
        name="xbench_report_ppi",
        path=("xbench", "report", "ppi"),
        parser="int",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=300,
        cli="--ppi",
        description="Raster pixels per inch.",
    ),
    ConfigSetting(
        name="xbench_report_legend_visible",
        path=("xbench", "report", "legend", "visible"),
        parser="bool",
        allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT),
        default=True,
        cli="--legend",
        description="Show metric-series legends.",
    ),
    *(
        ConfigSetting(
            name=f"xbench_report_{name}",
            path=("xbench", "report", *path),
            parser="raw",
            allowed_sources=(ConfigSource.CONFIG, ConfigSource.DEFAULT),
            default=default,
            description=description,
        )
        for name, path, default, description in (
            ("title", ("title",), None, "Report heading."),
            ("legend_location", ("legend", "location"), "upper center", "Legend placement inside the canvas."),
            ("legend_columns", ("legend", "columns"), 1, "Legend column count."),
        )
    ),
    *(
        ConfigSetting(
            name=f"xbench_report_{figure}_{field}",
            path=("xbench", "report", figure, field),
            parser="raw",
            allowed_sources=(ConfigSource.CLI, ConfigSource.CONFIG, ConfigSource.DEFAULT)
            if field in {"width_inches", "height_inches"}
            else (ConfigSource.CONFIG, ConfigSource.DEFAULT),
            default=default,
            description=description,
        )
        for figure in ("ttft", "itl", "throughput")
        for field, default, description in (
            ("width_inches", None, "Exported canvas width in inches; unset uses the layout preset."),
            ("height_inches", None, "Exported canvas height in inches; unset uses 2.4 inches per row."),
            ("columns", 1, "Subplot column count."),
            ("caption", None, "Markdown figure caption."),
        )
    ),
)


class XpoolDevConfig(ConfigModel):
    """Development settings with a production-owned, bootstrap-resolved cache view."""

    registry: ClassVar[tuple[ConfigSetting, ...]] = CONFIG_REGISTRY
    env_prefix: ClassVar[str] = "XKIT_"

    keep_runs: int = Field(strict=True, gt=0)
    xtest: XtestConfig
    xbench: XbenchConfig

    _cache_root: Path | None = PrivateAttr(default=None)
    _cache_source: ConfigSourceRecord | None = PrivateAttr(default=None)
    _development_config_path: Path | None = PrivateAttr(default=None)
    _runtime_config_path: Path | None = PrivateAttr(default=None)

    @property
    def cache_root(self) -> Path:
        """Return the invocation's normalized production cache root after bootstrap."""
        if self._cache_root is None:
            raise MissingRequiredConfig("development cache has not been resolved by init_global_config")
        return self._cache_root

    def record(self) -> ToolConfigRecord:
        """Capture typed execution settings and provenance for evidence and workers."""
        if self._cache_source is None:
            raise MissingRequiredConfig("development cache provenance has not been resolved")
        return ToolConfigRecord(
            settings=self,
            sources=self.sources,
            cache_root=self.cache_root,
            cache_source=self._cache_source,
            development_config_path=self._development_config_path,
            runtime_config_path=self._runtime_config_path,
        )

    @classmethod
    def from_record(cls, record: ToolConfigRecord) -> Self:
        """Restore the validated parent snapshot without file or environment reads."""
        config = cls.model_validate(record.settings.model_dump())
        config._sources = record.sources
        config._cache_root = record.cache_root
        config._cache_source = record.cache_source
        config._development_config_path = record.development_config_path
        config._runtime_config_path = record.runtime_config_path
        return config


class ToolConfigRecord(BaseModel):
    """Required execution evidence; its typed tree is also the worker snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    settings: XpoolDevConfig
    sources: tuple[ConfigSourceRecord, ...]
    cache_root: Path
    cache_source: ConfigSourceRecord
    development_config_path: Path | None
    runtime_config_path: Path | None


global_config: XpoolDevConfig | None = None
global_config_lock = Lock()


def init_global_config(
    resolved: XpoolDevConfig | None = None,
    *,
    config_path: str | Path | None = None,
    cli: Mapping[str, object] | None = None,
) -> XpoolDevConfig:
    """Install development policy once, or a parent's resolved worker snapshot.

    CLI/file resolution uses this process environment. A selected development
    or runtime file must be readable TOML. Offline cache lookup validates no
    runtime deployment. A resolved snapshot is exclusive with file/CLI inputs
    and reads no files or environment. Equal effective values are idempotent;
    different settings or cache locations raise ConfigError.
    """
    if resolved is not None:
        if config_path is not None or cli is not None:
            raise ConfigError("resolved development configuration is exclusive with file/CLI inputs")
        resolved.cache_root
    else:
        inputs = dict(cli or {})
        if config_path is not None:
            inputs["config_path"] = str(config_path)
        for dimension in ("width", "height"):
            if dimension in inputs:
                value = inputs.pop(dimension)
                for figure in ("ttft", "itl", "throughput"):
                    inputs[f"xbench_report_{figure}_{dimension}_inches"] = value
        path_setting = next(setting for setting in CONFIG_REGISTRY if setting.name == "config_path")
        path_value, path_source = path_setting.resolve({}, inputs, os.environ)
        development_path = None if path_source is None else Path(cast(str, path_value)).expanduser().resolve()
        resolved = (
            XpoolDevConfig.from_mapping({}, cli=inputs, env=os.environ)
            if development_path is None
            else XpoolDevConfig.from_file(development_path, cli=inputs, env=os.environ)
        )
        resolved._cache_root, resolved._cache_source = XpoolConfig.resolve_cache_root(cli=inputs, env=os.environ)
        resolved._development_config_path = development_path
        runtime_path_setting = next(
            setting for setting in xpool.config.CONFIG_REGISTRY if setting.name == "config_path"
        )
        runtime_path, runtime_source = runtime_path_setting.resolve({}, {}, os.environ)
        resolved._runtime_config_path = (
            None if runtime_source is None else Path(cast(str, runtime_path)).expanduser().resolve()
        )
    global global_config
    with global_config_lock:
        if global_config is not None:
            if (
                global_config.model_dump(mode="json") != resolved.model_dump(mode="json")
                or global_config.cache_root != resolved.cache_root
            ):
                raise ConfigError("development global config is already initialized with different values")
            return global_config
        global_config = resolved
    return resolved


def get_global_config() -> XpoolDevConfig:
    """Return the entry-installed aggregate; fail if bootstrap has not run."""
    if global_config is None:
        raise MissingRequiredConfig("development global config has not been loaded")
    return global_config


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Validated portable values used before complete runtime resolution."""

    atn: AtnConfig
    ffn: FfnConfig
    ffn_concurrency: int
    slo: LatencySloConfig
    models: tuple[ModelConfig, ...]

    @classmethod
    def from_file(cls, path: Path, *, model_ids: Sequence[ModelId]) -> Self:
        """Read complete portable topology and SLO for the selected model set.

        Host paths and unrelated runtime policies belong to the complete base.
        XpoolConfig field types own value validation. Every selected model supplies
        explicit attention and FFN geometry; graph policy remains catalogue-owned.
        """
        with path.open("rb") as source:
            payload = tomllib.load(source)
        fields = {
            "atn": {"devices", "device_memory_utilization"},
            "ffn": {"devices"},
            "scheduler": {"ffn_concurrency", "slo"},
            "models": {"id", "atn_tp_size", "atn_dp_size", "ffn_tp_size", "slo"},
        }
        required = {
            "atn": {"devices"},
            "ffn": {"devices"},
            "scheduler": {"ffn_concurrency", "slo"},
            "models": {"id", "atn_tp_size", "atn_dp_size", "ffn_tp_size"},
        }
        if unknown := payload.keys() - fields.keys():
            raise ConfigError(f"deployment fields are not portable: {sorted(unknown)}")
        if missing := required.keys() - payload.keys():
            raise ConfigError(f"deployment requires fields: {sorted(missing)}")
        for name, value in payload.items():
            entries = value if name == "models" and isinstance(value, list) else [value]
            if name == "models" and not isinstance(value, list):
                raise ConfigError("deployment models must be an array of tables")
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) - fields[name]:
                    raise ConfigError(f"deployment {name} contains unsupported fields")
                if missing := required[name] - entry.keys():
                    raise ConfigError(f"deployment {name} requires fields: {sorted(missing)}")
        atn = AtnConfig.model_validate(payload["atn"])
        ffn = FfnConfig.model_validate(payload["ffn"])
        validate_device_layout(atn, ffn)
        scheduler = payload["scheduler"]
        ffn_concurrency = TypeAdapter[int](
            SchedulerConfig.model_fields["ffn_concurrency"].rebuild_annotation()
        ).validate_python(scheduler["ffn_concurrency"])
        slo = LatencySloConfig.model_validate(scheduler["slo"])
        models = TypeAdapter(list[ModelConfig]).validate_python(payload["models"])
        declared_ids = tuple(model.id for model in models)
        if len(declared_ids) != len(set(declared_ids)) or set(declared_ids) != set(model_ids):
            raise ConfigError("deployment models must match the selected Model IDs exactly")
        for model in models:
            if model.atn_tp_size is None or model.atn_tp_size * model.atn_dp_size != len(atn.devices):
                raise ConfigError(f"{model.id}: deployment attention TP times DP must equal the attention World")
            if model.ffn_tp_size is None or model.ffn_tp_size > len(ffn.devices):
                raise ConfigError(f"{model.id}: deployment FFN TP must fit the FfnAgent Fleet")
        return cls(atn=atn, ffn=ffn, ffn_concurrency=ffn_concurrency, slo=slo, models=tuple(models))

    @property
    def model_by_id(self) -> Mapping[ModelId, ModelConfig]:
        """Return portable model policy keyed by canonical identity."""
        return MappingProxyType({model.id: model for model in self.models})


def select_models(models: Sequence[ModelConfig], model_ids: Sequence[ModelId]) -> list[ModelConfig]:
    """Preserve matching model policy; unregistered IDs use the vendor-root fallback."""
    selected = []
    for model_id in model_ids:
        matches = [model for model in models if model.id == model_id]
        if len(matches) > 1:
            raise ConfigError(f"base configuration contains duplicate model ID: {model_id}")
        selected.append(matches[0] if matches else ModelConfig(id=model_id))
    return selected


def merge_config(payload: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    """Merge explicit layers before defaults; None clears inherited optional fields."""
    merged = deepcopy(dict(payload))
    for name, value in overrides.items():
        inherited = merged.get(name)
        if value is None:
            merged.pop(name, None)
        elif isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[name] = merge_config(cast(Mapping[str, object], inherited), cast(Mapping[str, object], value))
        else:
            merged[name] = deepcopy(value)
    return merged


def resolve_model_weights(config: XpoolConfig, model_id: ModelId) -> Path:
    """Resolve a local checkpoint and require its directory and model metadata.

    Existing model paths take precedence over the vendor-root fallback.
    Unavailable paths raise ValueError; the consuming tool owns failure policy.
    """
    selected = config.model_copy(update={"models": select_models(config.models, (model_id,))})
    path = selected.model_path_of(model_id).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"model {model_id!r} weight directory is unavailable at {path}")
    if not (path / "config.json").is_file():
        raise ValueError(f"model {model_id!r} has no config.json at {path}")
    return path


def assemble_config(
    model_ids: Sequence[ModelId],
    *,
    deployment: Path | None = None,
    runtime_config: Path | None = None,
    overrides: Mapping[str, object] | None = None,
    cli: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
) -> XpoolConfig:
    """Select models and resolve a scene against a complete base exactly once.

    The explicit base path takes precedence over XPOOL_CONFIG. Portable scene
    and caller-owned test projections replace CONFIG-layer values; registered
    CLI/environment precedence and defaults are applied by XpoolConfig afterward.
    Portable scenes replace model geometry and SLO while retaining checkpoint
    paths and machine policy. New IDs use the inherited vendor root. No global
    configuration or resource is acquired.
    """
    environment = os.environ if env is None else env
    inputs = dict(cli or {})
    if runtime_config is not None:
        inputs["config_path"] = str(runtime_config)
    setting = next(setting for setting in xpool.config.CONFIG_REGISTRY if setting.name == "config_path")
    path, source = setting.resolve({}, inputs, environment)
    if source is None:
        raise MissingRequiredConfig("set XPOOL_CONFIG or supply a complete runtime_config")
    base_path = Path(cast(str, path)).expanduser().resolve()
    with base_path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    inherited = TypeAdapter(list[ModelConfig]).validate_python(payload.get("models", []))
    selected = {
        model.id: model.model_dump(mode="json", exclude_unset=True) for model in select_models(inherited, model_ids)
    }
    scene_layer: dict[str, object] = {}
    scene_models: tuple[ModelConfig, ...] = ()
    if deployment is not None:
        scene = DeploymentConfig.from_file(deployment, model_ids=model_ids)
        scheduler: dict[str, object] = {
            "ffn_concurrency": scene.ffn_concurrency,
            "slo": scene.slo.model_dump(mode="json"),
        }
        for model in selected.values():
            for field in ("atn_tp_size", "atn_dp_size", "ffn_tp_size", "slo"):
                model.pop(field, None)
        scene_layer = {
            "atn": scene.atn.model_dump(mode="json", exclude_unset=True),
            "ffn": scene.ffn.model_dump(mode="json", exclude_unset=True),
            "scheduler": scheduler,
        }
        scene_models = scene.models
    caller_layer = deepcopy(dict(overrides or {}))
    caller_models = TypeAdapter(list[ModelConfig]).validate_python(caller_layer.pop("models", []))
    for layer, model_overrides in ((scene_layer, scene_models), (caller_layer, caller_models)):
        seen: set[ModelId] = set()
        for model in model_overrides:
            if model.id not in selected or model.id in seen:
                raise ConfigError(f"scene model override is duplicate or not selected: {model.id}")
            seen.add(model.id)
            merged = merge_config(selected[model.id], model.model_dump(mode="json", exclude_unset=True))
            selected[model.id] = merged
        payload = merge_config(payload, layer)
    payload["models"] = list(selected.values())
    config = XpoolConfig.from_mapping(payload, cli=inputs, env=environment, origin=base_path.parent)
    for model in config.models:
        if model.ffn_tp_size is not None and model.ffn_tp_size > len(config.ffn.devices):
            raise ConfigError(f"{model.id}: FFN TP exceeds the selected FfnAgent Fleet")
    return config
