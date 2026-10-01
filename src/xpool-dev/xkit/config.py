"""Assemble catalogue scenes through the existing runtime configuration owner."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from pydantic import TypeAdapter

from xpool.config import (
    CONFIG_REGISTRY,
    AtnConfig,
    ConfigError,
    FfnConfig,
    LatencySloConfig,
    MissingRequiredConfig,
    ModelConfig,
    SchedulerConfig,
    XpoolConfig,
)
from xpool.model import ModelId


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Validated portable values used before complete runtime resolution."""

    atn: AtnConfig
    ffn: FfnConfig
    ffn_concurrency: int
    slo: LatencySloConfig
    models: tuple[ModelConfig, ...]

    @property
    def model_by_id(self) -> Mapping[ModelId, ModelConfig]:
        """Return portable model policy keyed by canonical identity."""
        return MappingProxyType({model.id: model for model in self.models})


def load_deployment(path: Path, *, model_ids: Sequence[ModelId]) -> DeploymentConfig:
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
    if sorted(atn.devices + ffn.devices) != list(range(len(atn.devices) + len(ffn.devices))):
        raise ConfigError("deployment devices must partition contiguous lease-local indices starting at zero")
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
    return DeploymentConfig(atn=atn, ffn=ffn, ffn_concurrency=ffn_concurrency, slo=slo, models=tuple(models))


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
    setting = next(setting for setting in CONFIG_REGISTRY if setting.name == "config_path")
    path, source = setting.resolve({}, inputs, environment)
    if source is None:
        raise MissingRequiredConfig("set XPOOL_CONFIG or supply a complete runtime_config")
    with Path(cast(str, path)).expanduser().open("rb") as config_file:
        payload = tomllib.load(config_file)
    inherited = TypeAdapter(list[ModelConfig]).validate_python(payload.get("models", []))
    selected = {
        model.id: model.model_dump(mode="json", exclude_unset=True) for model in select_models(inherited, model_ids)
    }
    scene_layer: dict[str, object] = {}
    scene_models: tuple[ModelConfig, ...] = ()
    if deployment is not None:
        scene = load_deployment(deployment, model_ids=model_ids)
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
    config = XpoolConfig.from_mapping(payload, cli=inputs, env=environment)
    for model in config.models:
        if model.ffn_tp_size is not None and model.ffn_tp_size > len(config.ffn.devices):
            raise ConfigError(f"{model.id}: FFN TP exceeds the selected FfnAgent Fleet")
    return config
