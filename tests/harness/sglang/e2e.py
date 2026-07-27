"""E2E endpoint reservation and task-local config materialization."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import tomli_w

from tests.harness.sglang.manifest import E2eManifest, E2eServingCase
from xpool.config import FfnSchedulingPolicy, LoopbackSite, XpoolConfig


@dataclass(frozen=True, slots=True)
class E2eLaunchModel:
    """One fully resolved model placement for a materialized E2E attempt."""

    alias: str
    model_id: str
    architecture: str
    max_total_tokens: int
    atn_tp_size: int
    atn_dp_size: int


@dataclass(frozen=True, slots=True)
class E2eLaunch:
    """Non-owning effective config and environment for one E2E attempt."""

    case_id: str
    models: tuple[E2eLaunchModel, ...]
    config: XpoolConfig
    config_path: Path
    environment: Mapping[str, str]
    loopback_site: LoopbackSite
    observer_outdir: Path


def materialize(
    manifest: E2eManifest,
    case: E2eServingCase,
    *,
    base_config: XpoolConfig,
    workdir: Path,
    daemon_port: int,
    loopback_site: LoopbackSite = LoopbackSite.FFNAGENT,
) -> E2eLaunch:
    """Materialize one task-local config and sanitized runtime environment."""

    if daemon_port <= 0:
        raise ValueError("E2E daemon_port must be positive")
    model_base_uri = base_config.vendor.model_base_uri
    if model_base_uri is None:
        raise ValueError("E2E requires vendor.model_base_uri in the external XPOOL_CONFIG")
    configured_models = {model.id: model for model in base_config.models}
    launch_models = tuple(
        E2eLaunchModel(
            alias=model.alias,
            model_id=model.model_id,
            architecture=model.architecture,
            max_total_tokens=model.max_total_tokens,
            atn_tp_size=placement.atn_tp_size,
            atn_dp_size=placement.atn_dp_size,
        )
        for placement in case.models
        for model in (manifest.model(placement.model),)
    )
    for model in launch_models:
        configured = configured_models.get(model.model_id)
        if configured is None:
            raise ValueError(f"E2E model {model.model_id!r} is absent from the external XPOOL_CONFIG")
        if configured.path is not None:
            raise ValueError(
                f"E2E model {model.model_id!r} must resolve only through vendor.model_base_uri, not models[].path"
            )
        validate_model_architecture(model, base_config.model_path_of(model.model_id))

    workdir.mkdir(parents=True, exist_ok=True)
    observer_outdir = (workdir / "observers").resolve()
    observer_outdir.mkdir(parents=True, exist_ok=True)
    config_path = (workdir / "xpool.toml").resolve()
    scheduler: dict[str, object] = {
        "atn_concurrency": len(case.models),
        "ffn_concurrency": case.executor_count,
        "ffn_policy": base_config.scheduler.ffn_policy.value,
    }
    if base_config.scheduler.ffn_policy is FfnSchedulingPolicy.RANDOM:
        scheduler["ffn_random_seed"] = base_config.scheduler.ffn_random_seed
    payload: dict[str, object] = {
        "daemon": {
            "host": base_config.daemon.host,
            "port": daemon_port,
        },
        "scheduler": scheduler,
        "vendor": {"model_base_uri": str(model_base_uri)},
        "devices": {
            "atn_cuda_devices": list(range(case.atnagent_count)),
            "ffn_cuda_devices": list(range(case.atnagent_count, case.required_gpu_count)),
        },
        "models": [{"id": model.model_id} for model in launch_models],
    }
    config_path.write_text(tomli_w.dumps(payload), encoding="utf-8")

    environment = runtime_environment(
        config_path=config_path,
        observer_outdir=observer_outdir,
        loopback_site=loopback_site,
        transport_trace_capacity=case.transport_trace_capacity,
        fabric_trace_capacity=case.fabric_trace_capacity,
    )
    config = XpoolConfig.from_file(config_path, env=environment)
    return E2eLaunch(
        case_id=case.id,
        models=launch_models,
        config=config,
        config_path=config_path,
        environment=MappingProxyType(environment),
        loopback_site=loopback_site,
        observer_outdir=observer_outdir,
    )


def runtime_environment(
    *,
    config_path: Path,
    observer_outdir: Path,
    loopback_site: LoopbackSite,
    transport_trace_capacity: int,
    fabric_trace_capacity: int,
) -> dict[str, str]:
    """Return a sanitized process environment for the E2E runtime tree."""

    inherited_names = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
    )
    environment = {name: os.environ[name] for name in inherited_names if name in os.environ}
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "SGLANG_PLUGINS": "xpool",
            "TRANSFORMERS_OFFLINE": "1",
            "XPOOL_CONFIG": str(config_path),
            "XPOOL_DEBUG_LOOPBACK_ENABLE": "1",
            "XPOOL_DEBUG_LOOPBACK_SITE": loopback_site.value,
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_TRACE_CAPACITY": str(transport_trace_capacity),
            "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_FABRIC_OBSERVER_TRACE_CAPACITY": str(fabric_trace_capacity),
        }
    )
    return environment


def validate_model_architecture(model: E2eLaunchModel, model_path: Path) -> None:
    """Require configured weights to expose the manifest architecture."""

    config_path = model_path / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"E2E model {model.model_id!r} has no readable config.json at {config_path}") from exc
    architectures = raw.get("architectures") if isinstance(raw, dict) else None
    if not isinstance(architectures, list) or model.architecture not in architectures:
        raise ValueError(
            f"E2E model {model.model_id!r} must expose architecture {model.architecture!r}, got {architectures!r}"
        )


def model_id_slug(model_id: str) -> str:
    """Normalize one model ID for deterministic artifact paths."""

    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
