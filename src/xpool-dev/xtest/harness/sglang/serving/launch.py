"""E2E endpoint reservation and task-local config materialization."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import xkit.device
from xkit.config import assemble_config
from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xkit.serving.sglang.deterministic import DeterministicConfig
from xkit.serving.sglang.graph import SglangGraphSettings
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.support.environment import qualification_environment


def prepare(
    case: E2eServingCase,
    *,
    base_config: ResolvedConfig,
    workdir: Path,
    graph_settings: SglangGraphSettings,
    deterministic_mode: Literal["none", "record", "replay"] = "none",
    replay_path: Path | None = None,
) -> ServingLaunch:
    """Prepare deployment policy and test observer inputs for startup.

    Shared serving startup owns endpoint binding and runtime snapshot creation.
    The input preserves declared model order and uses the selected base's model
    paths, while applying deployment-owned SLO and Elastic KV memory policy.
    """

    launch_models = tuple(
        SglangLaunchModel(
            model_id=model_id,
            graph_settings=graph_settings,
        )
        for model_id in case.models
    )
    workdir.mkdir(parents=True, exist_ok=True)
    observer_outdir = (workdir / "observers").resolve()
    observer_outdir.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, object] = {}
    scene = case.deployment_config
    atn_overrides: dict[str, object] = {}
    if case.elastic_kv is not None:
        atn_devices = scene.atn.devices
        visible_memory = xkit.device.query_visible_device_total_memory_bytes()
        atn_memory = tuple(visible_memory[device] for device in atn_devices if device < len(visible_memory))
        if len(atn_memory) != case.atnagent_count:
            raise ValueError(
                f"E2E elastic KV case requires {case.atnagent_count} visible attention devices, got {len(atn_memory)}"
            )
        if len(set(atn_memory)) != 1:
            raise ValueError("E2E elastic KV case requires attention devices with equal total memory")
        utilization = base_config.config.atn.device_memory_utilization
        if "device_memory_utilization" in scene.atn.model_fields_set:
            utilization = scene.atn.device_memory_utilization
        atn_overrides["device_memory_utilization"] = min(
            utilization,
            case.elastic_kv.atn_device_memory_budget_bytes / atn_memory[0],
        )
        overrides["atn"] = atn_overrides

    environment = runtime_environment(observer_outdir=observer_outdir)
    if deterministic_mode != "none":
        path = workdir / "sampling-record.jsonl" if deterministic_mode == "record" else replay_path
        if path is None:
            raise ValueError("replay launch requires a reference record path")
        path = path.resolve()
        DeterministicConfig(deterministic_mode, path).prepare()
        environment["SGLANG_PLUGINS"] = "xpool,xkit_deterministic"
        DETERMINISTIC_MODE.set(deterministic_mode, env=environment)
        DETERMINISTIC_PATH.set(str(path), env=environment)
    config = assemble_config(
        tuple(model.model_id for model in launch_models),
        deployment=case.deployment,
        runtime_config=base_config.path,
        overrides=overrides,
        env={**os.environ, **environment},
    )
    return ServingLaunch(
        models=launch_models,
        config=config,
        environment=MappingProxyType(environment),
        cwd=Path.cwd(),
    )


def runtime_environment(
    *,
    observer_outdir: Path,
) -> dict[str, str]:
    """Return a sanitized process environment for the E2E runtime tree."""

    environment = qualification_environment()
    environment.update(
        {
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY": "32768",
            "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR": str(observer_outdir),
            # The Coordinator and first FfnAgent share one Fabric trace buffer.
            "XPOOL_DEBUG_FABRIC_OBSERVER_RECORD_CAPACITY": "65536",
        }
    )
    return environment
