"""E2E endpoint reservation and task-local config materialization."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import xkit.device
from xkit.config import assemble_config
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase

SERVING_OBSERVER_RECORD_CAPACITY = 32768


def prepare(
    case: E2eServingCase,
    *,
    base_config: ResolvedConfig,
    workdir: Path,
    graph_settings: SglangGraphSettings,
) -> ServingLaunch:
    """Prepare deployment policy and test observer inputs for startup.

    Shared serving startup owns endpoint binding and runtime snapshot creation.
    The input preserves declared model order and uses the selected base's model
    paths, while applying deployment-owned SLO and Elastic KV memory policy.
    """

    graph_mode = next(mode for mode in SglangGraphMode if mode.settings() == graph_settings)
    launch_models = tuple(
        SglangLaunchModel(
            model_id=model_id,
            graph_mode=graph_mode,
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

    environment = runtime_environment(
        observer_outdir=observer_outdir,
        prefill_logit_observer=(
            {SglangGraphMode.EAGER, SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE}.issubset(case.graph_modes)
            and graph_settings
            in {SglangGraphMode.EAGER.settings(), SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings()}
        ),
    )
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
    prefill_logit_observer: bool,
) -> dict[str, str]:
    """Return a sanitized process environment for the E2E runtime tree."""

    inherited_names = (
        "PATH",
        "HOME",
        "LOGNAME",
        "USER",
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
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY": str(SERVING_OBSERVER_RECORD_CAPACITY),
            "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_FABRIC_OBSERVER_RECORD_CAPACITY": str(SERVING_OBSERVER_RECORD_CAPACITY),
        }
    )
    if prefill_logit_observer:
        environment.update(
            {
                "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE": "1",
                "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_OUTDIR": str(observer_outdir),
            }
        )
    return environment
