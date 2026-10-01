"""Multi-model serving startup, warmup and replay of one prepared workload."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import signal
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

import xbench
from xbench.harness.serving.case import BenchCase, OwnedBenchCase
from xbench.harness.serving.client import MeasurementRecorder, run_measurement
from xbench.harness.serving.execution import (
    ENVIRONMENT_KEYS,
    capture_owned_metadata,
    package_versions,
    warmup,
    write_environment,
)
from xbench.harness.serving.measure import BenchCaseManifest, RepetitionManifest, RequestRecord
from xbench.harness.serving.workload import PreparedWorkload
from xkit import ResourceRequirements
from xkit.config import load_deployment
from xkit.results import write_json
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.config import CONFIG_REGISTRY, XpoolConfig, init_global_config
from xpool.model import ModelId


def requirements_of(case: BenchCase) -> ResourceRequirements:
    """Declare owned serving resources while leaving external endpoints unleased."""

    if not isinstance(case, OwnedBenchCase):
        return ResourceRequirements(0, False, False, ())
    model_ids = tuple(target.model_id for target in case.targets)
    deployment = load_deployment(case.deployment, model_ids=model_ids)
    return ResourceRequirements(len(deployment.atn.devices) + len(deployment.ffn.devices), True, True, model_ids)


@xbench.parameterize("case")
@xbench.requirements(requirements_of)
def bench_multi_model_serving(case: BenchCase, workdir: Path) -> None:
    """Run one repetition against owned serving or externally supplied endpoints.

    Prepared prompts and arrivals belong to the parent case directory. Startup,
    warmup and local shutdown are outside the measured window; the outer harness
    certifies descendant cleanup and seals the retained request verdict.
    """

    asyncio.run(run_serving(case, workdir))


async def run_serving(case: BenchCase, directory: Path) -> None:
    case_manifest = BenchCaseManifest.model_validate_json((directory.parent / "case.json").read_bytes())
    workload = PreparedWorkload.load(directory.parent, case=case)
    system = None
    recorder = MeasurementRecorder(directory)
    infrastructure_error = None
    cancelled_signal = None
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("benchmark worker requires a running task")
    loop = asyncio.get_running_loop()

    def cancel(signum: int) -> None:
        nonlocal cancelled_signal
        cancelled_signal = signum
        task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, cancel, signum)
    capture_errors: list[str] = []
    environment: dict[str, JsonValue] = {
        "mode": case.mode,
        "tool_software": {
            "source": "local_distribution_metadata",
            "python": platform.python_version(),
            "packages": package_versions(("xpool-dev", "xpool", "httpx"), capture_errors),
        },
        "serving_metadata_source": "declared" if case_manifest.serving_metadata is not None else "unknown",
        "serving_metadata": None,
        "capture_errors": capture_errors,
        "environment_source": "unknown" if isinstance(case, OwnedBenchCase) else "local_client",
        "environment": {}
        if isinstance(case, OwnedBenchCase)
        else {name: os.environ[name] for name in ENVIRONMENT_KEYS if name in os.environ},
        "local_gpu_inventory": None,
        "cache_policy": {
            "controller": "owned" if isinstance(case, OwnedBenchCase) else "external",
            "flush_performed": False,
            "serving_state_reset": False,
            "warmup_requests_per_target": case.warmup_requests_per_target,
        },
    }
    write_environment(directory, environment)
    write_json(directory / "warmup.json", {"requests": [], "phase": "not_started"})
    try:
        if workload.requests:
            if isinstance(case, OwnedBenchCase):
                deployment = case_manifest.deployment
                if deployment is None:
                    raise ValueError("owned workload lacks its resolved configuration")
                config = XpoolConfig.model_validate_json(json.dumps(deployment.effective))
                launch = ServingLaunch(
                    config=config,
                    environment=dict(os.environ),
                    cwd=deployment.cwd,
                    models=tuple(SglangLaunchModel(target.model_id, target.graph_mode) for target in case.targets),
                )
                print("xbench phase=startup", file=sys.stderr)
                system = XpoolServingSystem.start(
                    launch, workdir=directory, startup_timeout_seconds=case.startup_timeout_seconds
                )
                for setting in CONFIG_REGISTRY:
                    if setting.env_var is not None:
                        os.environ.pop(setting.env_var, None)
                os.environ.update(system.launch.environment)
                init_global_config(config_path=system.launch.config_path)
                environment["environment_source"] = "effective_serving_launch"
                environment["environment"] = {
                    name: system.launch.environment[name]
                    for name in ENVIRONMENT_KEYS
                    if name in system.launch.environment
                }
                endpoints = {endpoint.model_id: endpoint.base_url for endpoint in system.endpoints}
                environment["local_gpu_inventory"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                uuids = tuple(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
                serving_metadata = capture_owned_metadata(
                    uuids,
                    target_gpu_uuids={target.model_id: uuids for target in case.targets},
                    role_gpu_uuids={
                        "atn": tuple(uuids[index] for index in config.atn.devices),
                        "ffn": tuple(uuids[index] for index in config.ffn.devices),
                    },
                    errors=capture_errors,
                )
                environment["serving_metadata_source"] = "observed"
                environment["serving_metadata"] = serving_metadata.model_dump(mode="json")
                environment["serving_software_source"] = "local_distribution_metadata"
                environment["driver_version_source"] = "nvidia-smi"
                environment["endpoints"] = {str(model_id): url for model_id, url in endpoints.items()}
                environment["commands"] = [list(server.command) for server in system.servers]
                environment["runtime_config"] = str(system.launch.config_path)
                environment["cache_policy"] = {
                    "controller": "owned",
                    "flush_performed": False,
                    "serving_state_reset": True,
                    "warmup_requests_per_target": case.warmup_requests_per_target,
                }
            else:
                endpoints = TypeAdapter(dict[ModelId, str]).validate_json(os.environ.pop("XBENCH_ENDPOINTS"))
            write_environment(directory, environment)
            print("xbench phase=warmup", file=sys.stderr)
            await warmup(case, workload, endpoints, directory, check_alive=system.check_alive if system else None)
            await run_measurement(
                case, workload, endpoints, recorder, check_alive=system.check_alive if system else None
            )
    except asyncio.CancelledError:
        infrastructure_error = "benchmark repetition cancelled"
    except Exception as error:
        infrastructure_error = f"{type(error).__name__}: {error}"
        print(f"xbench repetition failure: {infrastructure_error}", file=sys.stderr)
    finally:
        # Local close cannot certify escaped descendants; keep cleanup unknown.
        if system is not None:
            try:
                print("xbench phase=shutdown", file=sys.stderr)
                system.close()
            except Exception as error:
                infrastructure_error = f"serving cleanup failed: {type(error).__name__}: {error}"
        measured = (directory / "measurement.json").is_file()
        if not recorder.failed:
            for request in workload.requests:
                if request.request_id not in recorder.request_ids:
                    recorder.request(
                        RequestRecord(
                            **request.model_dump(),
                            outcome="failed" if measured else "not_sent",
                            error_kind="evidence_missing" if measured else "before_measurement",
                            error_message="terminal evidence unavailable" if measured else "measurement did not start",
                        )
                    )
        try:
            recorder.close()
        except OSError as error:
            infrastructure_error = f"recording failed: {type(error).__name__}"
        checkpoint = RepetitionManifest.model_validate_json((directory / "repetition.json").read_bytes()).model_copy(
            update={"infrastructure_error": infrastructure_error}
        )
        write_json(directory / "repetition.json", checkpoint.model_dump(mode="json"))
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
    if cancelled_signal is not None:
        raise SystemExit(128 + cancelled_signal)
    if infrastructure_error is not None or recorder.failed:
        raise RuntimeError(infrastructure_error or "benchmark recording failed")
