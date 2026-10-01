"""Installed daemon, Agent, and Instance composition for FFN qualification."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import safetensors.torch
import torch

from xkit.child import PythonChildProcess
from xkit.network import TcpEndpointReservation
from xkit.serving.cluster import XpoolCluster, XpoolClusterLaunch
from xkit.serving.launch import snapshot_cluster_launch
from xpool import bootstrap, devkit
from xpool.config import XpoolConfig, init_global_config
from xpool.model import ModelId
from xpool.native import RuntimeRole
from xpool.ops import ffn_shim
from xpool.runtime.instance import InstanceRankRuntime
from xpool.service.wire import ServingListener
from xpool.transport import FfnRequestMetadata
from xtest.harness.native.ffn.protocol import (
    FfnInstanceClosed,
    FfnInstanceCommand,
    FfnInstanceCompleted,
    FfnInstanceReady,
    FfnInstanceSpec,
)
from xtest.harness.native.mps import MpsServerObservation, query_mps_servers
from xtest.harness.support.kv import kv_capacity_profile
from xtest.harness.support.wait import remaining_seconds


@dataclass(frozen=True, slots=True)
class FfnProcessObservation:
    """One live supervised CUDA client and its configured CUDA device."""

    name: str
    process_id: int
    cuda_device: int


@dataclass(frozen=True, slots=True)
class FfnLiveTopologyObservation:
    """Live process and MPS membership captured before topology shutdown."""

    processes: tuple[FfnProcessObservation, ...]
    mps_servers: tuple[MpsServerObservation, ...]


def materialize_ffn_cluster_launch(
    *,
    config: XpoolConfig,
    daemon_port: int,
    workdir: Path,
) -> XpoolClusterLaunch:
    """Materialize one task-local config for an installed FFN process tree."""

    workdir.mkdir(parents=True, exist_ok=False)
    observer_outdir = (workdir / "observers").resolve()
    observer_outdir.mkdir()
    config_path = (workdir / "xpool.toml").resolve()
    environment = ffn_cluster_environment(config_path=config_path, observer_outdir=observer_outdir)
    return snapshot_cluster_launch(
        config,
        workdir=workdir,
        daemon_port=daemon_port,
        environment=environment,
        cwd=Path.cwd(),
    )


def ffn_cluster_environment(*, config_path: Path, observer_outdir: Path) -> dict[str, str]:
    """Return the sanitized environment shared by every qualification entity."""

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
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "XPOOL_CONFIG": str(config_path),
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_FFN_ROUTING_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FFN_ROUTING_OBSERVER_OUTDIR": str(observer_outdir),
            "XPOOL_DEBUG_FFN_ROUTING_OBSERVER_RECORD_CAPACITY": "64",
        }
    )
    return environment


def run_ffn_topology(
    *,
    launch: XpoolClusterLaunch,
    endpoint: TcpEndpointReservation,
    instance_specs: tuple[FfnInstanceSpec, ...],
    workdir: Path,
    timeout_seconds: float,
    observation_sink: Callable[[FfnLiveTopologyObservation], None] | None = None,
) -> tuple[FfnInstanceReady, ...]:
    """Run one installed daemon, Agent, and controlled Instance process tree."""

    if not instance_specs or timeout_seconds <= 0:
        raise ValueError("FFN topology requires Instance specs and a positive timeout")
    deadline = time.monotonic() + timeout_seconds
    cluster: XpoolCluster | None = None
    children: list[PythonChildProcess] = []
    try:
        cluster = XpoolCluster.start(launch, endpoint)
        instance_ordinals: dict[ModelId, int] = {}
        for spec in instance_specs:
            instance_ordinal = instance_ordinals.setdefault(spec.model_id, len(instance_ordinals))
            child_name = f"instance-{instance_ordinal}-rank-{spec.rank}"
            children.append(
                PythonChildProcess.start(
                    child_name,
                    run_ffn_instance,
                    spec,
                    log_path=workdir / f"{child_name}.log",
                )
            )
        ready = tuple(
            child.receive(
                FfnInstanceReady,
                timeout_seconds=remaining_seconds(deadline, "FFN qualification Instance readiness"),
            )
            for child in children
        )
        if len({entry.generation for entry in ready}) != 1:
            raise RuntimeError("FFN qualification Instance ranks observed different Fabric generations")
        for child in children:
            child.send(FfnInstanceCommand.RUN)
        for child, spec in zip(children, instance_specs, strict=True):
            completed = child.receive(
                FfnInstanceCompleted,
                timeout_seconds=remaining_seconds(deadline, "FFN qualification Instance completion"),
            )
            expected = tuple(invocation.case_id for invocation in spec.invocations)
            if completed.case_ids != expected:
                raise RuntimeError(f"FFN Instance completed {completed.case_ids}, expected {expected}")
        if observation_sink is not None:
            observation_sink(observe_live_topology(cluster, children, instance_specs))
        for child in children:
            child.send(FfnInstanceCommand.CLOSE)
        for child in children:
            child.receive(
                FfnInstanceClosed,
                timeout_seconds=remaining_seconds(deadline, "FFN qualification Instance close"),
            )
            child.wait(timeout_seconds=remaining_seconds(deadline, "FFN qualification Instance child exit"))
        cluster.close()
        cluster = None
        return ready
    finally:
        PythonChildProcess.terminate_all(tuple(children))
        for child in children:
            if not child.process.is_alive():
                child.close()
        if cluster is not None:
            cluster.close()
        endpoint.close()


def observe_live_topology(
    cluster: XpoolCluster,
    children: list[PythonChildProcess],
    instance_specs: tuple[FfnInstanceSpec, ...],
) -> FfnLiveTopologyObservation:
    """Observe supervised CUDA clients and their actual MPS membership."""

    configured_devices = {
        **{f"atnagent-{agent.cuda_device}": agent.cuda_device for agent in cluster.launch.config.atnagents},
        **{f"ffnagent-{agent.cuda_device}": agent.cuda_device for agent in cluster.launch.config.ffnagents},
    }
    processes = []
    for process in cluster.processes:
        if process.name == "daemon":
            continue
        try:
            cuda_device = configured_devices[process.name]
        except KeyError as error:
            raise RuntimeError(f"FFN topology observed unknown Agent process {process.name!r}") from error
        processes.append(FfnProcessObservation(process.name, process.process.pid, cuda_device))
    for child, spec in zip(children, instance_specs, strict=True):
        process_id = child.process.pid
        if process_id is None:
            raise RuntimeError(f"FFN topology Instance process {child.name!r} has no process ID")
        processes.append(
            FfnProcessObservation(
                child.name,
                process_id,
                cluster.launch.config.atn.devices[spec.rank],
            )
        )
    supervised_ids = {process.process_id for process in processes}
    servers = tuple(
        MpsServerObservation(
            process_id=server.process_id,
            client_process_ids=tuple(
                process_id for process_id in server.client_process_ids if process_id in supervised_ids
            ),
            active_thread_percentage=server.active_thread_percentage,
        )
        for server in query_mps_servers()
        if any(process_id in supervised_ids for process_id in server.client_process_ids)
    )
    return FfnLiveTopologyObservation(tuple(processes), servers)


def run_ffn_instance(connection: Connection, spec: FfnInstanceSpec) -> None:
    """Attach one production Instance rank and execute controlled tensors."""

    os.environ.clear()
    os.environ.update(spec.environment)
    config = init_global_config()
    cuda_device = config.atn.devices[spec.rank]
    bootstrap.init(cuda_device, RuntimeRole.INSTANCE)
    devkit.install()
    runtime = InstanceRankRuntime.start(
        model_id=spec.model_id,
        rank=spec.rank,
        transport=spec.transport,
        ffn_profile=spec.ffn_profile,
        kv_capacity=kv_capacity_profile(),
        atn_runtime_headroom_bytes=0,
    )
    try:
        plan = runtime.wait_for_fabric_executable()
        model_plan = plan.model_plans[runtime.instance_index]
        if model_plan.tp_size != spec.ffn_tp_size:
            raise RuntimeError(
                f"production FFN TP size {model_plan.tp_size} does not match qualification {spec.ffn_tp_size}"
            )
        runtime.attach_arena_from_daemon()
        runtime.start_failure_monitor()
        runtime.publish_initialized(ServingListener(host="127.0.0.1", port=1))
        runtime.wait_for_ready()
        connection.send(
            FfnInstanceReady(
                plan.generation,
                runtime.instance_index,
                tuple(layer.ffnagent_indices for layer in model_plan.layers),
            )
        )
        command = connection.recv()
        if command is not FfnInstanceCommand.RUN:
            raise RuntimeError(f"FFN Instance expected RUN, received {command!r}")
        for invocation in spec.invocations:
            tensors = safetensors.torch.load_file(invocation.input_path, device="cpu")
            if set(tensors) != {"hidden_states"}:
                raise RuntimeError(f"FFN production input has invalid tensor keys: {sorted(tensors)}")
            hidden_states = tensors["hidden_states"].to(device=cuda_device)
            dp_rank_payload_rows = (
                None
                if invocation.dp_rank_payload_rows is None
                else torch.tensor(invocation.dp_rank_payload_rows, dtype=torch.int64, device=cuda_device)
            )
            output = ffn_shim(
                hidden_states,
                dp_rank_payload_rows,
                FfnRequestMetadata(
                    layer_ordinal=invocation.layer_ordinal,
                    forward_mode=invocation.forward_mode,
                    output_requirement=invocation.output_requirement,
                    dp_row_layout=invocation.dp_row_layout,
                ),
            )
            output = output.detach().to(device="cpu").contiguous()
            with invocation.output_path.open("xb") as output_file:
                output_file.write(safetensors.torch.save({"hidden_states": output}))
        connection.send(FfnInstanceCompleted(tuple(invocation.case_id for invocation in spec.invocations)))
        command = connection.recv()
        if command is not FfnInstanceCommand.CLOSE:
            raise RuntimeError(f"FFN Instance expected CLOSE, received {command!r}")
    finally:
        runtime.close()
    connection.send(FfnInstanceClosed())
