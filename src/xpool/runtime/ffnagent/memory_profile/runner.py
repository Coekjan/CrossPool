"""Offline FFN memory-sampling participant and process orchestration."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import logging
import multiprocessing
import os
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
import tomli_w
import torch

import xpool.native
from xpool import bootstrap
from xpool.config import XpoolConfig, get_global_config, init_global_config
from xpool.fabric import FabricPlan, FabricRole
from xpool.memory import (
    FfnMemoryCalibration,
    MemoryCalibrationDevice,
    MemoryCalibrationEnvironment,
    XpoolMemoryCalibrationProfile,
    cuda_versions,
)
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.runtime.agent import project_fabric_arena
from xpool.runtime.ffnagent.device_memory import (
    DeviceMemoryEstimator,
    DeviceMemoryPoint,
    ensure_supported_cuda_allocator,
)
from xpool.runtime.ffnagent.memory_profile.corpus import (
    FIT_COORDINATES,
    HELD_OUT_COORDINATE,
    build_fabric_plan,
    calibration_corpus_spec,
    coordinate_members,
    materialize_calibration_weights,
)
from xpool.runtime.ffnagent.memory_profile.fitting import (
    MemoryObservation,
    MemoryProfileParticipantEvidence,
    MemoryProfileSoftwareEnvironment,
    MemoryProfileWorld,
    fit_worlds,
)
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MpsEndpoint, MpsScope
from xpool.utils.sighandler import defer_signal_exceptions, sighandle

CUDA_RUNTIME = importlib.import_module("cuda.bindings.runtime")
SAMPLE_PERIOD_SECONDS = 0.001
CHILD_TIMEOUT_SECONDS = 3600.0
FABRIC_DRAIN_TIMEOUT_SECONDS = 60.0
REPETITION_COUNT = 3
logger = logging.getLogger(__name__)


class PhaseRecorder:
    """High-frequency sampler for the four permanent startup phases."""

    def __init__(self, *, device: int, baseline_free_bytes: int) -> None:
        """Start one current-device sampling thread after a clean baseline."""

        self.device = device
        self.baseline_free_bytes = baseline_free_bytes
        self.current_phase: str | None = None
        self.samples: dict[str, list[int]] = {}
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self.sample_loop, name="ffn-memory-sampler")
        self.thread.start()
        self.ready.wait()
        self.raise_if_failed()

    def sample_loop(self) -> None:
        """Sample global device memory while one named phase is active."""

        try:
            checked_cuda(CUDA_RUNTIME.cudaSetDevice(self.device), "sampler cudaSetDevice")
            self.ready.set()
            deadline_ns = time.monotonic_ns()
            period_ns = int(SAMPLE_PERIOD_SECONDS * 1_000_000_000)
            while not self.stop_requested.is_set():
                deadline_ns += period_ns
                phase = self.current_phase
                if phase is not None:
                    self.samples[phase].append(self.read_sample())
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1_000_000_000)
        except Exception as error:
            self.failure = error
            self.ready.set()

    def read_sample(self) -> int:
        """Read current-device free memory in bytes."""

        free_bytes, _ = memory_info()
        return free_bytes

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one ordinary permanent startup operation."""

        self.begin(name)
        try:
            yield
        finally:
            self.finish(name)

    def begin(self, name: str) -> None:
        """Begin exactly one named phase."""

        if self.current_phase is not None or name in self.samples:
            raise RuntimeError(f"cannot begin duplicate or overlapping memory phase {name!r}")
        self.samples[name] = [self.read_sample()]
        self.current_phase = name

    def finish(self, name: str) -> None:
        """Synchronize and close the active phase."""

        if self.current_phase != name:
            raise RuntimeError(f"cannot finish inactive memory phase {name!r}")
        torch.cuda.synchronize(self.device)
        self.samples[name].append(self.read_sample())
        self.current_phase = None
        self.raise_if_failed()

    def observation(self, point: DeviceMemoryPoint) -> MemoryObservation:
        """Combine one phase peak with its exact feature row."""

        samples = self.samples.get(point.point)
        if not samples:
            raise RuntimeError(f"memory phase {point.point!r} has no samples")
        return MemoryObservation(
            point=point,
            observed_device_bytes=max(0, self.baseline_free_bytes - min(samples)),
        )

    def retained_observation(self, point: DeviceMemoryPoint) -> MemoryObservation:
        """Read the cleaned final retained point."""

        if point.point != "retained":
            raise ValueError("retained observation requires the retained ledger point")
        return MemoryObservation(
            point=point,
            observed_device_bytes=max(0, self.baseline_free_bytes - self.read_sample()),
        )

    def close(self) -> None:
        """Stop and join the sampler."""

        self.stop_requested.set()
        self.thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Propagate sampler failure on the owner thread."""

        if self.failure is not None:
            raise RuntimeError(f"CUDA memory sampler failed: {self.failure}") from self.failure


def checked_cuda(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    """Unwrap one CUDA Runtime binding result."""

    # cuda-python returns operation-specific tuple arities at this binding boundary.
    error, *values = result
    if error != CUDA_RUNTIME.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation}: {error.name}")
    return tuple(values)


def memory_info() -> tuple[int, int]:
    """Read current-device global free and total bytes."""

    free_bytes, total_bytes = checked_cuda(CUDA_RUNTIME.cudaMemGetInfo(), "cudaMemGetInfo")
    return int(free_bytes), int(total_bytes)


@contextlib.contextmanager
def record_registry_phases(recorder: PhaseRecorder) -> Iterator[None]:
    """Split Registry capture and native installation at their existing seam."""

    saved_install = xpool.native.ffnagent.install_execution
    invocation_count = 0

    def install_forwarder(projection: xpool.native.ffnagent.ExecutionProjection) -> None:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count != 1:
            raise RuntimeError("Registry invoked native execution installation more than once")
        recorder.finish("graph_capture")
        recorder.begin("execution_installation")
        saved_install(projection)

    setattr(xpool.native.ffnagent, "install_execution", install_forwarder)
    recorder.begin("graph_capture")
    try:
        yield
        if invocation_count != 1:
            raise RuntimeError("Registry did not invoke native execution installation exactly once")
        torch.cuda.synchronize(recorder.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(recorder.device)
        recorder.finish("execution_installation")
    finally:
        setattr(xpool.native.ffnagent, "install_execution", saved_install)


def write_world_config(path: Path, source: XpoolConfig, coordinate: str) -> None:
    """Write one temporary all-process-equal calibration world config."""

    members = coordinate_members(
        coordinate,
        atnagent_count=len(source.atn.devices),
        ffnagent_count=len(source.ffn.devices),
    )
    payload = {
        "daemon": {"host": source.daemon.host, "port": source.daemon.port},
        "scheduler": {
            "atn_concurrency": 1,
            "ffn_concurrency": source.scheduler.ffn_concurrency,
            "ffn_policy": "fifo",
            "slo": source.scheduler.slo.model_dump(mode="python"),
        },
        "atn": {"devices": source.atn.devices},
        "ffn": {
            "devices": source.ffn.devices,
            "device_memory_extra_margin_bytes": 0,
            "loader": {"parallelism": source.ffn.loader.parallelism},
            "placement": {
                "parallelism": source.ffn.placement.parallelism,
                "timeout_seconds": source.ffn.placement.timeout_seconds,
            },
        },
        "models": [
            {
                "id": f"calibration/{member}",
                "path": str(path.parent / f"calibration-member-{index}"),
                "ffn_tp_size": tp_size,
            }
            for index, (member, tp_size, _, _) in enumerate(members)
        ],
    }
    XpoolConfig.from_mapping(payload)
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def software_environment() -> MemoryProfileSoftwareEnvironment:
    """Collect exact software compatibility facts in one participant."""

    driver_version, runtime_version = cuda_versions()
    return MemoryProfileSoftwareEnvironment(
        native_abi_version=int(xpool.native.ABI_VERSION),
        cuda_driver_version=driver_version,
        cuda_runtime_version=runtime_version,
        torch_version=importlib.metadata.version("torch"),
        triton_version=importlib.metadata.version("triton"),
        sglang_version=importlib.metadata.version("sglang"),
        sglang_kernel_version=importlib.metadata.version("sglang-kernel"),
        nvshmem_version=importlib.metadata.version("nvidia-nvshmem-cu13"),
    )


def device_record(device: int, total_memory_bytes: int) -> MemoryCalibrationDevice:
    """Read one selected device's Profile identity and provenance UUID."""

    properties = torch.cuda.get_device_properties(device)
    uuid = str(properties.uuid).strip()
    if not uuid:
        raise RuntimeError("PyTorch returned an empty device UUID")
    if properties.total_memory != total_memory_bytes:
        raise RuntimeError("PyTorch and cudaMemGetInfo disagree on total device memory")
    return MemoryCalibrationDevice(
        name=properties.name,
        compute_capability=(properties.major, properties.minor),
        total_memory_bytes=properties.total_memory,
        uuid=uuid,
    )


def drain_and_finalize() -> None:
    """Collectively drain and release one participant runtime."""

    xpool.native.fabric.drain_async()
    deadline = time.monotonic() + FABRIC_DRAIN_TIMEOUT_SECONDS
    while xpool.native.fabric.drain_pending():
        if time.monotonic() >= deadline:
            raise RuntimeError("memory-profile Fabric drain timed out")
        time.sleep(0.01)
    xpool.native.fabric.finalize()


def require_command(connection: Connection, expected: str) -> None:
    """Read one exact Host-controller command."""

    command = connection.recv()
    if command != expected:
        raise RuntimeError(f"memory-profile participant expected {expected!r}, received {command!r}")


def uid_child(connection: Connection, config_path: Path) -> None:
    """Own one Fabric UID bootstrap socket for a complete profiling world."""

    try:
        init_global_config(config_path=config_path)
        bootstrap.init(None, RuntimeRole.DAEMON)
        connection.send(("uid", xpool.native.fabric.create_uid()))
        require_command(connection, "stop")
        connection.send(("stopped", None))
    except Exception as error:
        connection.send(("failure", f"{type(error).__name__}: {error}"))
        raise
    finally:
        connection.close()


def participant_child(
    connection: Connection,
    *,
    config_path: Path,
    fabric_plan: FabricPlan,
    pe: int,
    environment: dict[str, str],
) -> None:
    """Run one fresh native participant through execution installation."""

    try:
        os.environ.update(environment)
        config = init_global_config(config_path=config_path)
        placement = fabric_plan.pe_placements[pe]
        role = RuntimeRole.ATNAGENT if placement.role is FabricRole.ATNAGENT else RuntimeRole.FFNAGENT
        device = placement.device
        if role is RuntimeRole.FFNAGENT:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
        bootstrap.init(device, role)
        if role is RuntimeRole.ATNAGENT:
            visibility = visible_uuids()
            MpsEndpoint(tuple(visibility[index] for index in config.atn.devices)).require_client()
            connection.send(("prepared", pe))
            command = connection.recv()
            if command == "stop":
                connection.send(("stopped", pe))
                return
            if command != "join":
                raise RuntimeError(f"memory-profile participant expected join, received {command!r}")
            xpool.native.fabric.join(project_fabric_arena(fabric_plan), pe)
            connection.send(("joined", pe))
            require_command(connection, "finalize")
            drain_and_finalize()
            connection.send(("stopped", pe))
            return

        ensure_supported_cuda_allocator()
        model_specs = tuple(calibration_corpus_spec(model.id.name) for model in config.models)
        warmup = torch.empty(1, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize(device)
        del warmup
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_free_bytes, baseline_total_bytes = memory_info()
        recorder = PhaseRecorder(device=device, baseline_free_bytes=baseline_free_bytes)
        estimator = DeviceMemoryEstimator(
            model_specs=model_specs,
            instance_profiles=tuple(instance.ffn_profile for instance in fabric_plan.instance_plans),
        )
        atnagent_count = sum(item.role is FabricRole.ATNAGENT for item in fabric_plan.pe_placements)
        ffnagent_index = pe - atnagent_count
        ledger = estimator.allocation_ledger(fabric_plan=fabric_plan, ffnagent_index=ffnagent_index)
        try:
            with recorder.phase("weight_materialization"):
                layer_weights = materialize_calibration_weights(
                    fabric_plan=fabric_plan,
                    model_specs=model_specs,
                    ffnagent_index=ffnagent_index,
                )
            connection.send(("prepared", pe))
            command = connection.recv()
            if command == "stop":
                connection.send(("stopped", pe))
                return
            if command != "join":
                raise RuntimeError(f"memory-profile participant expected join, received {command!r}")
            with recorder.phase("fabric_join"):
                xpool.native.fabric.join(project_fabric_arena(fabric_plan), pe)
            connection.send(("joined", pe))
            command = connection.recv()
            if command == "finalize":
                drain_and_finalize()
                connection.send(("stopped", pe))
                return
            if command != "execute":
                raise RuntimeError(f"memory-profile participant expected execute, received {command!r}")
            with record_registry_phases(recorder):
                execution_registry = FfnExecutionRegistry.materialize(
                    fabric_plan=fabric_plan,
                    model_specs=model_specs,
                    ffnagent_index=ffnagent_index,
                    layer_weights=layer_weights,
                )
            recorder.close()
            observations = (
                *(recorder.observation(point) for point in ledger[:-1]),
                recorder.retained_observation(ledger[-1]),
            )
            connection.send(
                (
                    "execution",
                    MemoryProfileParticipantEvidence(
                        ffnagent_index=ffnagent_index,
                        device=device_record(device, baseline_total_bytes),
                        environment=software_environment(),
                        observations=observations,
                    ),
                )
            )
            require_command(connection, "finalize")
            drain_and_finalize()
            del execution_registry
            del layer_weights
            connection.send(("stopped", pe))
        finally:
            if recorder.thread.is_alive():
                recorder.close()
    except Exception as error:
        try:
            connection.send(("failure", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def run_world(coordinate: str, source: XpoolConfig) -> MemoryProfileWorld:
    """Own a calibration world with complete deployment visibility and ordered retirement.

    Signals record the first cancellation budget. The current preparation,
    collective join or execution transaction reaches its existing boundary;
    cancellation prevents the next transaction. Unknown peers retain this live
    owner rather than authorizing device-blind process termination.
    """

    context = multiprocessing.get_context("spawn")
    visibility = visible_uuids()
    if max(source.devices) >= len(visibility):
        raise ValueError("memory-profile placement exceeds original device visibility")
    scope = MpsScope(MpsEndpoint(tuple(visibility[device] for device in source.atn.devices)))
    attention_environment = scope.endpoint.environment()
    attention_environment["CUDA_VISIBLE_DEVICES"] = ",".join(visibility)
    ffn_environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(visibility),
        "CUDA_MPS_PIPE_DIRECTORY": "",
    }
    cancelled: int | None = None
    cleanup_deadline: float | None = None

    def record_cancellation(signum: int, frame: FrameType | None) -> None:
        nonlocal cancelled, cleanup_deadline
        if cancelled is None:
            cancelled = signum
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S

    def receive(connection: Connection, process: BaseProcess, expected: str) -> object:
        execution_deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
        while True:
            deadline = execution_deadline if cleanup_deadline is None else min(execution_deadline, cleanup_deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"memory-profile child {process.name} timed out waiting for {expected}")
            ready = wait((connection, process.sentinel), timeout=min(0.1, remaining))
            if connection in ready:
                break
            if process.sentinel in ready:
                raise RuntimeError(f"memory-profile child {process.name} exited with code {process.exitcode}")
        try:
            message = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"memory-profile child {process.name} closed before {expected}") from error
        if not isinstance(message, tuple) or len(message) != 2:
            raise RuntimeError(f"memory-profile child {process.name} returned an invalid message")
        tag, payload = message
        if tag == "failure":
            raise RuntimeError(f"memory-profile child {process.name} failed: {payload}")
        if tag != expected:
            raise RuntimeError(f"memory-profile child {process.name} returned {tag!r}, expected {expected!r}")
        return payload

    with contextlib.ExitStack() as handlers, tempfile.TemporaryDirectory(prefix="xpool-memory-profile-") as temporary:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers.enter_context(sighandle(signum, record_cancellation))
        config_path = Path(temporary) / "world.toml"
        write_world_config(config_path, source, coordinate)
        world_config = XpoolConfig.from_file(config_path)
        members = coordinate_members(
            coordinate,
            atnagent_count=len(world_config.atn.devices),
            ffnagent_count=len(world_config.ffn.devices),
        )
        model_specs = tuple(calibration_corpus_spec(member[0]) for member in members)
        uid_connection, uid_child_connection = context.Pipe()
        uid_process = context.Process(
            target=uid_child, args=(uid_child_connection, config_path), name="memory-profile-uid"
        )
        processes: list[BaseProcess] = []
        connections: list[Connection] = []
        stopped = False
        uid_ready = False
        try:
            with defer_signal_exceptions():
                uid_process.start()
                uid_child_connection.close()
            uid = receive(uid_connection, uid_process, "uid")
            if not isinstance(uid, str):
                raise RuntimeError("memory-profile UID child returned a non-string UID")
            uid_ready = True
            fabric_plan = build_fabric_plan(
                uid=uid, coordinate=coordinate, config=world_config, model_specs=model_specs
            )
            scope.start()
            for pe, placement in enumerate(fabric_plan.pe_placements):
                with defer_signal_exceptions():
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=participant_child,
                        kwargs={
                            "connection": child,
                            "config_path": config_path,
                            "fabric_plan": fabric_plan,
                            "pe": pe,
                            "environment": attention_environment
                            if placement.role is FabricRole.ATNAGENT
                            else ffn_environment,
                        },
                        name=f"memory-profile-{coordinate}-pe{pe}",
                    )
                    processes.append(process)
                    connections.append(parent)
                    process.start()
                    child.close()

            # Each broadcast completes one existing transaction. Signals record
            # intent instead of leaving only part of the collective world joined.
            for connection, process in zip(connections, processes, strict=True):
                receive(connection, process, "prepared")
            joined = cancelled is None
            rows = []
            if joined:
                for connection in connections:
                    connection.send("join")
                for connection, process in zip(connections, processes, strict=True):
                    receive(connection, process, "joined")
                if cancelled is None:
                    atnagent_count = len(world_config.atn.devices)
                    for connection in connections[atnagent_count:]:
                        connection.send("execute")
                    for connection, process in zip(
                        connections[atnagent_count:], processes[atnagent_count:], strict=True
                    ):
                        row = receive(connection, process, "execution")
                        if not isinstance(row, MemoryProfileParticipantEvidence):
                            raise RuntimeError("memory-profile participant returned invalid evidence")
                        rows.append(row)
            if cleanup_deadline is None:
                cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            for connection in connections:
                connection.send("finalize" if joined else "stop")
            for connection, process in zip(connections, processes, strict=True):
                receive(connection, process, "stopped")
            for process in processes:
                process.join(max(0.0, cleanup_deadline - time.monotonic()))
                if process.is_alive():
                    raise TimeoutError(f"memory-profile child {process.name} retirement is unconfirmed")
                if process.exitcode != 0:
                    raise RuntimeError(f"memory-profile child {process.name} exited with code {process.exitcode}")
            stopped = True
            if cancelled is not None:
                raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
            ordered = tuple(sorted(rows, key=lambda row: row.ffnagent_index))
            if tuple(row.ffnagent_index for row in ordered) != tuple(range(len(world_config.ffn.devices))):
                raise RuntimeError("memory-profile evidence does not cover every configured FfnAgent")
            environment = ordered[0].environment
            if any(row.environment != environment for row in ordered[1:]):
                raise RuntimeError("memory-profile participants disagree on the software environment")
            return MemoryProfileWorld(
                devices=tuple(row.device for row in ordered),
                environment=environment,
                observations=tuple(observation for row in ordered for observation in row.observations),
            )
        finally:
            if cleanup_deadline is None:
                cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            try:
                if not stopped and any(process.is_alive() for process in processes):
                    raise RuntimeError("memory-profile participant retirement is unconfirmed")
                # The host-only UID owner remains until all device participants
                # have retired. It owns no device context or collective teardown.
                if uid_process.pid is not None and uid_process.is_alive():
                    if uid_ready:
                        uid_connection.send("stop")
                        receive(uid_connection, uid_process, "stopped")
                    else:
                        uid_process.terminate()
                    uid_process.join(max(0.0, cleanup_deadline - time.monotonic()))
                if uid_process.is_alive():
                    raise TimeoutError("memory-profile UID owner retirement is unconfirmed")
                for connection in (uid_connection, uid_child_connection, *connections):
                    connection.close()
                for process in (uid_process, *processes):
                    process.close()
                scope.stop(deadline=cleanup_deadline)
            except BaseException as error:
                logger.error("memory-profile cleanup unconfirmed; retaining owner and MPS: %s", error)
                while True:
                    time.sleep(1.0)


def refuse_live_daemon(config: XpoolConfig) -> None:
    """Reject profiling while the configured daemon endpoint responds."""

    host = f"[{config.daemon.host}]" if ":" in config.daemon.host else config.daemon.host
    try:
        httpx.get(f"http://{host}:{config.daemon.port}/health", timeout=1.0)
    except httpx.HTTPError:
        return
    raise RuntimeError("xpool memory-profile refuses to run while the configured daemon endpoint responds")


def profile_ffn_memory() -> XpoolMemoryCalibrationProfile:
    """Run the fixed fresh-process matrix and return one qualified Profile."""

    config = get_global_config()
    if config.ffn.device_memory_calibration is None:
        raise RuntimeError("ffn.device_memory_calibration is required for xpool memory-profile")
    refuse_live_daemon(config)

    # Repeated fit worlds determine coefficients; the disjoint held-out world
    # determines the minimum headroom required for an unseen placement shape.
    fit_coordinates = FIT_COORDINATES if len(config.ffn.devices) >= 2 else FIT_COORDINATES[:-1]
    fit_worlds_evidence = tuple(
        run_world(coordinate, config) for coordinate in fit_coordinates for _ in range(REPETITION_COUNT)
    )
    held_out_worlds = tuple(run_world(HELD_OUT_COORDINATE, config) for _ in range(REPETITION_COUNT))
    all_worlds = (*fit_worlds_evidence, *held_out_worlds)
    devices = all_worlds[0].devices
    environment = all_worlds[0].environment
    if any(world.devices != devices or world.environment != environment for world in all_worlds[1:]):
        raise RuntimeError("xpool memory-profile worlds disagree on hardware or software environment")
    coefficients, minimum_headroom = fit_worlds(fit_worlds_evidence, held_out_worlds)
    if environment.native_abi_version != ABI_VERSION:
        raise RuntimeError("xpool memory-profile child native ABI disagrees with Python")
    return XpoolMemoryCalibrationProfile(
        environment=MemoryCalibrationEnvironment(
            native_abi_version=ABI_VERSION,
            ffnagent_devices=devices,
            cuda_driver_version=environment.cuda_driver_version,
            cuda_runtime_version=environment.cuda_runtime_version,
            torch_version=environment.torch_version,
            triton_version=environment.triton_version,
            sglang_version=environment.sglang_version,
            sglang_kernel_version=environment.sglang_kernel_version,
            nvshmem_version=environment.nvshmem_version,
        ),
        ffn=FfnMemoryCalibration(
            atnagent_count=len(config.atn.devices),
            executor_lane_count=config.scheduler.ffn_concurrency,
            minimum_held_out_headroom_bytes=minimum_headroom,
            coefficients=coefficients,
        ),
    )
