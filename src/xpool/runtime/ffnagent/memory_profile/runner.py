"""Offline FFN memory-sampling participant and process orchestration."""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import multiprocessing
import os
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import FrameType

import httpx
import tomli_w
import torch
from cuda.bindings import runtime  # ty: ignore[unresolved-import]

import xpool.native
from xpool import bootstrap, ffn
from xpool.config import (
    CONFIG_REGISTRY,
    XpoolConfig,
    get_global_config,
    init_global_config,
)
from xpool.fabric import FabricPlan, FabricRole
from xpool.memory import (
    FfnMemoryCalibration,
    MemoryCalibrationDevice,
    MemoryCalibrationEnvironment,
    XpoolMemoryCalibrationProfile,
)
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.runtime.agent import project_fabric_arena
from xpool.runtime.ffnagent.device_memory import (
    DeviceMemoryEstimator,
    DeviceMemoryPoint,
    ensure_supported_cuda_allocator,
)
from xpool.runtime.ffnagent.memory_profile.corpus import (
    calibration_experiments,
    materialize_calibration_weights,
)
from xpool.runtime.ffnagent.memory_profile.evidence import (
    MemoryObservation,
    MemoryProfileEvidence,
    MemoryProfileParticipantEvidence,
    MemoryProfileSoftwareEnvironment,
    MemoryProfileWorld,
)
from xpool.runtime.ffnagent.memory_profile.experiment import CalibrationExperiment
from xpool.runtime.ffnagent.memory_profile.fitting import fit_worlds
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.utils.config import ConfigSource, get_nested
from xpool.utils.cuda import check_runtime, cuda_versions, is_terminal_device_error
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MpsEndpoint, MpsScope
from xpool.utils.sighandler import block_signals, initialize_child_signals, sighandle

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
        self.minimum_free_bytes: dict[str, int] = {}
        self.lock = threading.Lock()
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
            (status,) = runtime.cudaSetDevice(self.device)
            check_runtime(status, "sampler cudaSetDevice")
            self.ready.set()
            deadline_ns = time.monotonic_ns()
            period_ns = int(SAMPLE_PERIOD_SECONDS * 1_000_000_000)
            while not self.stop_requested.is_set():
                deadline_ns += period_ns
                with self.lock:
                    phase = self.current_phase
                    if phase is not None:
                        self.minimum_free_bytes[phase] = min(self.minimum_free_bytes[phase], memory_info()[0])
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1_000_000_000)
        except Exception as error:
            self.failure = error
            self.ready.set()

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one ordinary permanent startup operation."""

        self.begin(name)
        try:
            yield
        except BaseException:
            self.stop_requested.set()
            raise
        else:
            self.finish(name)

    def begin(self, name: str) -> None:
        """Begin exactly one named phase."""

        with self.lock:
            self.raise_if_failed()
            if self.current_phase is not None or name in self.minimum_free_bytes:
                raise RuntimeError(f"cannot begin duplicate or overlapping memory phase {name!r}")
            self.minimum_free_bytes[name] = memory_info()[0]
            self.current_phase = name

    def finish(self, name: str) -> None:
        """Synchronize and close the active phase."""

        if self.current_phase != name:
            raise RuntimeError(f"cannot finish inactive memory phase {name!r}")
        self.raise_if_failed()
        torch.cuda.synchronize(self.device)
        with self.lock:
            self.raise_if_failed()
            self.minimum_free_bytes[name] = min(self.minimum_free_bytes[name], memory_info()[0])
            self.current_phase = None
        self.raise_if_failed()

    def observation(self, point: DeviceMemoryPoint) -> MemoryObservation:
        """Combine one phase peak with its exact feature row."""

        with self.lock:
            minimum = self.minimum_free_bytes.get(point.point)
        if minimum is None:
            raise RuntimeError(f"memory phase {point.point!r} has no samples")
        return MemoryObservation(
            point=point,
            observed_device_bytes=max(0, self.baseline_free_bytes - minimum),
        )

    def retained_observation(self, point: DeviceMemoryPoint) -> MemoryObservation:
        """Read the cleaned retained point after successful sampler shutdown."""

        if point.point != "retained":
            raise ValueError("retained observation requires the retained ledger point")
        return MemoryObservation(
            point=point,
            observed_device_bytes=max(0, self.baseline_free_bytes - memory_info()[0]),
        )

    def close(self) -> None:
        """Stop and join the sampler."""

        self.stop_requested.set()
        if self.failure is not None and is_terminal_device_error(self.failure):
            self.raise_if_failed()
        self.thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Propagate sampler failure on the owner thread."""

        if self.failure is not None:
            raise RuntimeError(f"CUDA memory sampler failed: {self.failure}") from self.failure


def memory_info() -> tuple[int, int]:
    """Read current-device global free and total bytes."""

    status, free_bytes, total_bytes = runtime.cudaMemGetInfo()
    check_runtime(status, "cudaMemGetInfo")
    return int(free_bytes), int(total_bytes)


@contextlib.contextmanager
def record_registry_phases(recorder: PhaseRecorder) -> Iterator[None]:
    """Split Registry capture and native installation at their existing seam."""

    saved_install = xpool.native.ffnagent.install_execution
    invocation_count = 0

    def install_forwarder(projection: xpool.native.ffnagent.ExecutionInstallDescriptor) -> None:
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
        recorder.raise_if_failed()
        torch.cuda.synchronize(recorder.device)
        recorder.raise_if_failed()
        torch.cuda.empty_cache()
        recorder.raise_if_failed()
        torch.cuda.synchronize(recorder.device)
        recorder.finish("execution_installation")
    finally:
        setattr(xpool.native.ffnagent, "install_execution", saved_install)


def write_world_config(path: Path, source: XpoolConfig, experiment: CalibrationExperiment) -> dict[str, str]:
    """Write the experiment snapshot and return its resolved env-only inputs."""

    snapshot = experiment.config_snapshot(source, directory=path.parent)
    values = source.model_dump(mode="python")
    config_environment = {
        setting.env_var: str(int(value)) if isinstance(value, bool) else str(value)
        for setting in CONFIG_REGISTRY
        if setting.path is not None
        and setting.env_var is not None
        and ConfigSource.CONFIG not in setting.allowed_sources
        and ConfigSource.ENV in setting.allowed_sources
        and (value := get_nested(values, setting.path)[1]) is not None
    }
    payload = snapshot.to_config_mapping()
    XpoolConfig.from_mapping(payload, env=config_environment)
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return config_environment


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


@dataclass(frozen=True, slots=True)
class ParticipantFailure:
    """A failed outcome and permission to await coordinated finalization."""

    message: str
    ready_to_finalize: bool


def uid_child(
    connection: Connection,
    config_path: Path,
    *,
    config_environment: dict[str, str],
    signal_mask: set[int],
) -> None:
    """Own one Host bootstrap socket until all device participants retire."""

    initialize_child_signals(signal_mask)
    try:
        init_global_config(config_path=config_path, env=config_environment)
        bootstrap.init(None, RuntimeRole.DAEMON)
        connection.send(("uid", xpool.native.fabric.create_uid()))
        require_command(connection, "stop")
        connection.send(("stopped", None))
    except Exception as error:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("failure", ParticipantFailure(f"{type(error).__name__}: {error}", False)))
        raise
    finally:
        connection.close()


def participant_child(
    connection: Connection,
    *,
    config_path: Path,
    config_environment: dict[str, str],
    model_specs: tuple[ffn.FfnModelSpec, ...],
    fabric_plan: FabricPlan,
    pe: int,
    environment: dict[str, str],
    signal_mask: set[int],
) -> None:
    """Measure one fresh participant and obey parent-coordinated retirement."""

    initialize_child_signals(signal_mask)
    recorder: PhaseRecorder | None = None
    failure: Exception | None = None
    join_started = False
    joined = False
    finalize_started = False
    finalized = False
    execution_registry: FfnExecutionRegistry | None = None

    def fail(error: Exception) -> None:
        nonlocal failure
        if failure is None:
            failure = error
        if is_terminal_device_error(error):
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("failure", ParticipantFailure(f"{type(failure).__name__}: {failure}", False)))
            logger.error("memory-profile terminal device failure pe=%s: %s", pe, error, exc_info=error)
            os._exit(1)

    try:
        os.environ.update(environment)
        config = init_global_config(config_path=config_path, env=config_environment)
        placement = fabric_plan.pe_placements[pe]
        role = RuntimeRole.ATNAGENT if placement.role is FabricRole.ATNAGENT else RuntimeRole.FFNAGENT
        device = placement.device
        if role is RuntimeRole.FFNAGENT:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
        bootstrap.init(device, role)
        if role is RuntimeRole.ATNAGENT:
            visibility = visible_uuids()
            MpsEndpoint(tuple(visibility[index] for index in config.atn.devices)).require_client()
        else:
            ensure_supported_cuda_allocator()
            warmup = torch.empty(1, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize(device)
            del warmup
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            baseline_free_bytes, baseline_total_bytes = memory_info()
            estimator = DeviceMemoryEstimator(
                model_specs=model_specs,
                instance_profiles=tuple(instance.ffn_profile for instance in fabric_plan.instance_plans),
            )
            atnagent_count = sum(item.role is FabricRole.ATNAGENT for item in fabric_plan.pe_placements)
            ffnagent_index = pe - atnagent_count
            ledger = estimator.allocation_ledger(fabric_plan=fabric_plan, ffnagent_index=ffnagent_index)
            recorder = PhaseRecorder(device=device, baseline_free_bytes=baseline_free_bytes)
            with recorder.phase("weight_materialization"):
                layer_weights = materialize_calibration_weights(
                    fabric_plan=fabric_plan,
                    model_specs=model_specs,
                    ffnagent_index=ffnagent_index,
                )
        connection.send(("prepared", pe))
        command = connection.recv()
        if command == "stop":
            if recorder is not None:
                recorder.close()
            connection.send(("stopped", pe))
            return
        if command != "join":
            raise RuntimeError(f"memory-profile participant expected join, received {command!r}")
        arena = project_fabric_arena(fabric_plan)
        with recorder.phase("fabric_join") if recorder is not None else contextlib.nullcontext():
            join_started = True
            xpool.native.fabric.join(arena, pe)
            joined = True
        connection.send(("joined", pe))
        if role is RuntimeRole.FFNAGENT:
            command = connection.recv()
            if command == "execute":
                if recorder is None:
                    raise AssertionError("FFN participant lacks its sampler")
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
            elif command != "finalize":
                raise RuntimeError(f"memory-profile participant expected execute or finalize, received {command!r}")
        else:
            require_command(connection, "finalize")
        if recorder is not None:
            recorder.close()
        finalize_started = True
        drain_and_finalize()
        finalized = True
        connection.send(("stopped", pe))
    except Exception as error:
        fail(error)
        if recorder is not None:
            if recorder.failure is not None:
                fail(recorder.failure)
            try:
                recorder.close()
            except Exception as sampler_error:
                fail(sampler_error)
        ready = joined and not finalize_started and (recorder is None or not recorder.thread.is_alive())
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("failure", ParticipantFailure(f"{type(failure).__name__}: {failure}", ready)))
        if ready:
            try:
                require_command(connection, "finalize")
                finalize_started = True
                drain_and_finalize()
                finalized = True
                connection.send(("stopped", pe))
            except Exception as retirement_error:
                fail(retirement_error)
                with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                    connection.send(("failure", ParticipantFailure(str(retirement_error), False)))
                logger.error(
                    "memory-profile participant retirement unconfirmed pe=%s: %s",
                    pe,
                    retirement_error,
                    exc_info=retirement_error,
                )
                while True:
                    time.sleep(1.0)
        if join_started and not finalized:
            logger.error(
                "memory-profile participant retaining unconfirmed collective pe=%s: %s", pe, failure, exc_info=failure
            )
            while True:
                time.sleep(1.0)
        if failure is not None:
            raise failure
    finally:
        if finalized:
            del execution_registry
        connection.close()


def run_world(experiment: CalibrationExperiment, source: XpoolConfig, *, repetition: int) -> MemoryProfileWorld:
    """Own fresh measurement and ordered retirement, retaining unknown worlds."""

    context = multiprocessing.get_context("spawn")
    visibility = visible_uuids()
    if max(source.devices) >= len(visibility):
        raise ValueError("memory-profile placement exceeds original device visibility")
    scope = MpsScope(MpsEndpoint(tuple(visibility[device] for device in source.atn.devices)))
    attention_environment = scope.endpoint.environment()
    attention_environment["CUDA_VISIBLE_DEVICES"] = ",".join(visibility)
    ffn_environment = {"CUDA_VISIBLE_DEVICES": ",".join(visibility), "CUDA_MPS_PIPE_DIRECTORY": ""}
    cancelled: int | None = None
    cleanup_deadline: float | None = None
    failure: RuntimeError | None = None
    failed: set[int] = set()
    join_started = False
    retirement_started = False
    stopped = False
    world: MemoryProfileWorld | None = None

    def record_cancellation(signum: int, frame: FrameType | None) -> None:
        nonlocal cancelled, cleanup_deadline
        if cancelled is None:
            cancelled = signum
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S

    def record_failure(message: str) -> None:
        nonlocal failure, cleanup_deadline
        if failure is None:
            failure = RuntimeError(message)
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S

    def receive(connection: Connection, process: BaseProcess, expected: str) -> object:
        execution_deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
        while True:
            if cancelled is not None and expected in ("uid", "prepared"):
                raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
            deadline = execution_deadline if cleanup_deadline is None else min(execution_deadline, cleanup_deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"memory-profile child {process.name} timed out waiting for {expected}")
            ready = wait((connection, process.sentinel), timeout=min(0.1, remaining))
            if connection not in ready:
                if process.sentinel in ready:
                    raise RuntimeError(f"memory-profile child {process.name} exited with code {process.exitcode}")
                continue
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"memory-profile child {process.name} closed before {expected}") from error
            if not isinstance(message, tuple) or len(message) != 2:
                raise RuntimeError(f"memory-profile child {process.name} returned an invalid message")
            tag, payload = message
            if tag == "failure":
                if not isinstance(payload, ParticipantFailure):
                    raise RuntimeError(f"memory-profile child {process.name} returned an invalid failure")
                record_failure(f"memory-profile child {process.name} failed: {payload.message}")
                return payload
            # A stop can be queued while a pre-join child is still preparing.
            if expected == "stopped" and tag == "prepared" and not join_started:
                continue
            if tag != expected:
                raise RuntimeError(f"memory-profile child {process.name} returned {tag!r}, expected {expected!r}")
            return payload

    workspace_root = source.cache_root / "memory-profile"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with (
        contextlib.ExitStack() as handlers,
        tempfile.TemporaryDirectory(prefix="xpool-memory-profile-", dir=workspace_root) as temporary,
    ):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers.enter_context(sighandle(signum, record_cancellation))
        config_path = Path(temporary) / "world.toml"
        config_environment = write_world_config(config_path, source, experiment)
        world_config = XpoolConfig.from_file(config_path, env=config_environment)
        model_specs = experiment.model_specs
        uid_connection, uid_child_connection = context.Pipe()
        uid_process: BaseProcess | None = None
        processes: list[BaseProcess] = []
        connections: list[Connection] = []
        uid_ready = False

        def retire_participants(command: str) -> None:
            nonlocal retirement_started, stopped, cleanup_deadline
            retirement_started = True
            if cleanup_deadline is None:
                cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            for index, (connection, process) in enumerate(zip(connections, processes, strict=True)):
                if command == "stop":
                    if index in failed:
                        continue
                    if not process.is_alive():
                        failed.add(index)
                        record_failure(f"memory-profile child {process.name} exited before stop: {process.exitcode}")
                        continue
                    try:
                        connection.send(command)
                    except (BrokenPipeError, EOFError, OSError) as error:
                        failed.add(index)
                        record_failure(f"memory-profile child {process.name} stop failed: {error}")
                else:
                    connection.send(command)
            for index, (connection, process) in enumerate(zip(connections, processes, strict=True)):
                if command == "stop" and index in failed:
                    continue
                response = receive(connection, process, "stopped")
                if isinstance(response, ParticipantFailure):
                    failed.add(index)
                    if command == "finalize":
                        raise RuntimeError(
                            f"memory-profile child {process.name} retirement is unconfirmed: {response.message}"
                        )
            for index, process in enumerate(processes):
                process.join(max(0.0, cleanup_deadline - time.monotonic()))
                if process.is_alive():
                    raise TimeoutError(f"memory-profile child {process.name} retirement is unconfirmed")
                if process.exitcode != 0 and (index not in failed or process.exitcode != 1):
                    if command == "finalize":
                        raise RuntimeError(f"memory-profile child {process.name} exited with code {process.exitcode}")
                    record_failure(f"memory-profile child {process.name} exited with code {process.exitcode}")
            stopped = True

        try:
            if cancelled is not None:
                raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
            with block_signals({signal.SIGINT, signal.SIGTERM}) as mask:
                # Publish the original mask before start; the child installs
                # ignored dispositions before restoring it at target entry.
                uid_process = context.Process(
                    target=uid_child,
                    args=(uid_child_connection, config_path),
                    kwargs={"config_environment": config_environment, "signal_mask": mask},
                    name="memory-profile-uid",
                )
                uid_process.start()
                uid_child_connection.close()
            uid = receive(uid_connection, uid_process, "uid")
            if isinstance(uid, ParticipantFailure):
                if failure is None:
                    raise AssertionError("UID failure was not retained")
                raise failure
            if not isinstance(uid, str):
                raise RuntimeError("memory-profile UID child returned a non-string UID")
            uid_ready = True
            fabric_plan = experiment.fabric_plan(uid, config=world_config)
            model_description = ",".join(f"{model.spec.model_id}-tp{model.ffn_tp_size}" for model in experiment.models)
            if cancelled is not None:
                raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
            scope.start()
            for pe, placement in enumerate(fabric_plan.pe_placements):
                if cancelled is not None:
                    raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
                with block_signals({signal.SIGINT, signal.SIGTERM}) as mask:
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=participant_child,
                        kwargs={
                            "connection": child,
                            "config_path": config_path,
                            "config_environment": config_environment,
                            "model_specs": model_specs,
                            "fabric_plan": fabric_plan,
                            "pe": pe,
                            "signal_mask": mask,
                            "environment": attention_environment
                            if placement.role is FabricRole.ATNAGENT
                            else ffn_environment,
                        },
                        name=f"memory-profile-{model_description}-pe{pe}",
                    )
                    process.start()
                    processes.append(process)
                    connections.append(parent)
                    child.close()

            for index, (connection, process) in enumerate(zip(connections, processes, strict=True)):
                response = receive(connection, process, "prepared")
                if isinstance(response, ParticipantFailure):
                    failed.add(index)
                    if response.ready_to_finalize:
                        join_started = True
                        raise RuntimeError("memory-profile participant reported join before its command")
            rows = []
            if failure is None and cancelled is None:
                join_started = True
                for connection in connections:
                    connection.send("join")
                joined = set()
                for index, (connection, process) in enumerate(zip(connections, processes, strict=True)):
                    response = receive(connection, process, "joined")
                    if isinstance(response, ParticipantFailure):
                        failed.add(index)
                        if not response.ready_to_finalize:
                            continue
                    joined.add(index)
                if len(joined) != len(processes):
                    if failure is None:
                        raise AssertionError("incomplete join has no retained failure")
                    raise failure
                eligible = True
                if failure is None and cancelled is None:
                    atnagent_count = len(world_config.atn.devices)
                    for connection in connections[atnagent_count:]:
                        connection.send("execute")
                    for index in range(atnagent_count, len(processes)):
                        response = receive(connections[index], processes[index], "execution")
                        if isinstance(response, ParticipantFailure):
                            failed.add(index)
                            eligible = eligible and response.ready_to_finalize
                        elif isinstance(response, MemoryProfileParticipantEvidence):
                            rows.append(response)
                        else:
                            raise RuntimeError("memory-profile participant returned invalid evidence")
                if not eligible:
                    if failure is None:
                        raise AssertionError("unavailable retirement has no retained failure")
                    raise failure
                retire_participants("finalize")
            else:
                retire_participants("stop")
            if failure is None and cancelled is None:
                ordered = tuple(sorted(rows, key=lambda row: row.ffnagent_index))
                if len(ordered) != len(world_config.ffn.devices):
                    raise RuntimeError("memory-profile evidence does not cover every configured FfnAgent")
                world = MemoryProfileWorld(experiment=experiment, repetition=repetition, participants=ordered)
        except BaseException as error:
            if failure is not None and error is not failure:
                raise failure from error
            raise
        finally:
            if cleanup_deadline is None:
                cleanup_deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
            try:
                if not stopped and not join_started and not retirement_started:
                    retire_participants("stop")
                if not stopped:
                    raise RuntimeError("memory-profile participant retirement is unconfirmed")
                if uid_process is not None and uid_process.pid is not None and uid_process.is_alive():
                    if uid_ready:
                        uid_connection.send("stop")
                        receive(uid_connection, uid_process, "stopped")
                    else:
                        uid_process.kill()
                    uid_process.join(max(0.0, cleanup_deadline - time.monotonic()))
                if uid_process is not None and uid_process.is_alive():
                    raise TimeoutError("memory-profile UID owner retirement is unconfirmed")
                if uid_ready and uid_process is not None and uid_process.exitcode != 0:
                    record_failure(f"memory-profile UID owner exited with code {uid_process.exitcode}")
                for connection in (uid_connection, uid_child_connection, *connections):
                    connection.close()
                if uid_process is not None:
                    uid_process.close()
                for process in processes:
                    process.close()
                scope.stop(deadline=cleanup_deadline)
            except BaseException as cleanup_error:
                logger.error(
                    "memory-profile cleanup unconfirmed; retaining owner and MPS: %s; execution failure: %s",
                    cleanup_error,
                    failure,
                    exc_info=cleanup_error,
                )
                while True:
                    time.sleep(1.0)

    if failure is not None:
        raise failure
    if cancelled is not None:
        raise KeyboardInterrupt(f"memory profiling cancelled by signal {cancelled}")
    if world is None:
        raise AssertionError("completed calibration world has no evidence")
    return world


def refuse_live_daemon(config: XpoolConfig) -> None:
    """Reject profiling while the configured daemon endpoint responds."""

    host = f"[{config.daemon.host}]" if ":" in config.daemon.host else config.daemon.host
    try:
        httpx.get(f"http://{host}:{config.daemon.port}/health", timeout=1.0)
    except httpx.HTTPError:
        return
    raise RuntimeError("xpool memory-profile refuses to run while the configured daemon endpoint responds")


def profile_ffn_memory() -> MemoryProfileEvidence:
    """Run the fixed matrix and return its qualified candidate with evidence."""

    config = get_global_config()
    if config.ffn.device_memory_calibration is None:
        raise RuntimeError("ffn.device_memory_calibration is required for xpool memory-profile")
    refuse_live_daemon(config)

    # Repeated fit worlds determine coefficients; the disjoint held-out world
    # determines the minimum headroom required for an unseen placement shape.
    fit_experiments, held_out_experiment = calibration_experiments(config)
    fit_worlds_evidence = tuple(
        run_world(experiment, config, repetition=repetition)
        for experiment in fit_experiments
        for repetition in range(1, REPETITION_COUNT + 1)
    )
    held_out_worlds = tuple(
        run_world(held_out_experiment, config, repetition=repetition) for repetition in range(1, REPETITION_COUNT + 1)
    )
    all_worlds = (*fit_worlds_evidence, *held_out_worlds)
    devices = all_worlds[0].devices
    environment = all_worlds[0].environment
    if any(world.devices != devices or world.environment != environment for world in all_worlds[1:]):
        raise RuntimeError("xpool memory-profile worlds disagree on hardware or software environment")
    coefficients, minimum_headroom = fit_worlds(fit_worlds_evidence, held_out_worlds)
    if environment.native_abi_version != ABI_VERSION:
        raise RuntimeError("xpool memory-profile child native ABI disagrees with Python")
    profile = XpoolMemoryCalibrationProfile(
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
    return MemoryProfileEvidence(profile=profile, fit_worlds=fit_worlds_evidence, held_out_worlds=held_out_worlds)
