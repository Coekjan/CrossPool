"""FfnAgent process lifecycle."""

from __future__ import annotations

import logging
import os
from time import monotonic

import torch

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase, FabricParticipantPhase, FabricRole
from xpool.memory import load_memory_calibration_profile
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.runtime.agent import Agent, AgentError, AgentHeartbeat
from xpool.runtime.ffnagent.architecture import load
from xpool.runtime.ffnagent.device_memory import DeviceMemoryEstimator, ensure_supported_cuda_allocator
from xpool.runtime.ffnagent.loader import materialize_layer_weights
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.runtime.ffnagent.weights import FfnLayerWeights
from xpool.service.client import XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import FfnAgentRegistration, HeartbeatResponse

logger = logging.getLogger(__name__)


class FfnAgent(Agent):
    """FfnAgent owning generation-scoped Fabric progress."""

    def __init__(self, *, cuda_device: int) -> None:
        """Initialize native FFN role state and daemon ownership.

        CUDA must not already be initialized. Construction installs the cuBLAS
        workspace policy, initializes the native FfnAgent role, validates any
        selected calibration profile against the local GPU, and loads every
        configured model specification.

        Raises:
            AgentError: If CUDA is already initialized or calibration does not
                match the configured FfnAgent GPU.
        """

        if torch.cuda.is_initialized():
            raise AgentError("FfnAgent CUDA initialized before installing its cuBLAS workspace policy")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
        super().__init__(cuda_device=cuda_device, runtime_role=RuntimeRole.FFNAGENT)
        ensure_supported_cuda_allocator()
        config = get_global_config()
        calibration_profile = load_memory_calibration_profile()
        if calibration_profile is not None:
            ffnagent = config.ffnagent_by_cuda_device.get(cuda_device)
            if ffnagent is None:
                raise AgentError(f"CUDA device {cuda_device} is not a configured FfnAgent")
            expected = calibration_profile.environment.ffnagent_gpus[ffnagent.rank]
            properties = torch.cuda.get_device_properties(cuda_device)
            mismatches = [
                f"{name}: expected {expected_value!r}, found {actual_value!r}"
                for name, expected_value, actual_value in (
                    ("name", expected.name, properties.name),
                    (
                        "compute capability",
                        expected.compute_capability,
                        (properties.major, properties.minor),
                    ),
                    ("total memory", expected.total_memory_bytes, properties.total_memory),
                )
                if actual_value != expected_value
            ]
            if mismatches:
                raise AgentError("memory calibration GPU is incompatible: " + "; ".join(mismatches))
        model_specs = tuple(
            load(model_id=model.id, model_path=config.model_path_of(model.id)) for model in config.models
        )
        torch.cuda.synchronize(cuda_device)
        torch.cuda.empty_cache()
        cuda_free_memory_bytes, cuda_total_memory_bytes = torch.cuda.mem_get_info(cuda_device)
        self.registration = FfnAgentRegistration(
            cuda_device=cuda_device,
            cuda_total_memory_bytes=cuda_total_memory_bytes,
            cuda_free_memory_bytes=cuda_free_memory_bytes,
            model_specs=model_specs,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        self.layer_weights: tuple[tuple[FfnLayerWeights | None, ...], ...] | None = None
        self.execution_registry: FfnExecutionRegistry | None = None
        self.heartbeat_worker = AgentHeartbeat(agent=self)
        logger.info(
            "initialized device=%s model_count=%s layer_count=%s",
            cuda_device,
            len(model_specs),
            sum(len(spec.layers) for spec in model_specs),
        )

    def activate_fabric(self) -> None:
        """Start persistent Fabric progress on this FfnAgent PE."""

        xpool.native.ffnagent.activate()
        xpool.native.ffnagent.check_health()

    def quiesce_fabric(self) -> None:
        """Keep coordinator progress alive while AtnAgents quiesce producers."""

    def register(self) -> None:
        """Register this FfnAgent with the daemon."""

        try:
            self.client.register_ffnagent(self.registration)
        except (XpoolDaemonError, XpoolClientError) as exc:
            self.registered = False
            if not exc.is_recoverable:
                raise AgentError(f"FfnAgent registration received unrecoverable daemon error: {exc}") from exc
            logger.debug("registration failed device=%s detail=%s", self.cuda_device, exc)
            return
        self.registered = True
        logger.info("registered device=%s pid=%s", self.cuda_device, self.proc_id.pid)

    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish this FfnAgent's heartbeat to its role-specific endpoint."""

        return self.client.heartbeat_ffnagent(self.cuda_device, self.process_ref)

    def prepare_fabric_join(self) -> bool:
        """Materialize production weights before entering the Fabric world."""

        plan = self.fabric_plan
        if plan is None:
            raise AgentError("FFN weight preparation requires a retained Fabric Plan")
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in plan.pe_placements)
        ffnagent_index = self.fabric_pe() - atnagent_count
        estimator = DeviceMemoryEstimator(
            model_specs=self.registration.model_specs,
            instance_profiles=tuple(instance.ffn_profile for instance in plan.instance_plans),
        )
        estimate = estimator.estimate(fabric_plan=plan, ffnagent_index=ffnagent_index)
        torch.cuda.empty_cache()
        free_memory_bytes, _ = torch.cuda.mem_get_info(self.cuda_device)
        extra_margin_bytes = get_global_config().ffn.device_memory_extra_margin_bytes
        required_bytes = estimate.peak_bytes + extra_margin_bytes
        admission = "calibrated" if estimator.coefficients is not None else "analytic"
        if required_bytes > free_memory_bytes:
            raise AgentError(
                f"FFN {admission} memory admission requires {required_bytes} bytes, "
                f"but CUDA device {self.cuda_device} has {free_memory_bytes} free bytes"
            )
        logger.info(
            "memory admitted kind=%s required_bytes=%s free_bytes=%s margin_bytes=%s device=%s",
            admission,
            required_bytes,
            free_memory_bytes,
            extra_margin_bytes,
            self.cuda_device,
        )
        weights_started_at = monotonic()
        self.layer_weights = materialize_layer_weights(
            fabric_plan=plan,
            model_specs=self.registration.model_specs,
            ffnagent_index=ffnagent_index,
        )
        logger.info(
            "weights materialized device=%s model_count=%s layer_count=%s elapsed=%.3fs",
            self.cuda_device,
            sum(any(layer is not None for layer in model_layers) for model_layers in self.layer_weights),
            sum(layer is not None for model_layers in self.layer_weights for layer in model_layers),
            monotonic() - weights_started_at,
        )

        return True

    def prepare_fabric_execution(self) -> None:
        """Capture and atomically install the Plan-selected local execution."""

        if self.fabric_plan is None or self.layer_weights is None:
            raise AgentError("FFN execution preparation requires a retained Plan and weights")
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in self.fabric_plan.pe_placements)
        self.execution_registry = FfnExecutionRegistry.materialize(
            fabric_plan=self.fabric_plan,
            model_specs=self.registration.model_specs,
            ffnagent_index=self.fabric_pe() - atnagent_count,
            layer_weights=self.layer_weights,
        )
        self.layer_weights = None

    def poll_fabric_health(self) -> None:
        """Check Fabric failure state and the selected FfnAgent runtime."""

        super().poll_fabric_health()
        report = self.participant_report
        if (
            report is not None
            and report.phase is FabricParticipantPhase.ACTIVE
            and report.invocation_failure is None
            and self.fabric_phase in {FabricGenerationPhase.ACTIVATING, FabricGenerationPhase.EXECUTABLE}
        ):
            try:
                xpool.native.ffnagent.check_health()
            except RuntimeError as error:
                self.report_local_control_failure(str(error))
                raise

    def close_role(self) -> None:
        """Release FFN-side resources after Fabric shutdown."""

        self.execution_registry = None
        self.layer_weights = None
