"""FfnAgent process lifecycle."""

from __future__ import annotations

import logging
from time import monotonic

import torch

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase, FabricParticipantPhase, FabricRole
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.runtime import memory_calibration
from xpool.runtime.agent import Agent, AgentError, AgentHeartbeat
from xpool.runtime.ffnagent import capture_workspace
from xpool.runtime.ffnagent.architecture import load
from xpool.runtime.ffnagent.loader import materialize_layer_weights
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.runtime.ffnagent.weights import FfnLayerWeights
from xpool.service.client import XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import FfnAgentRegistration, HeartbeatResponse

logger = logging.getLogger(__name__)


class FfnAgent(Agent):
    """FfnAgent owning generation-scoped Fabric progress."""

    def __init__(self, *, device: int) -> None:
        """Initialize native FFN role state and daemon ownership.

        CUDA must not already be initialized. Construction initializes the native
        FfnAgent role, discovers execution resource and common environment
        facts, and loads every configured model specification.

        Raises:
            AgentError: If CUDA is already initialized.
            pydantic.ValidationError: If registration memory observations disagree.
        """

        if torch.cuda.is_initialized():
            raise AgentError("FfnAgent CUDA initialized before native FFN bootstrap")
        super().__init__(device=device, runtime_role=RuntimeRole.FFNAGENT)
        execution_contract, self.external_resources = capture_workspace.prepare_execution_sources()
        config = get_global_config()
        calibration_environment = (
            memory_calibration.observe_environment(self.device)
            if config.ffn.device_memory_calibration is not None
            else None
        )
        model_specs = tuple(
            load(model_id=model.id, model_path=config.model_path_of(model.id)) for model in config.models
        )
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        device_free_memory_bytes, device_total_memory_bytes = torch.cuda.mem_get_info(self.device)
        self.registration = FfnAgentRegistration(
            device=device,
            calibration_environment=calibration_environment,
            device_free_memory_bytes=device_free_memory_bytes,
            device_total_memory_bytes=device_total_memory_bytes,
            execution_contract=execution_contract,
            model_specs=model_specs,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        self.layer_weights: tuple[tuple[FfnLayerWeights | None, ...], ...] | None = None
        self.execution_registry: FfnExecutionRegistry | None = None
        self.heartbeat_worker = AgentHeartbeat(agent=self)
        logger.info(
            "initialized device=%s model_count=%s layer_count=%s",
            device,
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
            logger.debug("registration failed device=%s detail=%s", self.device, exc)
            return
        self.registered = True
        logger.info("registered device=%s pid=%s", self.device, self.proc_id.pid)

    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish this FfnAgent's heartbeat to its role-specific endpoint."""

        return self.client.heartbeat_ffnagent(self.device, self.process_ref)

    def prepare_fabric_join(self) -> bool:
        """Materialize production weights before entering the Fabric world."""

        plan = self.fabric_plan
        if plan is None:
            raise AgentError("FFN weight preparation requires a retained Fabric Plan")
        config = get_global_config()
        if plan.executor_lane_count != config.scheduler.ffn_concurrency:
            raise AgentError("Fabric Plan executor Lane count disagrees with the process configuration")
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in plan.pe_placements)
        ffnagent_index = self.fabric_pe() - atnagent_count
        if plan.ffn_execution_contracts[ffnagent_index] != self.registration.execution_contract:
            raise AgentError("Fabric Plan differs from registered FfnAgent execution resources")
        estimate = plan.ffn_memory_estimates[ffnagent_index]
        torch.cuda.empty_cache()
        free_memory_bytes, _ = torch.cuda.mem_get_info(self.device)
        extra_margin_bytes = get_global_config().ffn.device_memory_extra_margin_bytes
        required_bytes = estimate.peak_bytes + extra_margin_bytes
        admission = "calibrated" if config.ffn.device_memory_calibration is not None else "analytic"
        if required_bytes > free_memory_bytes:
            raise AgentError(
                f"FFN {admission} memory admission requires {required_bytes} bytes, "
                f"but device {self.device} has {free_memory_bytes} free bytes"
            )
        logger.info(
            "memory admitted kind=%s required_bytes=%s free_bytes=%s margin_bytes=%s device=%s",
            admission,
            required_bytes,
            free_memory_bytes,
            extra_margin_bytes,
            self.device,
        )
        weights_started_at = monotonic()
        self.layer_weights = materialize_layer_weights(
            fabric_plan=plan,
            model_specs=self.registration.model_specs,
            ffnagent_index=ffnagent_index,
        )
        logger.info(
            "weights materialized device=%s model_count=%s layer_count=%s elapsed=%.3fs",
            self.device,
            sum(any(layer is not None for layer in model_layers) for model_layers in self.layer_weights),
            sum(layer is not None for model_layers in self.layer_weights for layer in model_layers),
            monotonic() - weights_started_at,
        )

        return True

    def prepare_fabric_execution(self) -> None:
        """Capture and atomically install the Plan-selected local execution."""

        if self.execution_registry is not None:
            raise AgentError("FFN execution preparation cannot be retried")
        if self.fabric_plan is None or self.layer_weights is None:
            raise AgentError("FFN execution preparation requires a retained Plan and weights")
        atnagent_count = sum(placement.role is FabricRole.ATNAGENT for placement in self.fabric_plan.pe_placements)
        self.execution_registry = FfnExecutionRegistry(
            layer_weights=self.layer_weights, external_resources=self.external_resources
        )
        self.layer_weights = None
        self.external_resources = ()
        self.execution_registry.materialize(
            fabric_plan=self.fabric_plan,
            model_specs=self.registration.model_specs,
            ffnagent_index=self.fabric_pe() - atnagent_count,
        )

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

        if self.execution_registry is not None:
            self.execution_registry.close()
        self.execution_registry = None
        self.layer_weights = None
        self.external_resources = ()
