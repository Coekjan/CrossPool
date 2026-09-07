"""Shared runtime lifecycle support for xpool agents."""

from __future__ import annotations

import logging
import signal
import threading
import time
from abc import ABC, abstractmethod

import xpool.native
from xpool import bootstrap, devkit
from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricGenerationPhase,
    FabricParticipantPhase,
    FabricPlan,
    FabricRole,
    FifoSchedulerPolicy,
    RandomSchedulerPolicy,
)
from xpool.native import ABI_VERSION, RuntimeRole
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    FabricInvocationFailure,
    FabricParticipantReport,
    FabricQuiesceRequest,
    HeartbeatResponse,
    ProcessRef,
)
from xpool.utils.background import BackgroundThread
from xpool.utils.procs import ProcUniqId, bail
from xpool.utils.sighandler import sighandle

AGENT_CONTROL_INTERVAL_S = 0.5
AGENT_HEARTBEAT_INTERVAL_S = 5.0
AGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S = 5.0
AGENT_SHUTDOWN_POLL_INTERVAL_S = 0.5
FABRIC_REPORT_RETRY_ATTEMPTS = 3
FABRIC_REPORT_RETRY_DELAY_S = 0.5
logger = logging.getLogger(__name__)


def project_fabric_arena(plan: FabricPlan) -> xpool.native.fabric.ArenaProjection:
    """Project one daemon-authored Plan into the sole native join value."""

    if isinstance(plan.scheduler, FifoSchedulerPolicy):
        scheduler = xpool.native.fabric.SchedulerPolicy.fifo()
    elif isinstance(plan.scheduler, RandomSchedulerPolicy):
        scheduler = xpool.native.fabric.SchedulerPolicy.random(plan.scheduler.seed)
    else:
        raise AgentError("Fabric plan contains an unsupported scheduler variant")
    atnagent_count = sum(item.role is FabricRole.ATNAGENT for item in plan.pe_placements)
    instances = []
    for model_plan, instance_plan in zip(plan.model_plans, plan.instance_plans, strict=True):
        profile = instance_plan.ffn_profile
        topology = instance_plan.instance_rank_topology
        layers = tuple(
            xpool.native.fabric.InstanceLayerProjection(
                layer_id=profile_layer.layer_id,
                kind=profile_layer.kind,
                effective_topk=0 if isinstance(layer_plan, DenseFfnLayerPlan) else layer_plan.effective_topk,
                ffnagent_indices=layer_plan.ffnagent_indices,
            )
            for profile_layer, layer_plan in zip(profile.layers, model_plan.layers, strict=True)
        )
        instances.append(
            xpool.native.fabric.InstanceProjection(
                payload_dtype=profile.payload_dtype,
                hidden_size=profile.hidden_size,
                decode_payload_row_capacity=profile.decode_payload_row_capacity,
                prefill_payload_row_capacity=profile.prefill_payload_row_capacity,
                group_sum_complete_admitted=profile.group_sum_complete_admitted,
                atn_tp_size=topology.atn_tp_size,
                atn_dp_size=topology.atn_dp_size,
                atnagent_indices=topology.atnagent_indices,
                layers=layers,
            )
        )
    return xpool.native.fabric.ArenaProjection(
        generation_high=plan.generation.high,
        generation_low=plan.generation.low,
        uid=plan.uid.value,
        atnagent_count=atnagent_count,
        ffnagent_count=len(plan.pe_placements) - atnagent_count,
        executor_lane_count=plan.executor_lane_count,
        scheduler=scheduler,
        instances=tuple(instances),
    )


class AgentError(RuntimeError):
    """Raised when an xpool Agent cannot preserve lifecycle invariants."""


class Agent(ABC):
    """Common process-local resources for one configured xpool Agent."""

    def __init__(self, *, cuda_device: int, runtime_role: RuntimeRole) -> None:
        """Initialize native state and daemon client for one Agent role."""

        bootstrap.init(cuda_device, runtime_role)
        devkit.install()
        self.cuda_device = cuda_device
        self.runtime_role = runtime_role
        self.proc_id = ProcUniqId.current()
        self.process_ref = ProcessRef(abi_version=ABI_VERSION, pid=self.proc_id.pid)
        self.client = XpoolClient()
        self.registered = False
        self.fabric_plan: FabricPlan | None = None
        self.fabric_arena_projection: xpool.native.fabric.ArenaProjection | None = None
        self.fabric_phase: FabricGenerationPhase | None = None
        self.participant_report: FabricParticipantReport | None = None
        self.fabric_stopped = False
        self.heartbeat_worker: AgentHeartbeat
        logger.info("starting device=%s pid=%s", self.cuda_device, self.proc_id.pid)

    @abstractmethod
    def activate_fabric(self) -> None:
        """Activate role-local device progress before reporting readiness."""

    def shutdown_fabric(self) -> None:
        """Request and complete coordinated Fabric quiesce and native drain."""

        if self.fabric_plan is None:
            return
        self.client.request_fabric_quiesce(
            FabricQuiesceRequest(owner=self.process_ref, generation=self.fabric_plan.generation)
        )
        while self.fabric_plan is not None:
            self.heartbeat_worker.raise_if_failed()
            if self.heartbeat_worker.consume_registration_missing():
                raise AgentError("Agent registration disappeared during coordinated Fabric shutdown")
            response = self.heartbeat_worker.consume_response()
            if response is not None:
                self.handle_heartbeat_response(response)
            self.poll_fabric_health()
            self.advance_fabric_lifecycle()
            if self.fabric_plan is not None:
                time.sleep(AGENT_SHUTDOWN_POLL_INTERVAL_S)

    def handle_heartbeat_response(self, response: HeartbeatResponse) -> None:
        """Apply one daemon-authoritative snapshot on the Agent main thread."""

        if self.fabric_plan is not None and response.generation != self.fabric_plan.generation:
            raise AgentError("daemon heartbeat returned a different Fabric generation")
        self.fabric_phase = response.fabric_phase

    def poll_fabric_health(self) -> None:
        """Publish canonical native failure or reject unexpected Resident exit."""

        report = self.participant_report
        if (
            self.fabric_plan is None
            or report is None
            or report.phase
            in {
                FabricParticipantPhase.JOIN_READY,
                FabricParticipantPhase.JOINING,
                FabricParticipantPhase.FINALIZED,
            }
        ):
            return
        try:
            failure = xpool.native.fabric.failure()
            if failure is not None:
                invocation_failure = FabricInvocationFailure(
                    result_code=failure.result_code,
                    origin_pe=failure.origin_pe,
                    instance_index=failure.key.instance_index,
                    invocation_sequence=failure.key.invocation_sequence,
                    layer_ordinal=failure.layer_ordinal,
                )
                if report.invocation_failure != invocation_failure:
                    self.report_fabric_phase(report.phase, invocation_failure=invocation_failure)
        except Exception as error:
            self.report_local_control_failure(str(error))
            raise

    def advance_fabric_lifecycle(self) -> None:
        """Advance startup and shutdown from the daemon-authoritative phase."""

        if self.fabric_plan is None:
            try:
                plan = self.client.fabric_plan()
            except (XpoolClientError, XpoolDaemonError) as error:
                if error.is_recoverable:
                    return
                raise AgentError(f"Fabric plan acquisition failed: {error}") from error
            self.fabric_plan = plan
            self.fabric_phase = FabricGenerationPhase.PREPARING_JOIN
            logger.info("fabric plan acquired generation=%s device=%s", plan.generation.format(), self.cuda_device)

        report = self.participant_report
        try:
            match self.fabric_phase:
                case FabricGenerationPhase.PREPARING_JOIN if report is None:
                    if not self.prepare_fabric_join():
                        return
                    self.fabric_arena_projection = project_fabric_arena(self.fabric_plan)
                    self.report_fabric_phase(FabricParticipantPhase.JOIN_READY)
                case FabricGenerationPhase.JOINING if report is not None:
                    if report.phase is FabricParticipantPhase.JOIN_READY:
                        self.report_fabric_phase(FabricParticipantPhase.JOINING)
                        report = self.participant_report
                    if report is not None and report.phase is FabricParticipantPhase.JOINING:
                        projection = self.fabric_arena_projection
                        if projection is None:
                            raise AgentError("Fabric join requires the retained Arena Projection")
                        xpool.native.fabric.join(projection, self.fabric_pe())
                        self.report_fabric_phase(FabricParticipantPhase.JOINED)
                        logger.info(
                            "fabric joined pe=%s generation=%s",
                            self.fabric_pe(),
                            self.fabric_plan.generation.format(),
                        )
                case FabricGenerationPhase.PREPARING_EXECUTION if report is not None:
                    if report.phase is FabricParticipantPhase.JOINED:
                        self.prepare_fabric_execution()
                        self.report_fabric_phase(FabricParticipantPhase.EXECUTION_READY)
                case FabricGenerationPhase.ACTIVATING if report is not None:
                    if report.phase is FabricParticipantPhase.EXECUTION_READY:
                        self.activate_fabric()
                        self.report_fabric_phase(FabricParticipantPhase.ACTIVE)
                        logger.info(
                            "active pe=%s generation=%s",
                            self.fabric_pe(),
                            self.fabric_plan.generation.format(),
                        )
                case FabricGenerationPhase.ABORTING:
                    raise AgentError("daemon selected fail-stop Fabric abort")
                case FabricGenerationPhase.QUIESCING if report is not None:
                    if report.phase is FabricParticipantPhase.ACTIVE:
                        self.quiesce_fabric()
                        self.report_fabric_phase(FabricParticipantPhase.QUIESCED)
                case FabricGenerationPhase.DRAINING if report is not None:
                    if report.phase is FabricParticipantPhase.QUIESCED:
                        xpool.native.fabric.drain_async()
                        self.report_fabric_phase(FabricParticipantPhase.DRAINING)
                    elif report.phase is FabricParticipantPhase.DRAINING and not xpool.native.fabric.drain_pending():
                        self.report_fabric_phase(FabricParticipantPhase.DRAINED)
                case FabricGenerationPhase.FINALIZING if report is not None:
                    if report.phase is FabricParticipantPhase.DRAINED:
                        xpool.native.fabric.finalize()
                        self.report_fabric_phase(FabricParticipantPhase.FINALIZED)
                case FabricGenerationPhase.STOPPED if report is not None:
                    if report.phase is FabricParticipantPhase.FINALIZED:
                        self.fabric_plan = None
                        self.fabric_arena_projection = None
                        self.fabric_phase = None
                        self.participant_report = None
                        self.fabric_stopped = True
                        self.heartbeat_worker.stop()
                case _:
                    return
        except Exception as error:
            self.report_local_control_failure(f"Fabric lifecycle failed: {error}")
            raise AgentError(f"Fabric lifecycle failed: {error}") from error

    @abstractmethod
    def quiesce_fabric(self) -> None:
        """Stop role-local admissions before native Fabric drain begins."""

    @abstractmethod
    def register(self) -> None:
        """Register the concrete Agent type with the daemon."""

    @abstractmethod
    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish one role-specific process-liveness heartbeat."""

    def report_fabric_phase(
        self,
        phase: FabricParticipantPhase,
        *,
        invocation_failure: FabricInvocationFailure | None = None,
        control_failure: str | None = None,
    ) -> None:
        """Commit one local phase or failure enrichment through daemon 204."""

        if self.fabric_plan is None:
            raise AgentError("cannot report Fabric participant progress before retaining a plan")
        previous = self.participant_report
        if previous is None:
            if phase is not FabricParticipantPhase.JOIN_READY:
                raise AgentError("first local Fabric participant phase must be join_ready")
        elif phase is not previous.phase and not previous.phase.allows(phase):
            raise AgentError(f"local Fabric participant phase cannot move from {previous.phase.value} to {phase.value}")

        candidate = FabricParticipantReport(
            owner=self.process_ref,
            generation=self.fabric_plan.generation,
            pe=self.fabric_pe(),
            phase=phase,
            invocation_failure=(
                invocation_failure
                if invocation_failure is not None
                else None
                if previous is None
                else previous.invocation_failure
            ),
            control_failure=(
                control_failure
                if control_failure is not None
                else None
                if previous is None
                else previous.control_failure
            ),
        )
        last_error: XpoolClientError | XpoolDaemonError | None = None
        for attempt in range(FABRIC_REPORT_RETRY_ATTEMPTS):
            try:
                self.client.report_fabric_participant(candidate)
            except (XpoolClientError, XpoolDaemonError) as error:
                last_error = error
                if not error.is_recoverable:
                    raise AgentError(f"daemon rejected Fabric participant report: {error}") from error
                if attempt + 1 < FABRIC_REPORT_RETRY_ATTEMPTS:
                    time.sleep(FABRIC_REPORT_RETRY_DELAY_S)
            else:
                self.participant_report = candidate
                return
        raise AgentError(f"Fabric participant report was not acknowledged: {last_error}")

    def report_local_control_failure(self, message: str) -> None:
        """Best-effort enrich the current report with one local control failure."""

        if self.fabric_plan is None or self.participant_report is None:
            return
        try:
            self.report_fabric_phase(
                self.participant_report.phase,
                control_failure=message,
            )
        except AgentError:
            logger.exception("failed to report local fabric control failure")

    def fabric_pe(self) -> int:
        """Return this Agent's unique immutable Fabric PE index."""

        if self.fabric_plan is None:
            raise AgentError("Fabric placement requires a retained plan")
        role = FabricRole.ATNAGENT if self.runtime_role is RuntimeRole.ATNAGENT else FabricRole.FFNAGENT
        pes = tuple(
            pe
            for pe, item in enumerate(self.fabric_plan.pe_placements)
            if item.cuda_device == self.cuda_device and item.role is role
        )
        if len(pes) != 1:
            raise AgentError("Fabric plan has no unique placement for this Agent")
        return pes[0]

    @abstractmethod
    def prepare_fabric_join(self) -> bool:
        """Prepare role-local resources and report whether Fabric may join."""

    @abstractmethod
    def prepare_fabric_execution(self) -> None:
        """Install role-local post-join resources before activation."""

    @abstractmethod
    def close_role(self) -> None:
        """Release role-local resources after coordinated Fabric shutdown."""

    def run_control_loop(self, shutdown_requested: threading.Event) -> None:
        """Advance registration and Fabric lifecycle until shutdown."""

        while not self.fabric_stopped and not shutdown_requested.is_set():
            self.heartbeat_worker.raise_if_failed()
            if self.heartbeat_worker.consume_registration_missing():
                self.heartbeat_worker.stop()
                if self.fabric_plan is not None:
                    raise AgentError("agent registration disappeared after fabric plan acquisition")
                self.registered = False
            response = self.heartbeat_worker.consume_response()
            if response is not None:
                self.handle_heartbeat_response(response)
            if shutdown_requested.is_set():
                break
            if not self.registered:
                self.register()
                if self.registered:
                    self.heartbeat_worker.start()
            if shutdown_requested.is_set():
                break
            if not self.registered:
                shutdown_requested.wait(AGENT_CONTROL_INTERVAL_S)
                continue
            self.advance_fabric_lifecycle()
            if shutdown_requested.is_set():
                break
            self.poll_fabric_health()
            if not self.fabric_stopped:
                shutdown_requested.wait(AGENT_CONTROL_INTERVAL_S)

    def run(self) -> None:
        """Run registration, role preparation, and Fabric lifecycle."""

        shutdown_requested = threading.Event()

        def request_shutdown(signal_number: int, frame: object) -> None:
            logger.info("shutdown requested device=%s pid=%s", self.cuda_device, self.proc_id.pid)
            shutdown_requested.set()

        with (
            sighandle(signal.SIGINT, request_shutdown),
            sighandle(signal.SIGTERM, request_shutdown),
        ):
            try:
                self.run_control_loop(shutdown_requested)
            except Exception:
                logger.exception(
                    "agent failed and will exit without unsafe native cleanup device=%s",
                    self.cuda_device,
                )
                if self.participant_report is None:
                    owners = (
                        ("heartbeat worker", self.heartbeat_worker.close),
                        ("role resources", self.close_role),
                        ("daemon client", self.client.close),
                    )
                    for name, close in owners:
                        try:
                            close()
                        except Exception:
                            logger.exception("failed to close pre-join %s device=%s", name, self.cuda_device)
                bail(code=1)
            try:
                self.shutdown_fabric()
            except Exception:
                logger.exception(
                    "coordinated fabric shutdown failed; skipping unsafe native cleanup device=%s",
                    self.cuda_device,
                )
                bail(code=1)
            self.heartbeat_worker.close()
            try:
                self.close_role()
            finally:
                self.client.close()
            logger.info("stopped device=%s pid=%s", self.cuda_device, self.proc_id.pid)


class AgentHeartbeat:
    """Background liveness worker and single-slot daemon snapshot producer."""

    def __init__(
        self,
        *,
        agent: Agent,
        interval_s: float = AGENT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Create a stopped Agent heartbeat worker."""

        self.agent = agent
        self.interval_s = interval_s
        self.worker = BackgroundThread.periodic(
            name=f"xpool-agent-heartbeat-{agent.cuda_device}",
            interval_s=interval_s,
            target=self.heartbeat_once,
            join_timeout_s=AGENT_HEARTBEAT_STOP_JOIN_TIMEOUT_S,
        )
        self.lock = threading.Lock()
        self.registration_missing = False
        self.latest_response: HeartbeatResponse | None = None
        self.closed = False

    @property
    def thread(self) -> threading.Thread | None:
        """Return the current heartbeat thread, if started."""

        return self.worker.thread

    def start(self) -> None:
        """Start periodic heartbeats."""

        with self.lock:
            if self.closed:
                raise AgentError("cannot restart a closed Agent heartbeat worker")
            if self.worker.is_running:
                return
            self.registration_missing = False
        self.worker.start()

    def stop(self) -> None:
        """Stop periodic heartbeats."""

        self.worker.stop()

    def close(self) -> None:
        """Stop and permanently close the worker."""

        with self.lock:
            if self.closed:
                return
        self.worker.close()
        with self.lock:
            self.closed = True

    def consume_registration_missing(self) -> bool:
        """Return and clear the missing-registration signal."""

        with self.lock:
            missing = self.registration_missing
            self.registration_missing = False
        return missing

    def consume_response(self) -> HeartbeatResponse | None:
        """Return and clear the newest daemon snapshot."""

        with self.lock:
            response = self.latest_response
            self.latest_response = None
        return response

    def raise_if_failed(self) -> None:
        """Raise a fatal worker failure, if recorded."""

        self.worker.raise_if_failed()

    def heartbeat_once(self) -> bool:
        """Send one heartbeat and retain only its latest response."""

        try:
            response = self.agent.send_heartbeat()
            with self.lock:
                self.latest_response = response
        except XpoolDaemonError as error:
            if error.is_recoverable:
                logger.debug("registration missing device=%s", self.agent.cuda_device)
                with self.lock:
                    self.registration_missing = True
                return False
            raise AgentError(f"Agent heartbeat received unrecoverable daemon error: {error}") from error
        except XpoolClientError as error:
            if not error.is_recoverable:
                raise AgentError(f"Agent heartbeat received unrecoverable client error: {error}") from error
            logger.debug("heartbeat failed device=%s detail=%s", self.agent.cuda_device, error)
        except Exception as error:
            raise AgentError(f"Agent heartbeat failed with unexpected error: {error}") from error
        return True
