"""Shared runtime lifecycle support for xpool agents."""

from __future__ import annotations

import logging
import signal
import threading
import time
from abc import ABC, abstractmethod

import xpool.native
from xpool import bootstrap, devkit
from xpool.abi import ABI_VERSION, FfnResultCode
from xpool.config import LoopbackSite, get_global_config
from xpool.fabric import (
    FabricGenerationPhase,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FifoSchedulerPlan,
    RandomSchedulerPlan,
)
from xpool.runtime import RuntimeRole
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    FabricInvocationFailure,
    FabricParticipantReport,
    FabricProtocolFailure,
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
AGENT_FABRIC_SHUTDOWN_TIMEOUT_S = 60.0
FABRIC_REPORT_RETRY_ATTEMPTS = 3
FABRIC_REPORT_RETRY_DELAY_S = 0.5
logger = logging.getLogger(__name__)


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
        self.fabric_phase: FabricGenerationPhase | None = None
        self.participant_report: FabricParticipantReport | None = None
        self.fabric_stopped = False
        self.heartbeat_worker: AgentHeartbeat

    def join_fabric(self) -> None:
        """Join the daemon's immutable Fabric plan and report device readiness."""

        if self.fabric_plan is not None:
            return
        if get_global_config().debug.loopback.site is LoopbackSite.INSTANCE:
            return
        try:
            fabric_plan = self.client.fabric_plan()
        except (XpoolClientError, XpoolDaemonError) as error:
            if error.is_recoverable:
                return
            raise AgentError(f"Fabric plan acquisition failed: {error}") from error

        role = FabricRole.ATNAGENT if self.runtime_role is RuntimeRole.ATNAGENT else FabricRole.FFNAGENT
        placements = tuple(
            placement
            for placement in fabric_plan.pe_placements
            if placement.role is role and placement.cuda_device == self.cuda_device
        )
        if len(placements) != 1:
            raise AgentError("Fabric plan does not contain exactly one placement for this Agent")
        placement = placements[0]
        self.fabric_plan = fabric_plan
        self.report_fabric_phase(FabricParticipantPhase.JOINING)
        try:
            xpool.native.fabric.join(self.native_join_metadata(fabric_plan, placement.pe))
            self.report_fabric_phase(FabricParticipantPhase.JOINED)
            self.activate_fabric_plan()
            xpool.native.fabric.check_health()
            self.report_fabric_phase(FabricParticipantPhase.ACTIVE)
        except (RuntimeError, XpoolClientError, XpoolDaemonError, AgentError) as error:
            self.report_local_protocol_failure(f"Fabric join or activation failed: {error}")
            raise AgentError(f"Fabric join failed: {error}") from error

    def native_join_metadata(self, plan: FabricPlan, pe: int) -> xpool.native.FabricJoinMetadata:
        """Project one immutable Python plan into native join metadata."""

        if isinstance(plan.scheduler, FifoSchedulerPlan):
            scheduler = xpool.native.FfnSchedulerPolicy.fifo()
        elif isinstance(plan.scheduler, RandomSchedulerPlan):
            scheduler = xpool.native.FfnSchedulerPolicy.random(plan.scheduler.seed)
        else:
            raise AgentError("Fabric plan contains an unsupported scheduler variant")
        atnagent_count = sum(item.role is FabricRole.ATNAGENT for item in plan.pe_placements)
        return xpool.native.FabricJoinMetadata(
            uid=plan.uid.value,
            pe=pe,
            atnagent_count=atnagent_count,
            ffnagent_count=len(plan.pe_placements) - atnagent_count,
            executor_count=plan.executor_count,
            scheduler_policy=scheduler,
            models=[
                xpool.native.FabricModelMetadata(
                    max_decode_rows=model.workload.max_decode_rows,
                    max_prefill_rows=model.workload.max_prefill_rows,
                    dtype=int(model.workload.dtype),
                    hidden_size=model.workload.hidden_size,
                    atn_tp_size=model.atn_tp_size,
                    atn_dp_size=model.atn_dp_size,
                    layers=[
                        xpool.native.FabricLayerMetadata(layer_id=layer.layer_id, kind=int(layer.kind))
                        for layer in model.workload.layers
                    ],
                )
                for model in plan.models
            ],
        )

    @abstractmethod
    def activate_fabric_plan(self) -> None:
        """Activate role-local device progress before reporting readiness."""

    def shutdown_fabric(self) -> None:
        """Request and complete coordinated Fabric quiesce and native drain."""

        if self.fabric_plan is None:
            return
        self.client.request_fabric_quiesce(
            FabricQuiesceRequest(owner=self.process_ref, generation=self.fabric_plan.generation)
        )
        deadline = time.monotonic() + AGENT_FABRIC_SHUTDOWN_TIMEOUT_S
        while self.fabric_plan is not None and time.monotonic() < deadline:
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
        if self.fabric_plan is not None:
            raise AgentError("timed out waiting for coordinated Fabric shutdown")

    def handle_heartbeat_response(self, response: HeartbeatResponse) -> None:
        """Apply one daemon-authoritative snapshot on the Agent main thread."""

        if self.fabric_plan is not None and response.generation != self.fabric_plan.generation:
            raise AgentError("daemon heartbeat returned a different Fabric generation")
        self.fabric_phase = response.fabric_phase

    def poll_fabric_health(self) -> None:
        """Publish canonical native failure or reject unexpected Resident exit."""

        report = self.participant_report
        if self.fabric_plan is None or report is None or report.phase is FabricParticipantPhase.FINALIZED:
            return
        failure = xpool.native.fabric.failure()
        if failure is not None:
            payload = failure.payload
            invocation_failure = FabricInvocationFailure(
                result_code=FfnResultCode(payload.result_code),
                origin_pe=payload.origin_pe,
                model_index=payload.key.model_index,
                invocation_sequence=payload.key.invocation_sequence,
                layer_ordinal=payload.layer_ordinal,
            )
            if report.invocation_failure != invocation_failure:
                self.report_fabric_phase(report.phase, invocation_failure=invocation_failure)
            return
        if report.phase is FabricParticipantPhase.ACTIVE and self.fabric_phase in {
            FabricGenerationPhase.JOINING,
            FabricGenerationPhase.EXECUTABLE,
        }:
            try:
                xpool.native.fabric.check_health()
            except RuntimeError as error:
                self.report_local_protocol_failure(f"Fabric Resident completed unexpectedly: {error}")
                raise AgentError(f"Fabric Resident completed unexpectedly: {error}") from error

    def advance_fabric_lifecycle(self) -> None:
        """Advance local work required by the daemon-authoritative phase."""

        if self.fabric_plan is None or self.participant_report is None:
            return
        report = self.participant_report
        match self.fabric_phase:
            case FabricGenerationPhase.ABORTING:
                raise AgentError("daemon selected fail-stop Fabric abort")
            case FabricGenerationPhase.QUIESCING if report.phase is FabricParticipantPhase.ACTIVE:
                self.quiesce_fabric()
                self.report_fabric_phase(FabricParticipantPhase.QUIESCED)
            case FabricGenerationPhase.DRAINING:
                if report.phase is FabricParticipantPhase.QUIESCED:
                    xpool.native.fabric.drain_async()
                    self.report_fabric_phase(FabricParticipantPhase.DRAINING)
                elif report.phase is FabricParticipantPhase.DRAINING and not xpool.native.fabric.drain_pending():
                    self.report_fabric_phase(FabricParticipantPhase.DRAINED)
            case FabricGenerationPhase.FINALIZING if report.phase is FabricParticipantPhase.DRAINED:
                xpool.native.fabric.shutdown()
                self.report_fabric_phase(FabricParticipantPhase.FINALIZED)
            case FabricGenerationPhase.STOPPED if report.phase is FabricParticipantPhase.FINALIZED:
                self.fabric_plan = None
                self.fabric_phase = None
                self.participant_report = None
                self.fabric_stopped = True
                self.heartbeat_worker.stop()
            case _:
                return

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
        protocol_failure: FabricProtocolFailure | None = None,
    ) -> None:
        """Commit one local phase or failure enrichment through daemon 204."""

        if self.fabric_plan is None:
            raise AgentError("cannot report Fabric participant progress before retaining a plan")
        previous = self.participant_report
        if previous is None:
            if phase is not FabricParticipantPhase.JOINING:
                raise AgentError("first local Fabric participant phase must be joining")
        elif phase is not previous.phase and not previous.phase.allows(phase):
            raise AgentError(f"local Fabric participant phase cannot move from {previous.phase.value} to {phase.value}")

        placement = self.fabric_placement()
        candidate = FabricParticipantReport(
            owner=self.process_ref,
            generation=self.fabric_plan.generation,
            pe=placement.pe,
            phase=phase,
            plan_digest=self.fabric_plan.digest(),
            invocation_failure=(
                invocation_failure
                if invocation_failure is not None
                else None
                if previous is None
                else previous.invocation_failure
            ),
            protocol_failure=(
                protocol_failure
                if protocol_failure is not None
                else None
                if previous is None
                else previous.protocol_failure
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

    def report_local_protocol_failure(self, message: str) -> None:
        """Best-effort enrich the current report with one local protocol failure."""

        if self.fabric_plan is None or self.participant_report is None:
            return
        try:
            self.report_fabric_phase(
                self.participant_report.phase,
                protocol_failure=FabricProtocolFailure(message=message),
            )
        except AgentError:
            logger.exception("failed to report local Fabric protocol failure")

    def fabric_placement(self) -> FabricPePlacement:
        """Return this Agent's unique immutable Fabric PE placement."""

        if self.fabric_plan is None:
            raise AgentError("Fabric placement requires a retained plan")
        role = FabricRole.ATNAGENT if self.runtime_role is RuntimeRole.ATNAGENT else FabricRole.FFNAGENT
        placements = tuple(
            item
            for item in self.fabric_plan.pe_placements
            if item.cuda_device == self.cuda_device and item.role is role
        )
        if len(placements) != 1:
            raise AgentError("Fabric plan has no unique placement for this Agent")
        return placements[0]

    @abstractmethod
    def prepare_fabric(self) -> bool:
        """Prepare role-local resources and report whether Fabric may join."""

    @abstractmethod
    def close_role(self) -> None:
        """Release role-local resources after coordinated Fabric shutdown."""

    def run(self) -> None:
        """Run registration, role preparation, and Fabric lifecycle."""

        shutdown_requested = threading.Event()

        def request_shutdown(signal_number: int, frame: object) -> None:
            shutdown_requested.set()

        graceful_exit = False
        with (
            sighandle(signal.SIGINT, request_shutdown),
            sighandle(signal.SIGTERM, request_shutdown),
        ):
            try:
                while not self.fabric_stopped and not shutdown_requested.is_set():
                    self.heartbeat_worker.raise_if_failed()
                    if self.heartbeat_worker.consume_registration_missing():
                        self.heartbeat_worker.stop()
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
                    prepared = self.prepare_fabric()
                    if shutdown_requested.is_set():
                        break
                    if prepared:
                        self.join_fabric()
                        if shutdown_requested.is_set():
                            break
                        self.poll_fabric_health()
                        if shutdown_requested.is_set():
                            break
                        self.advance_fabric_lifecycle()
                    if not self.fabric_stopped:
                        shutdown_requested.wait(AGENT_CONTROL_INTERVAL_S)
                graceful_exit = True
            except Exception:
                logger.exception("xpool Agent failed and will exit without unsafe native cleanup")
                bail(code=1)
            finally:
                if graceful_exit:
                    try:
                        self.shutdown_fabric()
                    except Exception:
                        logger.exception("coordinated Fabric shutdown failed; skipping unsafe native cleanup")
                        bail(code=1)
                    finally:
                        self.heartbeat_worker.close()
                        try:
                            self.close_role()
                        finally:
                            self.client.close()


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
                logger.warning("Agent registration for CUDA device %s is missing", self.agent.cuda_device)
                with self.lock:
                    self.registration_missing = True
                return False
            raise AgentError(f"Agent heartbeat received unrecoverable daemon error: {error}") from error
        except XpoolClientError as error:
            if not error.is_recoverable:
                raise AgentError(f"Agent heartbeat received unrecoverable client error: {error}") from error
            logger.warning("xpool Agent heartbeat failed: %s", error)
        except Exception as error:
            raise AgentError(f"Agent heartbeat failed with unexpected error: {error}") from error
        return True
