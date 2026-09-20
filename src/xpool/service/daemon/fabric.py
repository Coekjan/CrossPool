"""Fabric generation state for the daemon control plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NoReturn

from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import (
    FabricGenerationId,
    FabricGenerationPhase,
    FabricInstancePlan,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    InstanceRankTopology,
)
from xpool.service.daemon.registration import InstanceRankId, RegistrationBook
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import (
    FabricInvocationFailure,
    FabricOwnerFailure,
    FabricParticipantReport,
)
from xpool.utils.procs import ProcUniqId

logger = logging.getLogger(__name__)

PRE_EXECUTABLE_PHASES = frozenset(
    {
        FabricGenerationPhase.PREPARING_JOIN,
        FabricGenerationPhase.JOINING,
        FabricGenerationPhase.PREPARING_EXECUTION,
        FabricGenerationPhase.ACTIVATING,
    }
)


@dataclass(frozen=True, slots=True)
class FabricMembership:
    """Immutable complete membership captured under the control-plane lock."""

    revision: int
    model_specs: tuple[ffn.FfnModelSpec, ...]
    instance_plans: tuple[FabricInstancePlan, ...]
    ffnagent_free_memory_bytes: tuple[int, ...]
    pe_placements: tuple[FabricPePlacement, ...]
    agent_owners: tuple[tuple[int, ProcUniqId], ...]
    instance_owners: tuple[tuple[InstanceRankId, ProcUniqId], ...]

    @classmethod
    def capture(
        cls,
        config: XpoolConfig,
        registrations: RegistrationBook,
        revision: int,
    ) -> FabricMembership | None:
        """Capture one complete, internally consistent configured membership."""

        instance_by_rank = {registration.instance: registration for registration in registrations.instances.values()}
        atnagent_by_device = {
            registration.cuda_device: registration for registration in registrations.atnagents.values()
        }
        ffnagent_by_device = {
            registration.cuda_device: registration for registration in registrations.ffnagents.values()
        }
        if any(device not in atnagent_by_device for device in config.atn.devices):
            return None
        if any(device not in ffnagent_by_device for device in config.ffn.devices):
            return None

        model_specs = registrations.ffn_model_specs
        if model_specs is None:
            return None
        instance_plans: list[FabricInstancePlan] = []
        instance_owners: list[tuple[InstanceRankId, ProcUniqId]] = []
        for instance in config.instances:
            rank_zero = instance_by_rank.get(InstanceRankId(instance_id=instance.id, rank=0))
            if rank_zero is None:
                return None
            rank_count = rank_zero.transport.atn_tp_size * rank_zero.transport.atn_dp_size
            ranks = tuple(
                instance_by_rank.get(InstanceRankId(instance_id=instance.id, rank=rank)) for rank in range(rank_count)
            )
            if any(registration is None for registration in ranks):
                return None
            complete_ranks = tuple(registration for registration in ranks if registration is not None)
            ffn_profile = complete_ranks[0].ffn_profile
            if any(registration.ffn_profile != ffn_profile for registration in complete_ranks[1:]):
                raise XpoolDaemonError("conflict", "FFN ffn_profile disagrees across instance ranks")
            atn_tp_size = complete_ranks[0].transport.atn_tp_size
            atn_dp_size = complete_ranks[0].transport.atn_dp_size
            for dp_rank in range(atn_dp_size):
                capacity_group = tuple(
                    registration for registration in complete_ranks if registration.transport.atn_dp_rank == dp_rank
                )
                kv_capacity = capacity_group[0].kv_capacity
                if any(registration.kv_capacity != kv_capacity for registration in capacity_group[1:]):
                    raise XpoolDaemonError("conflict", "kv capacity geometry disagrees within a capacity group")
            expected_coordinates = {
                (tp_rank, dp_rank) for dp_rank in range(atn_dp_size) for tp_rank in range(atn_tp_size)
            }
            coordinates = {
                (registration.transport.atn_tp_rank, registration.transport.atn_dp_rank)
                for registration in complete_ranks
            }
            if coordinates != expected_coordinates:
                raise XpoolDaemonError("conflict", "instance ranks do not cover each TP-fastest coordinate once")
            instance_plans.append(
                FabricInstancePlan(
                    instance_id=instance.id,
                    ffn_profile=ffn_profile,
                    instance_rank_topology=InstanceRankTopology(
                        atn_tp_size=atn_tp_size,
                        atn_dp_size=atn_dp_size,
                        atnagent_indices=tuple(range(rank_count)),
                    ),
                )
            )
            instance_owners.extend((registration.instance, registration.proc) for registration in complete_ranks)

        placements = tuple(
            FabricPePlacement(role=FabricRole.ATNAGENT, cuda_device=cuda_device) for cuda_device in config.atn.devices
        ) + tuple(
            FabricPePlacement(
                role=FabricRole.FFNAGENT,
                cuda_device=cuda_device,
            )
            for cuda_device in config.ffn.devices
        )
        agent_owners = tuple(
            (
                pe,
                atnagent_by_device[placement.cuda_device].proc
                if placement.role is FabricRole.ATNAGENT
                else ffnagent_by_device[placement.cuda_device].proc,
            )
            for pe, placement in enumerate(placements)
        )
        return cls(
            revision=revision,
            model_specs=model_specs,
            instance_plans=tuple(instance_plans),
            ffnagent_free_memory_bytes=tuple(
                ffnagent_by_device[device].cuda_free_memory_bytes for device in config.ffn.devices
            ),
            pe_placements=placements,
            agent_owners=agent_owners,
            instance_owners=tuple(instance_owners),
        )

    def owners(self) -> tuple[ProcUniqId, ...]:
        """Return every distinct process owner in deterministic membership order."""

        return tuple(dict.fromkeys(owner for _, owner in (*self.agent_owners, *self.instance_owners)))


@dataclass(slots=True)
class FabricGenerationState:
    """Authoritative state for one immutable Fabric generation."""

    plan: FabricPlan
    phase: FabricGenerationPhase
    phase_started_at: float
    invocation_failure: FabricInvocationFailure | None
    owner_failure: FabricOwnerFailure | None
    control_failure: str | None
    agent_owners: dict[int, ProcUniqId]
    instance_owners: dict[InstanceRankId, ProcUniqId]
    participants: dict[int, FabricParticipantReport] = field(default_factory=dict)
    initialized_instances: dict[InstanceRankId, ProcUniqId] = field(default_factory=dict)

    def record_invocation_failure(self, failure: FabricInvocationFailure) -> None:
        """Retain the first canonical invocation failure."""

        if self.invocation_failure is None:
            self.invocation_failure = failure
        elif self.invocation_failure != failure:
            self.record_control_failure("participants reported conflicting canonical invocation failures")

    def record_owner_failure(self, failure: FabricOwnerFailure) -> None:
        """Retain the first daemon-observed owner failure."""

        if self.owner_failure is None:
            self.owner_failure = failure

    def record_control_failure(self, failure: str) -> None:
        """Retain the first nonempty control or lifecycle diagnostic."""

        if not failure:
            raise ValueError("Fabric control failure must be nonempty")
        if self.control_failure is None:
            self.control_failure = failure

    def transition(self, phase: FabricGenerationPhase, *, now: float) -> None:
        """Move through one exact generation edge without resetting retries."""

        if self.phase is phase:
            return
        if not self.phase.allows(phase):
            raise XpoolDaemonError(
                "conflict",
                f"Fabric generation cannot transition from {self.phase.value} to {phase.value}",
            )
        if self.phase in PRE_EXECUTABLE_PHASES:
            logger.info(
                "generation phase changed generation=%s from=%s to=%s",
                self.plan.generation.format(),
                self.phase.value,
                phase.value,
            )
        else:
            logger.info("generation phase changed from=%s to=%s", self.phase.value, phase.value)
        self.phase = phase
        self.phase_started_at = now


class FabricController:
    """Caller-synchronized owner of the installed Fabric generation."""

    __slots__ = ("generation",)

    def __init__(self) -> None:
        """Create a controller without an installed generation."""

        self.generation: FabricGenerationState | None = None

    def plan(self) -> FabricPlan | None:
        """Return the installed immutable plan, if any."""

        return None if self.generation is None else self.generation.plan

    def require_plan(self) -> FabricPlan:
        """Return the installed plan or report incomplete membership."""

        plan = self.plan()
        if plan is None:
            raise XpoolDaemonError("not_ready", "fabric plan requires every configured owner and ffn_profile")
        return plan

    def install(self, generation: FabricGenerationState) -> FabricPlan:
        """Install a generation when no generation is retained."""

        if self.generation is not None:
            return self.generation.plan
        self.generation = generation
        return generation.plan

    def quiesce(self, *, now: float) -> None:
        """Stop admission or abort a generation whose join is incomplete."""

        generation = self.generation
        if generation is None:
            raise XpoolDaemonError("conflict", "fabric quiesce requires a retained generation")
        match generation.phase:
            case (
                FabricGenerationPhase.PREPARING_JOIN
                | FabricGenerationPhase.JOINING
                | FabricGenerationPhase.PREPARING_EXECUTION
                | FabricGenerationPhase.ACTIVATING
            ):
                generation.transition(FabricGenerationPhase.ABORTING, now=now)
            case FabricGenerationPhase.EXECUTABLE:
                generation.transition(FabricGenerationPhase.QUIESCING, now=now)
            case (
                FabricGenerationPhase.QUIESCING
                | FabricGenerationPhase.DRAINING
                | FabricGenerationPhase.FINALIZING
                | FabricGenerationPhase.ABORTING
                | FabricGenerationPhase.STOPPED
            ):
                return

    def abort(self, *, now: float) -> None:
        """Select fail-stop cleanup for any live generation phase."""

        generation = self.generation
        if generation is None or generation.phase in {
            FabricGenerationPhase.ABORTING,
            FabricGenerationPhase.STOPPED,
        }:
            return
        generation.transition(FabricGenerationPhase.ABORTING, now=now)

    def instance_departed(
        self,
        instance: InstanceRankId,
        failure: FabricOwnerFailure,
        *,
        termination_requested: bool,
        now: float,
    ) -> None:
        """Remove initialization state and select cleanup for an Instance-rank loss."""

        generation = self.generation
        if generation is None:
            return
        generation.initialized_instances.pop(instance, None)
        if generation.phase is FabricGenerationPhase.QUIESCING and termination_requested:
            return
        if generation.phase in {
            FabricGenerationPhase.DRAINING,
            FabricGenerationPhase.FINALIZING,
            FabricGenerationPhase.ABORTING,
            FabricGenerationPhase.STOPPED,
        }:
            return
        generation.record_owner_failure(failure)
        self.quiesce(now=now)

    def record_initialized(
        self,
        instance: InstanceRankId,
        owner: ProcUniqId,
        *,
        generation: FabricGenerationId,
    ) -> None:
        """Commit one owner-validated Instance-rank initialization barrier."""

        current = self.generation
        if current is None:
            raise XpoolDaemonError("not_ready", "fabric generation retired during initialization")
        if generation != current.plan.generation:
            raise XpoolDaemonError("conflict", "instance initialized generation differs")
        if current.phase is not FabricGenerationPhase.EXECUTABLE:
            raise XpoolDaemonError("conflict", "instance initialization requires executable Fabric")
        expected_owner = current.instance_owners.get(instance)
        if expected_owner != owner:
            raise XpoolDaemonError("conflict", "instance initialization owner differs from Fabric plan")
        current.initialized_instances[instance] = owner

    def record_participant(self, report: FabricParticipantReport, *, now: float) -> None:
        """Validate, commit, and apply one owner-validated participant report."""

        generation = self.generation
        if generation is None:
            raise XpoolDaemonError("conflict", "participant reported a retired Fabric generation")
        previous = generation.participants.get(report.pe)
        if previous == report:
            return
        if report.generation != generation.plan.generation:
            self.reject_report("participant reported a different Fabric generation", now=now)
        if report.pe < 0 or report.pe >= len(generation.plan.pe_placements):
            self.reject_report("participant reported an unknown Fabric PE", now=now)
        if generation.phase in {FabricGenerationPhase.ABORTING, FabricGenerationPhase.STOPPED}:
            raise XpoolDaemonError("conflict", "terminal Fabric generation accepts only an exact report retry")

        if previous is None:
            if report.phase is not FabricParticipantPhase.JOIN_READY:
                self.reject_report("first participant report must be join_ready", now=now)
        elif report.phase is not previous.phase and not previous.phase.allows(report.phase):
            self.reject_report(
                f"participant phase cannot transition from {previous.phase.value} to {report.phase.value}",
                now=now,
            )
        self.validate_enrichment(previous, report, now=now)
        allowed_phases = {
            FabricGenerationPhase.PREPARING_JOIN: {FabricParticipantPhase.JOIN_READY},
            FabricGenerationPhase.JOINING: {
                FabricParticipantPhase.JOINING,
                FabricParticipantPhase.JOINED,
            },
            FabricGenerationPhase.PREPARING_EXECUTION: {FabricParticipantPhase.EXECUTION_READY},
            FabricGenerationPhase.ACTIVATING: {FabricParticipantPhase.ACTIVE},
            FabricGenerationPhase.QUIESCING: {FabricParticipantPhase.QUIESCED},
            FabricGenerationPhase.DRAINING: {
                FabricParticipantPhase.DRAINING,
                FabricParticipantPhase.DRAINED,
            },
            FabricGenerationPhase.FINALIZING: {FabricParticipantPhase.FINALIZED},
        }.get(generation.phase, set())
        phase_changed = previous is None or report.phase is not previous.phase
        if phase_changed and report.phase not in allowed_phases:
            self.reject_report(
                f"participant phase {report.phase.value} is invalid while generation is {generation.phase.value}",
                now=now,
            )

        generation.participants[report.pe] = report
        if report.invocation_failure is not None:
            generation.record_invocation_failure(report.invocation_failure)
        if report.control_failure is not None:
            generation.record_control_failure(report.control_failure)
        if generation.phase in PRE_EXECUTABLE_PHASES:
            logger.debug(
                "participant acknowledged pe=%s phase=%s generation=%s",
                report.pe,
                report.phase.value,
                report.generation.format(),
            )
        else:
            logger.debug("participant acknowledged pe=%s phase=%s", report.pe, report.phase.value)

        if generation.control_failure is not None:
            self.abort(now=now)
            return
        if generation.invocation_failure is not None and generation.phase not in {
            FabricGenerationPhase.QUIESCING,
            FabricGenerationPhase.DRAINING,
            FabricGenerationPhase.FINALIZING,
        }:
            self.quiesce(now=now)
            return
        self.advance_barrier(now=now)

    def validate_enrichment(
        self,
        previous: FabricParticipantReport | None,
        report: FabricParticipantReport,
        *,
        now: float,
    ) -> None:
        """Require failure fields to be absent-to-present or equal repeats."""

        if previous is None:
            return
        for name, old, new in (
            ("invocation_failure", previous.invocation_failure, report.invocation_failure),
            ("control_failure", previous.control_failure, report.control_failure),
        ):
            if old is not None and new != old:
                self.reject_report(f"participant cannot clear or replace {name}", now=now)

    def reject_report(self, message: str, *, now: float) -> NoReturn:
        """Record one report-protocol failure, select abort, and reject it."""

        generation = self.generation
        if generation is not None:
            generation.record_control_failure(message)
            self.abort(now=now)
        raise XpoolDaemonError("conflict", message)

    def advance_barrier(self, *, now: float) -> None:
        """Advance one exact all-participant generation barrier when complete."""

        generation = self.generation
        if generation is None:
            return
        expected_pes = set(range(len(generation.plan.pe_placements)))
        if set(generation.participants) != expected_pes:
            return
        phases = {report.phase for report in generation.participants.values()}
        barrier = {
            FabricGenerationPhase.PREPARING_JOIN: (
                FabricParticipantPhase.JOIN_READY,
                FabricGenerationPhase.JOINING,
            ),
            FabricGenerationPhase.JOINING: (
                FabricParticipantPhase.JOINED,
                FabricGenerationPhase.PREPARING_EXECUTION,
            ),
            FabricGenerationPhase.PREPARING_EXECUTION: (
                FabricParticipantPhase.EXECUTION_READY,
                FabricGenerationPhase.ACTIVATING,
            ),
            FabricGenerationPhase.ACTIVATING: (
                FabricParticipantPhase.ACTIVE,
                FabricGenerationPhase.EXECUTABLE,
            ),
            FabricGenerationPhase.QUIESCING: (
                FabricParticipantPhase.QUIESCED,
                FabricGenerationPhase.DRAINING,
            ),
            FabricGenerationPhase.DRAINING: (
                FabricParticipantPhase.DRAINED,
                FabricGenerationPhase.FINALIZING,
            ),
            FabricGenerationPhase.FINALIZING: (
                FabricParticipantPhase.FINALIZED,
                FabricGenerationPhase.STOPPED,
            ),
        }.get(generation.phase)
        if barrier is not None and phases == {barrier[0]}:
            generation.transition(barrier[1], now=now)
