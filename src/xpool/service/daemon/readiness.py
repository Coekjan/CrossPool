"""Immutable readiness projection for the daemon control plane."""

from __future__ import annotations

from dataclasses import dataclass

from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationPhase
from xpool.service.daemon.fabric import FabricController
from xpool.service.daemon.registration import InstanceRankId, RegistrationBook
from xpool.service.daemon.transport import TransportBroker
from xpool.service.wire import (
    ControlPlaneWarning,
    ReadinessAtnAgent,
    ReadinessFfnAgent,
    ReadinessInstanceRank,
    ReadinessSnapshot,
    ReadinessStatus,
)


@dataclass(frozen=True, slots=True)
class ControlPlaneProjection:
    """Self-contained wire projection captured under the domain lock."""

    readiness: ReadinessSnapshot
    warnings: tuple[ControlPlaneWarning, ...]

    @classmethod
    def capture(
        cls,
        *,
        config: XpoolConfig,
        now: float,
        mps_online: bool,
        registrations: RegistrationBook,
        transport: TransportBroker,
        fabric: FabricController,
    ) -> ControlPlaneProjection:
        """Copy current registration, Transport, and Fabric facts into wire values."""

        # Process projection: copy every configured role slot and its cached
        # liveness status without retaining registration objects.
        atnagent_by_device = {
            registration.cuda_device: registration for registration in registrations.atnagents.values()
        }
        ffnagent_by_device = {
            registration.cuda_device: registration for registration in registrations.ffnagents.values()
        }
        instance_by_id = {registration.instance: registration for registration in registrations.instances.values()}
        generation = fabric.generation

        atnagents = [
            ReadinessAtnAgent(
                pid=None if registration is None else registration.proc.pid,
                cuda_device=agent.cuda_device,
                status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
            )
            for agent in config.atnagents
            for registration in (atnagent_by_device.get(agent.cuda_device),)
        ]
        ffnagents = [
            ReadinessFfnAgent(
                pid=None if registration is None else registration.proc.pid,
                cuda_device=agent.cuda_device,
                status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
            )
            for agent in config.ffnagents
            for registration in (ffnagent_by_device.get(agent.cuda_device),)
        ]
        instance_slots = (
            tuple(
                (instance.id, rank, cuda_device)
                for instance in config.instances
                for rank, cuda_device in enumerate(config.devices.atn_cuda_devices)
            )
            if generation is None
            else tuple(
                (
                    instance_plan.instance_id,
                    rank,
                    config.devices.atn_cuda_devices[atnagent_index],
                )
                for instance_plan in generation.plan.instance_plans
                for rank, atnagent_index in enumerate(instance_plan.instance_rank_topology.atnagent_indices)
            )
        )
        instances = [
            ReadinessInstanceRank(
                pid=None if registration is None else registration.proc.pid,
                instance_id=instance_id,
                cuda_device=cuda_device,
                rank=rank,
                status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
            )
            for instance_id, rank, cuda_device in instance_slots
            for registration in (instance_by_id.get(InstanceRankId(instance_id=instance_id, rank=rank)),)
        ]

        # Warning projection: derive heartbeat and Transport-quiesce diagnostics
        # from the same role snapshot used by readiness.
        quiescing_devices = {
            cuda_device for cuda_device in config.devices.atn_cuda_devices if transport.is_quiescing(cuda_device)
        }
        warnings: list[ControlPlaneWarning] = []
        for entry in atnagents:
            if entry.status is not ReadinessStatus.ONLINE:
                warnings.append(
                    ControlPlaneWarning(
                        kind="stale_atnagent",
                        cuda_device=entry.cuda_device,
                        message=f"AtnAgent on CUDA device {entry.cuda_device} is not heartbeating",
                    )
                )
            elif entry.cuda_device in quiescing_devices:
                warnings.append(
                    ControlPlaneWarning(
                        kind="quiescing_atnagent",
                        cuda_device=entry.cuda_device,
                        message=f"AtnAgent on CUDA device {entry.cuda_device} is quiescing transport leases",
                    )
                )
        warnings.extend(
            ControlPlaneWarning(
                kind="stale_ffnagent",
                cuda_device=entry.cuda_device,
                message=f"FfnAgent on CUDA device {entry.cuda_device} is not heartbeating",
            )
            for entry in ffnagents
            if entry.status is not ReadinessStatus.ONLINE
        )
        warnings.extend(
            ControlPlaneWarning(
                kind="stale_instance",
                cuda_device=entry.cuda_device,
                message=(
                    f"instance {entry.instance_id} rank {entry.rank} on CUDA device "
                    f"{entry.cuda_device} is not heartbeating"
                ),
            )
            for entry in instances
            if entry.status is not ReadinessStatus.ONLINE
        )

        # Data-plane projection: require exact Transport geometry and the
        # complete Fabric initialization barrier for every configured rank.
        transport_ready = all(
            (registration := instance_by_id.get(instance_id)) is not None
            and atnagent_by_device.get(cuda_device) is not None
            and cuda_device not in quiescing_devices
            and (publication := transport.published_for(cuda_device, instance_id)) is not None
            and publication.transport == registration.transport
            for configured_instance_id, rank, cuda_device in instance_slots
            for instance_id in (InstanceRankId(instance_id=configured_instance_id, rank=rank),)
        )
        expected_initialized = {
            InstanceRankId(instance_id=instance_id, rank=rank) for instance_id, rank, _ in instance_slots
        }
        instances_initialized = generation is not None and set(generation.initialized_instances) == expected_initialized
        invocation_failure = None if generation is None else generation.invocation_failure
        owner_failure = None if generation is None else generation.owner_failure
        control_failure = None if generation is None else generation.control_failure
        processes_ready = all(entry.status is ReadinessStatus.ONLINE for entry in (*atnagents, *ffnagents, *instances))
        fabric_executable = generation is not None and generation.phase is FabricGenerationPhase.EXECUTABLE
        ready = (
            processes_ready
            and transport_ready
            and fabric_executable
            and instances_initialized
            and mps_online
            and invocation_failure is None
            and owner_failure is None
            and control_failure is None
        )
        # Wire materialization: copy terminal failure and lifecycle facts into a
        # self-contained response that can outlive the domain lock.
        readiness = ReadinessSnapshot(
            ready=ready,
            generation=None if generation is None else generation.plan.generation,
            fabric_phase=None if generation is None else generation.phase,
            fabric_invocation_failure=invocation_failure,
            fabric_owner_failure=owner_failure,
            fabric_control_failure=control_failure,
            transport_ready=transport_ready,
            instances_initialized=instances_initialized,
            mps_status=ReadinessStatus.ONLINE if mps_online else ReadinessStatus.OFFLINE,
            cuda_devices=config.cuda_devices,
            atnagents=atnagents,
            ffnagents=ffnagents,
            instances=instances,
        )
        return cls(readiness=readiness, warnings=tuple(warnings))
