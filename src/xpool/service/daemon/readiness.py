"""Immutable readiness projection for the daemon control plane."""

from __future__ import annotations

from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationPhase
from xpool.service.daemon.fabric import FabricController
from xpool.service.daemon.registration import InstanceRankId, RegistrationBook
from xpool.service.daemon.transport import TransportBroker
from xpool.service.wire import (
    ReadinessAtnAgent,
    ReadinessFfnAgent,
    ReadinessInstanceRank,
    ReadinessSnapshot,
    ReadinessStatus,
)


def capture_readiness(
    *,
    config: XpoolConfig,
    now: float,
    mps_online: bool | None,
    registrations: RegistrationBook,
    transport: TransportBroker,
    fabric: FabricController,
) -> ReadinessSnapshot:
    """Project current daemon state while the caller holds the domain lock."""

    # Process projection: copy every configured role slot and its cached
    # liveness status without retaining registration objects.
    generation = fabric.generation

    atnagents = [
        ReadinessAtnAgent(
            pid=None if registration is None else registration.proc.pid,
            device=agent.device,
            status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
        )
        for agent in config.atnagents
        for registration in (registrations.atnagents.query(agent.device),)
    ]
    ffnagents = [
        ReadinessFfnAgent(
            pid=None if registration is None else registration.proc.pid,
            device=agent.device,
            status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
        )
        for agent in config.ffnagents
        for registration in (registrations.ffnagents.query(agent.device),)
    ]
    instance_slots = (
        tuple(
            (instance.model_id, rank, device)
            for instance in config.instances
            for rank, device in enumerate(config.atn.devices)
        )
        if generation is None
        else tuple(
            (
                instance_plan.model_id,
                rank,
                config.atn.devices[atnagent_index],
            )
            for instance_plan in generation.plan.instance_plans
            for rank, atnagent_index in enumerate(instance_plan.instance_rank_topology.atnagent_indices)
        )
    )
    instances = [
        ReadinessInstanceRank(
            pid=None if registration is None else registration.proc.pid,
            model_id=model_id,
            device=device,
            rank=rank,
            status=ReadinessStatus.OFFLINE if registration is None else registration.readiness_status(now),
        )
        for model_id, rank, device in instance_slots
        for registration in (registrations.instances.query(InstanceRankId(model_id=model_id, rank=rank)),)
    ]

    quiescing_devices = {device for device in config.atn.devices if transport.is_quiescing(device)}

    # Data-plane projection: require exact Transport geometry and the
    # complete Fabric initialization barrier for every configured rank.
    transport_ready = all(
        (registration := registrations.instances.query(instance_rank)) is not None
        and registrations.atnagents.query(device) is not None
        and device not in quiescing_devices
        and (publication := transport.published_for(device, instance_rank)) is not None
        and publication.transport == registration.transport
        for model_id, rank, device in instance_slots
        for instance_rank in (InstanceRankId(model_id=model_id, rank=rank),)
    )
    instances_initialized = (
        generation is not None and generation.initialized_instances == generation.instance_owners.keys()
    )
    invocation_failure = None if generation is None else generation.invocation_failure
    owner_failure = None if generation is None else generation.owner_failure
    control_failure = generation.control_failure if generation is not None else None
    if fabric.planning_failure is not None:
        control_failure = fabric.planning_failure.message
    mps_status: ReadinessStatus | None = None
    if mps_online is not None:
        mps_status = ReadinessStatus.ONLINE if mps_online else ReadinessStatus.OFFLINE
    processes_ready = all(entry.status is ReadinessStatus.ONLINE for entry in (*atnagents, *ffnagents, *instances))
    fabric_executable = generation is not None and generation.phase is FabricGenerationPhase.EXECUTABLE
    ready = (
        processes_ready
        and transport_ready
        and fabric_executable
        and instances_initialized
        and mps_online is True
        and invocation_failure is None
        and owner_failure is None
        and control_failure is None
    )
    # Wire materialization: copy terminal failure and lifecycle facts into a
    # self-contained response that can outlive the domain lock.
    return ReadinessSnapshot(
        ready=ready,
        generation=None if generation is None else generation.plan.generation,
        fabric_phase=None if generation is None else generation.phase,
        fabric_invocation_failure=invocation_failure,
        fabric_owner_failure=owner_failure,
        fabric_control_failure=control_failure,
        transport_ready=transport_ready,
        instances_initialized=instances_initialized,
        mps_status=mps_status,
        devices=config.devices,
        atnagents=atnagents,
        ffnagents=ffnagents,
        instances=instances,
    )
