"""Generation-scoped elastic KV capacity policy."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import xpool.native
from xpool.config import XpoolConfig, get_global_config
from xpool.fabric import FabricGenerationId
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.daemon.registration import InstanceRankId, InstanceRankRegistrationState, RegistrationBook
from xpool.service.wire import KvCapacityChannelRef

__all__ = ["KvCapacityPolicy"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KvCapacityPolicy:
    """Own one Generation's shared channel and initial capacity allocation."""

    generation: FabricGenerationId
    channel: xpool.native.kv.DaemonCapacityChannel
    commands: list[xpool.native.kv.KvCapacityCommand | None]
    pressure_sequences: list[int]
    pressure_queue: deque[int]
    priority_borrower_index: int | None = None
    pool_capacity_bytes: tuple[int, ...] | None = None

    @classmethod
    def create(cls, config: XpoolConfig, generation: FabricGenerationId) -> KvCapacityPolicy:
        """Create the fixed shared-memory channel for one complete membership."""

        slot_count = len(config.instances) * config.atn_world_size
        return cls(
            generation=generation,
            channel=xpool.native.kv.DaemonCapacityChannel.create(
                pool_count=config.atn_world_size,
                group_count=slot_count,
                partition_count=slot_count,
            ),
            commands=[None] * slot_count,
            pressure_sequences=[0] * slot_count,
            pressure_queue=deque(),
        )

    @property
    def channel_ref(self) -> KvCapacityChannelRef:
        """Return the discovery value for this policy's native channel."""

        return KvCapacityChannelRef(generation=self.generation, name=self.channel.name)

    def partition(
        self,
        registrations: RegistrationBook,
        instance_id: str,
        rank: int,
    ) -> InstanceRankRegistrationState:
        """Return a retained Generation partition registration."""

        registration = registrations.instances.query(InstanceRankId(instance_id=instance_id, rank=rank))
        if registration is None:
            raise RuntimeError(f"kv capacity policy lost registration for instance {instance_id} rank {rank}")
        return registration

    def freeze_pools(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        backing_reports: list[xpool.native.kv.KvCapacityBackingReport | None],
        device_reports: list[xpool.native.kv.KvDeviceMemoryReport | None],
    ) -> None:
        """Freeze each attention GPU's immutable post-capture physical pool."""

        config = get_global_config()
        pool_capacities: list[int] = []
        pool_floor_bytes: list[int] = []
        for pool_index, device_report in enumerate(device_reports):
            if device_report is None:
                raise RuntimeError("kv capacity pool freezing requires every device-memory report")
            already_mapped_bytes = 0
            floor_bytes = 0
            for instance_index, instance_plan in enumerate(fabric.plan.instance_plans):
                partition_index = instance_index * config.atn_world_size + pool_index
                backing_report = backing_reports[partition_index]
                if backing_report is None:
                    raise RuntimeError("kv capacity pool freezing requires every partition backing report")
                profile = self.partition(registrations, instance_plan.instance_id, pool_index).kv_capacity
                already_mapped_bytes += profile.bundle_bytes * backing_report.backed_bundles
                floor_bytes += profile.bundle_bytes * profile.floor_bundles

            non_kv_used_bytes = device_report.total_bytes - device_report.free_bytes - already_mapped_bytes
            physical_pool_bytes = (
                int(device_report.total_bytes * config.atn.device_memory_utilization) - non_kv_used_bytes
            )
            if physical_pool_bytes < floor_bytes:
                raise RuntimeError(
                    f"attention device {config.atn.devices[pool_index]} cannot retain all elastic kv floors: "
                    f"pool_bytes={physical_pool_bytes} floor_bytes={floor_bytes}"
                )
            pool_capacities.append(physical_pool_bytes)
            pool_floor_bytes.append(floor_bytes)
        self.pool_capacity_bytes = tuple(pool_capacities)
        for device, capacity_bytes, floor_bytes in zip(
            config.atn.devices,
            pool_capacities,
            pool_floor_bytes,
            strict=True,
        ):
            logger.info(
                "kv capacity pool frozen generation=%s device=%s capacity_bytes=%s floor_bytes=%s",
                self.generation.format(),
                device,
                capacity_bytes,
                floor_bytes,
            )

    def initial_target(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
    ) -> int:
        """Return the common floor-first target for one TP capacity group."""

        pool_capacities = self.pool_capacity_bytes
        if pool_capacities is None:
            raise RuntimeError("kv capacity pools are not frozen")
        instance_plan = fabric.plan.instance_plans[instance_index]
        topology = instance_plan.instance_rank_topology
        candidates: list[int] = []
        for tp_rank in range(topology.atn_tp_size):
            worker_rank = dp_rank * topology.atn_tp_size + tp_rank
            local_profiles = tuple(
                self.partition(registrations, plan.instance_id, worker_rank).kv_capacity
                for plan in fabric.plan.instance_plans
            )
            profile = local_profiles[instance_index]
            floor_bytes = sum(candidate.bundle_bytes * candidate.floor_bundles for candidate in local_profiles)
            incremental_share_bytes = (pool_capacities[worker_rank] - floor_bytes) // len(fabric.plan.instance_plans)
            candidates.append(
                min(
                    profile.bundle_capacity,
                    profile.floor_bundles + incremental_share_bytes // profile.bundle_bytes,
                )
            )
        return min(candidates)

    def capacity_groups(self, fabric: FabricGenerationState) -> list[tuple[int, int, int]]:
        """Return ``(group index, instance index, DP rank)`` in stable order."""

        stride = get_global_config().atn_world_size
        return [
            (instance_index * stride + dp_rank, instance_index, dp_rank)
            for instance_index, instance_plan in enumerate(fabric.plan.instance_plans)
            for dp_rank in range(instance_plan.instance_rank_topology.atn_dp_size)
        ]

    def service_ceiling(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
    ) -> int:
        """Return the fixed and pool-physical ceiling common to a TP group."""

        pool_capacities = self.pool_capacity_bytes
        if pool_capacities is None:
            raise RuntimeError("kv capacity pools are not frozen")
        instance_plan = fabric.plan.instance_plans[instance_index]
        topology = instance_plan.instance_rank_topology
        candidates: list[int] = []
        for tp_rank in range(topology.atn_tp_size):
            worker_rank = dp_rank * topology.atn_tp_size + tp_rank
            profile = self.partition(registrations, instance_plan.instance_id, worker_rank).kv_capacity
            other_floor_bytes = 0
            for other_index, other_plan in enumerate(fabric.plan.instance_plans):
                if other_index == instance_index:
                    continue
                other_profile = self.partition(registrations, other_plan.instance_id, worker_rank).kv_capacity
                other_floor_bytes += other_profile.bundle_bytes * other_profile.floor_bundles
            candidates.append(
                min(
                    profile.bundle_capacity,
                    (pool_capacities[worker_rank] - other_floor_bytes) // profile.bundle_bytes,
                )
            )
        return min(candidates)

    def pool_free_bytes(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        backing_reports: list[xpool.native.kv.KvCapacityBackingReport | None],
    ) -> list[int]:
        """Return actual uncommitted bytes in each attention-device pool."""

        pool_capacities = self.pool_capacity_bytes
        if pool_capacities is None:
            raise RuntimeError("kv capacity pools are not frozen")
        stride = get_global_config().atn_world_size
        free = list(pool_capacities)
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            command = self.commands[group_index]
            if command is None:
                continue
            instance_plan = fabric.plan.instance_plans[instance_index]
            topology = instance_plan.instance_rank_topology
            for tp_rank in range(topology.atn_tp_size):
                worker_rank = dp_rank * topology.atn_tp_size + tp_rank
                partition_index = instance_index * stride + worker_rank
                report = backing_reports[partition_index]
                if report is None:
                    raise RuntimeError("kv capacity accounting requires every partition backing report")
                profile = self.partition(registrations, instance_plan.instance_id, worker_rank).kv_capacity
                free[worker_rank] -= max(command.target_bundles, report.backed_bundles) * profile.bundle_bytes
        return free

    def publish_initial_target(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
    ) -> bool:
        """Publish the next missing floor-first target."""

        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            instance_plan = fabric.plan.instance_plans[instance_index]
            topology = instance_plan.instance_rank_topology
            if self.commands[group_index] is not None:
                continue
            target = self.initial_target(registrations, fabric, instance_index, dp_rank)
            floor = self.partition(
                registrations,
                instance_plan.instance_id,
                dp_rank * topology.atn_tp_size,
            ).kv_capacity.floor_bundles
            command = xpool.native.kv.KvCapacityCommand(
                sequence=1,
                target_bundles=target,
                active_bundles=floor,
            )
            self.channel.publish_command(group_index, command)
            self.commands[group_index] = command
            return True
        return False

    def activate_prepared_target(
        self,
        fabric: FabricGenerationState,
        backing_reports: list[xpool.native.kv.KvCapacityBackingReport | None],
    ) -> bool:
        """Activate the next target prepared by every TP rank."""

        config = get_global_config()
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            command = self.commands[group_index]
            if command is None or command.active_bundles == command.target_bundles:
                continue
            topology = fabric.plan.instance_plans[instance_index].instance_rank_topology
            prepared = all(
                (
                    report := backing_reports[
                        instance_index * config.atn_world_size + dp_rank * topology.atn_tp_size + tp_rank
                    ]
                )
                is not None
                and report.prepared_sequence == command.sequence
                and report.backed_bundles >= command.target_bundles
                for tp_rank in range(topology.atn_tp_size)
            )
            if prepared:
                active = xpool.native.kv.KvCapacityCommand(
                    sequence=command.sequence,
                    target_bundles=command.target_bundles,
                    active_bundles=command.target_bundles,
                )
                self.channel.publish_command(group_index, active)
                self.commands[group_index] = active
                logger.info(
                    "kv capacity activated generation=%s instance=%s dp_rank=%s active_bundles=%s command_sequence=%s",
                    self.generation.format(),
                    fabric.plan.instance_plans[instance_index].instance_id,
                    dp_rank,
                    active.active_bundles,
                    active.sequence,
                )
                return True
        return False

    def consume_pressure(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        pressure_reports: list[xpool.native.kv.KvCapacityPressureReport | None],
    ) -> set[int]:
        """Consume new pressure edges and return currently pressured groups."""

        groups = self.capacity_groups(fabric)
        for group_index, instance_index, dp_rank in groups:
            report = pressure_reports[group_index]
            if report is None or report.sequence <= self.pressure_sequences[group_index]:
                continue
            self.pressure_sequences[group_index] = report.sequence
            if report.active_bundles is None:
                self.pressure_queue = deque(index for index in self.pressure_queue if index != group_index)
                if self.priority_borrower_index == group_index:
                    self.priority_borrower_index = None
                logger.info(
                    "kv capacity pressure cleared generation=%s instance=%s dp_rank=%s pressure_sequence=%s",
                    self.generation.format(),
                    fabric.plan.instance_plans[instance_index].instance_id,
                    dp_rank,
                    report.sequence,
                )
                continue
            command = self.commands[group_index]
            if command is None:
                continue
            ceiling = self.service_ceiling(registrations, fabric, instance_index, dp_rank)
            if self.priority_borrower_index == group_index and command.active_bundles >= ceiling:
                self.priority_borrower_index = None
            if command.active_bundles < ceiling and group_index not in self.pressure_queue:
                self.pressure_queue.append(group_index)
            logger.info(
                "kv capacity pressure entered generation=%s instance=%s dp_rank=%s active_bundles=%s "
                "pressure_sequence=%s",
                self.generation.format(),
                fabric.plan.instance_plans[instance_index].instance_id,
                dp_rank,
                report.active_bundles,
                report.sequence,
            )

        return {
            group_index
            for group_index, _, _ in groups
            if (report := pressure_reports[group_index]) is not None and report.active_bundles is not None
        }

    def schedule_service_transition(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        backing_reports: list[xpool.native.kv.KvCapacityBackingReport | None],
        pressure_reports: list[xpool.native.kv.KvCapacityPressureReport | None],
        pressured_groups: set[int],
    ) -> bool:
        """Publish at most one FIFO growth or reclaim transition."""

        groups = self.capacity_groups(fabric)
        locations = {group_index: (instance_index, dp_rank) for group_index, instance_index, dp_rank in groups}
        for borrower_index in tuple(self.pressure_queue):
            location = locations.get(borrower_index)
            report = pressure_reports[borrower_index]
            command = self.commands[borrower_index]
            if location is None or report is None or report.active_bundles is None or command is None:
                self.pressure_queue.remove(borrower_index)
                continue
            if command.target_bundles != command.active_bundles:
                continue
            instance_index, dp_rank = location
            ceiling = self.service_ceiling(registrations, fabric, instance_index, dp_rank)
            desired = min(report.active_bundles + 1, ceiling)
            if command.active_bundles >= desired:
                self.pressure_queue.remove(borrower_index)
                continue

            instance_plan = fabric.plan.instance_plans[instance_index]
            topology = instance_plan.instance_rank_topology
            delta = desired - command.target_bundles
            free = self.pool_free_bytes(registrations, fabric, backing_reports)
            required: dict[int, int] = {}
            for tp_rank in range(topology.atn_tp_size):
                worker_rank = dp_rank * topology.atn_tp_size + tp_rank
                profile = self.partition(registrations, instance_plan.instance_id, worker_rank).kv_capacity
                required[worker_rank] = delta * profile.bundle_bytes

            if all(free[pool_index] >= byte_count for pool_index, byte_count in required.items()):
                next_command = xpool.native.kv.KvCapacityCommand(
                    sequence=command.sequence + 1,
                    target_bundles=desired,
                    active_bundles=command.active_bundles,
                )
                self.channel.publish_command(borrower_index, next_command)
                self.commands[borrower_index] = next_command
                self.pressure_queue.remove(borrower_index)
                logger.info(
                    "kv capacity growth requested generation=%s instance=%s dp_rank=%s active_bundles=%s "
                    "target_bundles=%s command_sequence=%s",
                    self.generation.format(),
                    instance_plan.instance_id,
                    dp_rank,
                    next_command.active_bundles,
                    next_command.target_bundles,
                    next_command.sequence,
                )
                return True

            short_pools = {pool_index for pool_index, byte_count in required.items() if free[pool_index] < byte_count}

            def donor_candidates(*, allow_pressure: bool) -> list[tuple[int, int]]:
                candidates: list[tuple[int, int]] = []
                for donor_index, donor_instance_index, donor_dp_rank in groups:
                    if donor_index == borrower_index or (donor_index in pressured_groups) != allow_pressure:
                        continue
                    donor_command = self.commands[donor_index]
                    donor_plan = fabric.plan.instance_plans[donor_instance_index]
                    donor_topology = donor_plan.instance_rank_topology
                    donor_profile = self.partition(
                        registrations,
                        donor_plan.instance_id,
                        donor_dp_rank * donor_topology.atn_tp_size,
                    ).kv_capacity
                    donor_pools = {
                        donor_dp_rank * donor_topology.atn_tp_size + tp_rank
                        for tp_rank in range(donor_topology.atn_tp_size)
                    }
                    if (
                        donor_command is not None
                        and donor_command.target_bundles == donor_command.active_bundles
                        and donor_command.target_bundles > donor_profile.floor_bundles
                        and donor_pools & short_pools
                    ):
                        candidates.append((donor_command.target_bundles * donor_profile.bundle_bytes, donor_index))
                return candidates

            candidates = donor_candidates(allow_pressure=False)
            if not candidates:
                if self.priority_borrower_index is None:
                    self.priority_borrower_index = borrower_index
                if self.priority_borrower_index != borrower_index:
                    continue
                candidates = [
                    candidate
                    for candidate in donor_candidates(allow_pressure=True)
                    if candidate[1] != self.priority_borrower_index
                ]
            if not candidates:
                continue

            _, donor_index = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
            donor_instance_index, donor_dp_rank = locations[donor_index]
            donor_command = self.commands[donor_index]
            if donor_command is None:
                continue
            next_command = xpool.native.kv.KvCapacityCommand(
                sequence=donor_command.sequence + 1,
                target_bundles=donor_command.target_bundles - 1,
                active_bundles=min(donor_command.active_bundles, donor_command.target_bundles - 1),
            )
            self.channel.publish_command(donor_index, next_command)
            self.commands[donor_index] = next_command
            logger.info(
                "kv capacity reclaim requested generation=%s borrower_instance=%s borrower_dp_rank=%s "
                "donor_instance=%s donor_dp_rank=%s donor_active_bundles=%s donor_target_bundles=%s "
                "donor_pressured=%s command_sequence=%s",
                self.generation.format(),
                instance_plan.instance_id,
                dp_rank,
                fabric.plan.instance_plans[donor_instance_index].instance_id,
                donor_dp_rank,
                donor_command.active_bundles,
                next_command.target_bundles,
                donor_index in pressured_groups,
                next_command.sequence,
            )

            donor_report = pressure_reports[donor_index]
            if donor_report is not None and donor_report.active_bundles is not None:
                ceiling = self.service_ceiling(registrations, fabric, donor_instance_index, donor_dp_rank)
                if (
                    next_command.active_bundles < min(donor_report.active_bundles + 1, ceiling)
                    and donor_index not in self.pressure_queue
                ):
                    self.pressure_queue.append(donor_index)
            return True
        return False

    def step(self, registrations: RegistrationBook, fabric: FabricGenerationState) -> None:
        """Advance at most one startup or service capacity transition."""

        backing_reports = self.channel.read_backing_reports()
        if any(report is None for report in backing_reports):
            return
        device_reports = self.channel.read_device_memory()
        if any(report is None for report in device_reports):
            return
        if self.pool_capacity_bytes is None:
            self.freeze_pools(registrations, fabric, backing_reports, device_reports)

        if self.publish_initial_target(registrations, fabric):
            return
        if self.activate_prepared_target(fabric, backing_reports):
            return

        groups = self.capacity_groups(fabric)
        group_commands = tuple(
            command for group_index, _, _ in groups if (command := self.commands[group_index]) is not None
        )
        if all(command.sequence == 1 for command in group_commands) and any(
            command.active_bundles != command.target_bundles for command in group_commands
        ):
            return

        pressure_reports = self.channel.read_pressure_reports()
        pressured_groups = self.consume_pressure(registrations, fabric, pressure_reports)
        self.schedule_service_transition(
            registrations,
            fabric,
            backing_reports,
            pressure_reports,
            pressured_groups,
        )

    def close(self) -> None:
        """Close and unlink this Generation's channel."""

        self.channel.close()
