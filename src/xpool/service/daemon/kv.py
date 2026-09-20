"""Generation-scoped elastic KV capacity policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter_ns

import xpool.native
from xpool.config import XpoolConfig, get_global_config
from xpool.fabric import FabricGenerationId
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.daemon.registration import InstanceRankId, InstanceRankRegistrationState, RegistrationBook
from xpool.service.wire import KvCapacityPartitionProfile, KvControlChannelRef

__all__ = ["KvCapacityPolicy"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapacityOperation:
    """One outstanding immutable group capacity operation."""

    command: xpool.native.kv.KvCapacityCommand
    start_bundles: int


@dataclass(slots=True)
class FundingAttempt:
    """One borrower's serialized donor negotiation."""

    borrower_index: int
    evaluated_sequence: int
    target_bundles: int
    deadline_monotonic_ns: int
    attempted_donors: set[int] = field(default_factory=set)
    donor_index: int | None = None


@dataclass(slots=True)
class KvCapacityPolicy:
    """Own one Generation's shared channel and elastic capacity accounting."""

    generation: FabricGenerationId
    channel: xpool.native.kv.DaemonControlChannel
    command_sequences: list[int] = field(init=False)
    applied_sequences: list[int] = field(init=False)
    active_bundles: list[int | None] = field(init=False)
    operations: list[CapacityOperation | None] = field(init=False)
    demands: list[xpool.native.kv.KvCapacityDemand | None] = field(init=False)
    completed_grant_orders: list[int | None] = field(init=False)
    next_grant_order: int = 0
    service_ceilings: list[int | None] = field(init=False)
    funding_attempt: FundingAttempt | None = None
    pool_capacity_bytes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        slot_count = len(get_global_config().instances) * get_global_config().atn_world_size
        self.command_sequences = [0] * slot_count
        self.applied_sequences = [0] * slot_count
        self.active_bundles = [None] * slot_count
        self.operations = [None] * slot_count
        self.demands = [None] * slot_count
        self.completed_grant_orders = [None] * slot_count
        self.service_ceilings = [None] * slot_count

    @classmethod
    def create(cls, config: XpoolConfig, generation: FabricGenerationId) -> KvCapacityPolicy:
        """Create the fixed shared-memory channel for one complete membership."""

        slot_count = len(config.instances) * config.atn_world_size
        return cls(
            generation=generation,
            channel=xpool.native.kv.DaemonControlChannel.create(
                pool_count=config.atn_world_size,
                group_count=slot_count,
                partition_count=slot_count,
            ),
        )

    @property
    def channel_ref(self) -> KvControlChannelRef:
        """Return the discovery value for this policy's native channel."""

        return KvControlChannelRef(generation=self.generation, name=self.channel.name)

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

    def capacity_groups(self, fabric: FabricGenerationState) -> list[tuple[int, int, int]]:
        """Return ``(group index, instance index, DP rank)`` in stable order."""

        stride = get_global_config().atn_world_size
        return [
            (instance_index * stride + dp_rank, instance_index, dp_rank)
            for instance_index, instance_plan in enumerate(fabric.plan.instance_plans)
            for dp_rank in range(instance_plan.instance_rank_topology.atn_dp_size)
        ]

    def group_profiles(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
    ) -> tuple[tuple[int, KvCapacityPartitionProfile], ...]:
        """Return one Capacity Group's physical-pool profiles in TP-rank order."""

        instance_plan = fabric.plan.instance_plans[instance_index]
        topology = instance_plan.instance_rank_topology
        start_rank = dp_rank * topology.atn_tp_size
        return tuple(
            (
                worker_rank,
                self.partition(registrations, instance_plan.instance_id, worker_rank).kv_capacity,
            )
            for worker_rank in range(start_rank, start_rank + topology.atn_tp_size)
        )

    def group_partition_indices(
        self,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
    ) -> tuple[int, ...]:
        """Return one Capacity Group's partition slots in TP-rank order."""

        topology = fabric.plan.instance_plans[instance_index].instance_rank_topology
        row = instance_index * get_global_config().atn_world_size
        return tuple(row + dp_rank * topology.atn_tp_size + tp_rank for tp_rank in range(topology.atn_tp_size))

    def freeze_pools(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        initial_backing: list[int | None],
        device_reports: list[xpool.native.kv.KvDeviceMemoryReport | None],
    ) -> None:
        """Freeze physical pools, group floors, and immutable service ceilings."""

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
                backed_bundles = initial_backing[partition_index]
                if backed_bundles is None:
                    raise RuntimeError("kv capacity pool freezing requires every partition backing report")
                profile = self.partition(registrations, instance_plan.instance_id, pool_index).kv_capacity
                already_mapped_bytes += profile.bundle_bytes * backed_bundles
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

        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            profiles = self.group_profiles(registrations, fabric, instance_index, dp_rank)
            floors = {profile.floor_bundles for _, profile in profiles}
            backing = {
                initial_backing[index] for index in self.group_partition_indices(fabric, instance_index, dp_rank)
            }
            if len(floors) != 1 or backing != floors:
                raise RuntimeError("kv capacity group ranks did not publish one common configured floor")
            self.active_bundles[group_index] = floors.pop()
            ceiling = self.service_ceiling(registrations, fabric, instance_index, dp_rank)
            self.service_ceilings[group_index] = ceiling
            self.channel.publish_service_ceiling(group_index, ceiling)

        for device, capacity_bytes, floor_bytes in zip(
            config.atn.devices,
            pool_capacities,
            pool_floor_bytes,
            strict=True,
        ):
            logger.info(
                "kv pool frozen device=%s capacity_bytes=%s floor_bytes=%s",
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
        """Return the common floor-first target for one TP Capacity Group."""

        pool_capacities = self.pool_capacity_bytes
        if pool_capacities is None:
            raise RuntimeError("kv capacity pools are not frozen")
        candidates: list[int] = []
        for pool_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank):
            local_profiles = tuple(
                self.partition(registrations, plan.instance_id, pool_index).kv_capacity
                for plan in fabric.plan.instance_plans
            )
            floor_bytes = sum(candidate.bundle_bytes * candidate.floor_bundles for candidate in local_profiles)
            incremental_share_bytes = (pool_capacities[pool_index] - floor_bytes) // len(fabric.plan.instance_plans)
            candidates.append(
                min(
                    profile.bundle_capacity,
                    profile.floor_bundles + incremental_share_bytes // profile.bundle_bytes,
                )
            )
        return min(candidates)

    def service_ceiling(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
    ) -> int:
        """Return the fixed physical ceiling common to one TP Capacity Group."""

        pool_capacities = self.pool_capacity_bytes
        if pool_capacities is None:
            raise RuntimeError("kv capacity pools are not frozen")
        candidates: list[int] = []
        for pool_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank):
            other_floor_bytes = sum(
                candidate.bundle_bytes * candidate.floor_bundles
                for other_index, plan in enumerate(fabric.plan.instance_plans)
                if other_index != instance_index
                for candidate in (self.partition(registrations, plan.instance_id, pool_index).kv_capacity,)
            )
            candidates.append(
                min(
                    profile.bundle_capacity,
                    (pool_capacities[pool_index] - other_floor_bytes) // profile.bundle_bytes,
                )
            )
        return min(candidates)

    def pool_free_bytes(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
    ) -> list[int]:
        """Return unassigned bytes after conservative in-flight charging."""

        if self.pool_capacity_bytes is None:
            raise RuntimeError("kv capacity pools are not frozen")
        free = list(self.pool_capacity_bytes)
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            active = self.active_bundles[group_index]
            if active is None:
                continue
            operation = self.operations[group_index]
            charged_bundles = max(active, operation.command.target_bundles) if operation is not None else active
            for pool_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank):
                free[pool_index] -= charged_bundles * profile.bundle_bytes
        if any(byte_count < 0 for byte_count in free):
            raise RuntimeError("kv capacity accounting exceeded a physical pool")
        return free

    def growth_bytes(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
        start_bundles: int,
        target_bundles: int,
    ) -> dict[int, int]:
        """Return additional per-pool bytes required by one group growth."""

        return {
            pool_index: (target_bundles - start_bundles) * profile.bundle_bytes
            for pool_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank)
        }

    def publish_operation(self, group_index: int, target_bundles: int) -> None:
        """Publish one immutable operation for an idle Capacity Group."""

        start_bundles = self.active_bundles[group_index]
        if start_bundles is None or self.operations[group_index] is not None:
            raise RuntimeError("kv capacity operation requires one initialized idle group")
        sequence = self.command_sequences[group_index] + 1
        command = xpool.native.kv.KvCapacityCommand(sequence=sequence, target_bundles=target_bundles)
        self.channel.publish_command(group_index, command)
        self.command_sequences[group_index] = sequence
        self.operations[group_index] = CapacityOperation(command=command, start_bundles=start_bundles)

    def retire_operations(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        completions: list[xpool.native.kv.KvCapacityCompletion | None],
    ) -> list[tuple[int, CapacityOperation]]:
        """Retire an operation only after every TP partition completes physical work."""

        retired: list[tuple[int, CapacityOperation]] = []
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            operation = self.operations[group_index]
            if operation is None:
                continue
            sequence = operation.command.sequence
            replies = [completions[index] for index in self.group_partition_indices(fabric, instance_index, dp_rank)]
            if not all(reply is not None and reply.sequence == sequence for reply in replies):
                continue
            expected = operation.command.target_bundles
            if any(reply is None or reply.backed_bundles != expected for reply in replies):
                raise RuntimeError("kv capacity completion does not match the command target")
            self.active_bundles[group_index] = expected
            self.applied_sequences[group_index] = sequence
            self.operations[group_index] = None
            if sequence == 1:
                profile = self.group_profiles(registrations, fabric, instance_index, dp_rank)[0][1]
                logger.info(
                    "kv capacity initialized %s[dp=%s] (bundles: %s -> %s; tokens: %s -> %s)",
                    fabric.plan.instance_plans[instance_index].instance_id,
                    dp_rank,
                    operation.start_bundles,
                    expected,
                    profile.usable_tokens(operation.start_bundles),
                    profile.usable_tokens(expected),
                )
            elif expected > operation.start_bundles:
                self.completed_grant_orders[group_index] = self.next_grant_order
                self.next_grant_order += 1
            retired.append((group_index, operation))
        return retired

    def publish_next_initial_operation(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
    ) -> bool:
        """Publish the next sequential startup allocation."""

        if any(operation is not None for operation in self.operations):
            return False
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            if self.command_sequences[group_index] == 0:
                self.publish_operation(
                    group_index,
                    self.initial_target(registrations, fabric, instance_index, dp_rank),
                )
                return True
        return False

    def consume_demands(self, fabric: FabricGenerationState) -> None:
        """Refresh persistent coherent demand snapshots."""

        publications = self.channel.read_demands()
        for group_index, _, _ in self.capacity_groups(fabric):
            demand = publications[group_index]
            if demand is None:
                continue
            if demand.evaluated_sequence > self.command_sequences[group_index]:
                raise RuntimeError("kv capacity demand refers to an unpublished operation")
            self.demands[group_index] = demand

    def eligible_demand(self, group_index: int) -> xpool.native.kv.KvCapacityDemand | None:
        """Return one idle group's currently eligible complete demand witness."""

        demand = self.demands[group_index]
        active = self.active_bundles[group_index]
        ceiling = self.service_ceilings[group_index]
        if (
            demand is None
            or active is None
            or ceiling is None
            or self.operations[group_index] is not None
            or demand.evaluated_sequence != self.applied_sequences[group_index]
            or demand.requested_bundles is None
            or demand.deadline_monotonic_ns is None
            or demand.requested_bundles <= active
            or demand.requested_bundles > ceiling
        ):
            return None
        return demand

    def ordered_borrowers(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        now_ns: int,
    ) -> list[int]:
        """Order overdue demands, then predeadline demands on unrelated pools."""

        candidates: list[tuple[int, int, set[int]]] = []
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            demand = self.eligible_demand(group_index)
            if demand is None or demand.deadline_monotonic_ns is None:
                continue
            pools = {
                pool_index for pool_index, _ in self.group_profiles(registrations, fabric, instance_index, dp_rank)
            }
            candidates.append((group_index, demand.deadline_monotonic_ns, pools))
        overdue = [candidate for candidate in candidates if candidate[1] <= now_ns]
        protected_pools = set().union(*(pools for _, _, pools in overdue))
        ordered_overdue = sorted(
            overdue,
            key=lambda candidate: (
                self.completed_grant_orders[candidate[0]] is not None,
                self.completed_grant_orders[candidate[0]] or 0,
                candidate[1],
                candidate[0],
            ),
        )
        ordered_predeadline = sorted(
            (
                candidate
                for candidate in candidates
                if candidate[1] > now_ns and candidate[2].isdisjoint(protected_pools)
            ),
            key=lambda candidate: (candidate[1], candidate[0]),
        )
        return [group_index for group_index, _, _ in (*ordered_overdue, *ordered_predeadline)]

    def attempt_is_current(self, attempt: FundingAttempt) -> bool:
        """Return whether a donor attempt still serves its exact demand witness."""

        demand = self.eligible_demand(attempt.borrower_index)
        return (
            demand is not None
            and demand.evaluated_sequence == attempt.evaluated_sequence
            and demand.requested_bundles == attempt.target_bundles
            and demand.deadline_monotonic_ns == attempt.deadline_monotonic_ns
        )

    def advance_attempt(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        attempt: FundingAttempt,
    ) -> bool:
        """Publish one borrower growth or donor shrink for an exact witness."""

        if attempt.donor_index is not None:
            return False
        if not self.attempt_is_current(attempt):
            self.funding_attempt = None
            return False
        locations = {
            group_index: (instance_index, dp_rank)
            for group_index, instance_index, dp_rank in self.capacity_groups(fabric)
        }
        free = self.pool_free_bytes(registrations, fabric)
        borrower_instance, borrower_dp = locations[attempt.borrower_index]
        borrower_active = self.active_bundles[attempt.borrower_index]
        if borrower_active is None:
            raise RuntimeError("kv capacity funding borrower is uninitialized")
        required = self.growth_bytes(
            registrations,
            fabric,
            borrower_instance,
            borrower_dp,
            borrower_active,
            attempt.target_bundles,
        )
        if all(free[pool_index] >= byte_count for pool_index, byte_count in required.items()):
            self.publish_operation(attempt.borrower_index, attempt.target_bundles)
            self.funding_attempt = None
            profile = self.group_profiles(registrations, fabric, borrower_instance, borrower_dp)[0][1]
            logger.info(
                "kv capacity growth requested %s[dp=%s] (bundles: %s -> %s; tokens: %s -> %s)",
                fabric.plan.instance_plans[borrower_instance].instance_id,
                borrower_dp,
                borrower_active,
                attempt.target_bundles,
                profile.usable_tokens(borrower_active),
                profile.usable_tokens(attempt.target_bundles),
            )
            return True
        short_pools = {
            pool_index: byte_count - free[pool_index]
            for pool_index, byte_count in required.items()
            if byte_count > free[pool_index]
        }

        candidates: list[tuple[tuple[int, int, int, int], int, int, int]] = []
        for donor_index, donor_instance, donor_dp in self.capacity_groups(fabric):
            if donor_index == attempt.borrower_index or donor_index in attempt.attempted_donors:
                continue
            donor_active = self.active_bundles[donor_index]
            donor_demand = self.demands[donor_index]
            if (
                donor_active is None
                or self.operations[donor_index] is not None
                or donor_demand is None
                or donor_demand.evaluated_sequence != self.applied_sequences[donor_index]
            ):
                continue
            profiles = self.group_profiles(registrations, fabric, donor_instance, donor_dp)
            floors = {profile.floor_bundles for _, profile in profiles}
            if len(floors) != 1:
                raise RuntimeError("kv capacity donor ranks have different floors")
            floor = floors.pop()
            if donor_active <= floor:
                continue
            release_bundles = max(
                (
                    (short_pools[pool_index] + profile.bundle_bytes - 1) // profile.bundle_bytes
                    for pool_index, profile in profiles
                    if pool_index in short_pools
                ),
                default=0,
            )
            target = max(floor, donor_active - release_bundles)
            if target < donor_active:
                has_demand = donor_demand.requested_bundles is not None
                deadline = donor_demand.deadline_monotonic_ns or 0
                grant_order = self.completed_grant_orders[donor_index]
                candidates.append(
                    (
                        (
                            int(has_demand),
                            -deadline if has_demand else 0,
                            -(grant_order if grant_order is not None else -1),
                            donor_index,
                        ),
                        donor_index,
                        target,
                        donor_active,
                    )
                )

        if not candidates:
            self.funding_attempt = None
            return False

        _, donor_index, target, donor_active = min(candidates)
        donor_instance, donor_dp = locations[donor_index]
        self.publish_operation(donor_index, target)
        attempt.donor_index = donor_index
        donor_profile = self.group_profiles(registrations, fabric, donor_instance, donor_dp)[0][1]
        borrower_profile = self.group_profiles(registrations, fabric, borrower_instance, borrower_dp)[0][1]
        logger.info(
            "kv capacity transfer requested %s[dp=%s] -> %s[dp=%s] "
            "(donor bundles: %s -> %s; donor tokens: %s -> %s; "
            "borrower bundles: %s -> %s; borrower tokens: %s -> %s)",
            fabric.plan.instance_plans[donor_instance].instance_id,
            donor_dp,
            fabric.plan.instance_plans[borrower_instance].instance_id,
            borrower_dp,
            donor_active,
            target,
            donor_profile.usable_tokens(donor_active),
            donor_profile.usable_tokens(target),
            borrower_active,
            attempt.target_bundles,
            borrower_profile.usable_tokens(borrower_active),
            borrower_profile.usable_tokens(attempt.target_bundles),
        )
        return True

    def note_retired_donor(self, group_index: int) -> None:
        """Release one completed donor slot for exact-witness revalidation."""

        attempt = self.funding_attempt
        if attempt is None or attempt.donor_index != group_index:
            return
        attempt.donor_index = None
        attempt.attempted_donors.add(group_index)

    def step(self, registrations: RegistrationBook, fabric: FabricGenerationState) -> None:
        """Advance completions and publish at most one new capacity operation."""

        initial_backing = self.channel.read_initial_backing()
        if any(report is None for report in initial_backing):
            return
        device_reports = self.channel.read_device_memory()
        if any(report is None for report in device_reports):
            return
        if self.pool_capacity_bytes is None:
            self.freeze_pools(registrations, fabric, initial_backing, device_reports)

        retired = self.retire_operations(
            registrations,
            fabric,
            self.channel.read_completions(),
        )
        for group_index, _ in retired:
            self.note_retired_donor(group_index)

        groups = self.capacity_groups(fabric)
        if any(self.command_sequences[group_index] == 0 for group_index, _, _ in groups):
            self.publish_next_initial_operation(registrations, fabric)
            return
        if any(operation is not None and operation.command.sequence == 1 for operation in self.operations):
            return

        self.consume_demands(fabric)
        attempt = self.funding_attempt
        if attempt is not None:
            if self.advance_attempt(registrations, fabric, attempt):
                return
            if attempt.donor_index is not None:
                locations = {group_index: (instance_index, dp_rank) for group_index, instance_index, dp_rank in groups}
                borrower_instance, borrower_dp = locations[attempt.borrower_index]
                donor_instance, donor_dp = locations[attempt.donor_index]
                reserved_pools = {
                    pool_index
                    for instance_index, dp_rank in ((borrower_instance, borrower_dp), (donor_instance, donor_dp))
                    for pool_index, _ in self.group_profiles(registrations, fabric, instance_index, dp_rank)
                }
                free = self.pool_free_bytes(registrations, fabric)
                for candidate_index in self.ordered_borrowers(registrations, fabric, perf_counter_ns()):
                    candidate_instance, candidate_dp = locations[candidate_index]
                    required_pools = self.group_profiles(registrations, fabric, candidate_instance, candidate_dp)
                    if any(pool_index in reserved_pools for pool_index, _ in required_pools):
                        continue
                    demand = self.eligible_demand(candidate_index)
                    active = self.active_bundles[candidate_index]
                    if demand is None or demand.requested_bundles is None or active is None:
                        continue
                    required = self.growth_bytes(
                        registrations, fabric, candidate_instance, candidate_dp, active, demand.requested_bundles
                    )
                    if all(free[pool_index] >= byte_count for pool_index, byte_count in required.items()):
                        self.publish_operation(candidate_index, demand.requested_bundles)
                        profile = self.group_profiles(registrations, fabric, candidate_instance, candidate_dp)[0][1]
                        logger.info(
                            "kv capacity growth requested %s[dp=%s] (bundles: %s -> %s; tokens: %s -> %s)",
                            fabric.plan.instance_plans[candidate_instance].instance_id,
                            candidate_dp,
                            active,
                            demand.requested_bundles,
                            profile.usable_tokens(active),
                            profile.usable_tokens(demand.requested_bundles),
                        )
                        return
                return
        for borrower_index in self.ordered_borrowers(registrations, fabric, perf_counter_ns()):
            demand = self.eligible_demand(borrower_index)
            if demand is None or demand.requested_bundles is None or demand.deadline_monotonic_ns is None:
                continue
            attempt = FundingAttempt(
                borrower_index=borrower_index,
                evaluated_sequence=demand.evaluated_sequence,
                target_bundles=demand.requested_bundles,
                deadline_monotonic_ns=demand.deadline_monotonic_ns,
            )
            self.funding_attempt = attempt
            if self.advance_attempt(registrations, fabric, attempt):
                return

    def close(self) -> None:
        """Close and unlink this Generation's channel."""

        self.channel.close()
