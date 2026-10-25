"""Generation-scoped physical Elastic Context State byte-quota policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter_ns

import xpool.native
from xpool.config import XpoolConfig, get_global_config
from xpool.fabric import FabricGenerationId
from xpool.model import ModelId
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.daemon.registration import InstanceRankId, InstanceRankRegistrationState, RegistrationBook
from xpool.service.wire import ElasticContextStateControlChannelRef, ElasticContextStatePartitionProfile

__all__ = ["CapacityPolicy"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapacityOperation:
    """One immutable group grant; old permission remains charged until completion."""

    command: xpool.native.ecs.QuotaCommand
    start_quota_bytes: tuple[int, ...]


@dataclass(slots=True)
class FundingAttempt:
    """One borrower's exact persistent witness and serialized donor negotiation."""

    borrower_index: int
    evaluated_sequence: int
    target_quota_bytes: tuple[int, ...]
    deadline_monotonic_ns: int
    attempted_donors: set[int] = field(default_factory=set)
    donor_index: int | None = None


@dataclass(slots=True)
class CapacityPolicy:
    """Own one Generation's channel and physical permission across its devices."""

    generation: FabricGenerationId
    channel: xpool.native.ecs.DaemonControlChannel
    command_sequences: list[int] = field(init=False)
    applied_sequences: list[int] = field(init=False)
    active_quotas: list[tuple[int, ...] | None] = field(init=False)
    operations: list[CapacityOperation | None] = field(init=False)
    demands: list[xpool.native.ecs.QuotaDemand | None] = field(init=False)
    completed_grant_orders: list[int | None] = field(init=False)
    service_ceilings: list[tuple[int, ...] | None] = field(init=False)
    next_grant_order: int = 0
    funding_attempt: FundingAttempt | None = None
    pool_capacity_bytes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        group_count = sum(model.atn_dp_size for model in get_global_config().models)
        self.command_sequences = [0] * group_count
        self.applied_sequences = [0] * group_count
        self.active_quotas = [None] * group_count
        self.operations = [None] * group_count
        self.demands = [None] * group_count
        self.completed_grant_orders = [None] * group_count
        self.service_ceilings = [None] * group_count

    @classmethod
    def create(cls, config: XpoolConfig, generation: FabricGenerationId) -> CapacityPolicy:
        """Create contiguous group ranges in the Generation's existing TP-fastest order."""

        return cls(
            generation=generation,
            channel=xpool.native.ecs.DaemonControlChannel.create(
                device_count=config.atn_world_size,
                group_partition_counts=tuple(
                    config.atn_tp_size_of(model.id) for model in config.models for _ in range(model.atn_dp_size)
                ),
            ),
        )

    @property
    def channel_ref(self) -> ElasticContextStateControlChannelRef:
        """Return the discovery value for this Generation's native channel."""

        return ElasticContextStateControlChannelRef(generation=self.generation, name=self.channel.name)

    def partition(self, registrations: RegistrationBook, model_id: ModelId, rank: int) -> InstanceRankRegistrationState:
        """Return a retained Generation partition registration."""

        registration = registrations.instances.query(InstanceRankId(model_id=model_id, rank=rank))
        if registration is None:
            raise RuntimeError(
                f"elastic context state quota policy lost registration for instance {model_id} rank {rank}"
            )
        return registration

    def capacity_groups(self, fabric: FabricGenerationState) -> list[tuple[int, int, int]]:
        """Return ``(group index, instance index, DP rank)`` in directory order."""

        return [
            (group_index, instance_index, dp_rank)
            for group_index, (instance_index, dp_rank) in enumerate(
                (instance_index, dp_rank)
                for instance_index, plan in enumerate(fabric.plan.instance_plans)
                for dp_rank in range(plan.instance_rank_topology.atn_dp_size)
            )
        ]

    def group_profiles(
        self, registrations: RegistrationBook, fabric: FabricGenerationState, instance_index: int, dp_rank: int
    ) -> tuple[tuple[int, ElasticContextStatePartitionProfile], ...]:
        """Return rank-local device indices and backing bounds, in TP-rank order."""

        plan = fabric.plan.instance_plans[instance_index]
        topology = plan.instance_rank_topology
        start_rank = dp_rank * topology.atn_tp_size
        return tuple(
            (rank, self.partition(registrations, plan.model_id, rank).context_state_profile)
            for rank in range(start_rank, start_rank + topology.atn_tp_size)
        )

    def group_partition_indices(
        self, fabric: FabricGenerationState, instance_index: int, dp_rank: int
    ) -> tuple[int, ...]:
        """Return the group's physical partition slots in TP-rank order."""

        width = fabric.plan.instance_plans[instance_index].instance_rank_topology.atn_tp_size
        row = instance_index * get_global_config().atn_world_size
        return tuple(row + dp_rank * width + rank for rank in range(width))

    def freeze_pools(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        initial_backing: list[int | None],
        device_reports: list[xpool.native.ecs.DeviceMemoryReport | None],
    ) -> None:
        """Freeze post-capture device budgets, partition floors and immutable ceilings."""

        config = get_global_config()
        budgets: list[int] = []
        for device_index, report in enumerate(device_reports):
            if report is None:
                raise RuntimeError("elastic context state quota freezing requires every device report")
            mapped = floors = headroom = 0
            for instance_index, plan in enumerate(fabric.plan.instance_plans):
                backing = initial_backing[instance_index * config.atn_world_size + device_index]
                registration = self.partition(registrations, plan.model_id, device_index)
                profile = registration.context_state_profile
                if backing is None or backing != profile.floor_bytes:
                    raise RuntimeError("elastic context state startup backing differs from its registered floor")
                mapped += backing
                floors += profile.floor_bytes
                headroom += registration.atn_runtime_headroom_bytes
            utilization_reserve = report.total_bytes - int(report.total_bytes * config.atn.device_memory_utilization)
            reserve = max(utilization_reserve, headroom)
            budget = report.free_bytes + mapped - reserve
            if budget < floors:
                raise RuntimeError(
                    f"attention device {config.atn.devices[device_index]} cannot retain all backing floors: "
                    f"total_bytes={report.total_bytes} free_bytes={report.free_bytes} bootstrap_bytes={mapped} "
                    f"utilization_reserve_bytes={utilization_reserve} runtime_headroom_bytes={headroom} "
                    f"budget_bytes={budget} floor_bytes={floors}"
                )
            budgets.append(budget)
            logger.info(
                "elastic context state budget frozen device=%s total_bytes=%s free_bytes=%s bootstrap_bytes=%s "
                "reserve_bytes=%s runtime_headroom_bytes=%s budget_bytes=%s floor_bytes=%s",
                config.atn.devices[device_index],
                report.total_bytes,
                report.free_bytes,
                mapped,
                reserve,
                headroom,
                budget,
                floors,
            )
        self.pool_capacity_bytes = tuple(budgets)
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            self.active_quotas[group_index] = tuple(
                profile.floor_bytes
                for _, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank)
            )
            ceiling = self.service_ceiling(registrations, fabric, instance_index, dp_rank)
            self.service_ceilings[group_index] = ceiling
            self.channel.publish_service_ceiling(group_index, ceiling)

    def initial_target(
        self, registrations: RegistrationBook, fabric: FabricGenerationState, instance_index: int, dp_rank: int
    ) -> tuple[int, ...]:
        """Add one equal byte increment per device, capped by each real reservation."""

        budgets = self.pool_capacity_bytes
        if budgets is None:
            raise RuntimeError("attention device budgets are not frozen")
        targets: list[int] = []
        for device_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank):
            floors = sum(
                self.partition(registrations, plan.model_id, device_index).context_state_profile.floor_bytes
                for plan in fabric.plan.instance_plans
            )
            extra = (budgets[device_index] - floors) // len(fabric.plan.instance_plans)
            targets.append(min(profile.reservation_bytes, profile.floor_bytes + extra))
        return tuple(targets)

    def service_ceiling(
        self, registrations: RegistrationBook, fabric: FabricGenerationState, instance_index: int, dp_rank: int
    ) -> tuple[int, ...]:
        """Return independent device ceilings while every other partition retains its floor."""

        budgets = self.pool_capacity_bytes
        if budgets is None:
            raise RuntimeError("attention device budgets are not frozen")
        ceilings: list[int] = []
        for device_index, profile in self.group_profiles(registrations, fabric, instance_index, dp_rank):
            other_floors = sum(
                self.partition(registrations, plan.model_id, device_index).context_state_profile.floor_bytes
                for index, plan in enumerate(fabric.plan.instance_plans)
                if index != instance_index
            )
            ceilings.append(min(profile.reservation_bytes, budgets[device_index] - other_floors))
        return tuple(ceilings)

    def pool_free_bytes(self, registrations: RegistrationBook, fabric: FabricGenerationState) -> list[int]:
        """Subtract permission, charging max(old, target) until whole-group completion."""

        if self.pool_capacity_bytes is None:
            raise RuntimeError("attention device budgets are not frozen")
        free = list(self.pool_capacity_bytes)
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            active = self.active_quotas[group_index]
            if active is None:
                continue
            operation = self.operations[group_index]
            target = operation.command.quota_bytes if operation is not None else active
            for (device_index, _), old, new in zip(
                self.group_profiles(registrations, fabric, instance_index, dp_rank), active, target, strict=True
            ):
                free[device_index] -= max(old, new)
        if any(value < 0 for value in free):
            raise RuntimeError("elastic context state quota accounting exceeded a physical device budget")
        return free

    def growth_bytes(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        instance_index: int,
        dp_rank: int,
        start: tuple[int, ...],
        target: tuple[int, ...],
    ) -> dict[int, int]:
        """Return the requested permission increment on each participating device."""

        return {
            device_index: new - old
            for (device_index, _), old, new in zip(
                self.group_profiles(registrations, fabric, instance_index, dp_rank), start, target, strict=True
            )
        }

    def publish_operation(self, group_index: int, quota_bytes: tuple[int, ...]) -> None:
        """Publish one immutable operation for an initialized idle group."""

        active = self.active_quotas[group_index]
        if active is None or self.operations[group_index] is not None:
            raise RuntimeError("elastic context state quota operation requires one initialized idle group")
        command = xpool.native.ecs.QuotaCommand(self.command_sequences[group_index] + 1, quota_bytes)
        self.channel.publish_command(group_index, command)
        self.command_sequences[group_index] = command.sequence
        self.operations[group_index] = CapacityOperation(command, active)
        logger.info(
            "elastic context state quota requested group=%s sequence=%s from_bytes=%s quota_bytes=%s",
            group_index,
            command.sequence,
            active,
            quota_bytes,
        )

    def retire_operations(
        self,
        registrations: RegistrationBook,
        fabric: FabricGenerationState,
        completions: list[xpool.native.ecs.QuotaCompletion | None],
    ) -> list[tuple[int, CapacityOperation]]:
        """Credit a shrink only after all ranks adopted the grant and fit it physically."""

        retired: list[tuple[int, CapacityOperation]] = []
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            operation = self.operations[group_index]
            if operation is None:
                continue
            sequence = operation.command.sequence
            replies = tuple(
                completions[index] for index in self.group_partition_indices(fabric, instance_index, dp_rank)
            )
            if not all(reply is not None and reply.sequence == sequence for reply in replies):
                continue
            target = operation.command.quota_bytes
            for reply, (_, profile), quota in zip(
                replies, self.group_profiles(registrations, fabric, instance_index, dp_rank), target, strict=True
            ):
                if reply is None or not profile.floor_bytes <= reply.backed_bytes <= quota:
                    raise RuntimeError("elastic context state quota completion is outside its floor and adopted grant")
            self.active_quotas[group_index] = target
            self.applied_sequences[group_index] = sequence
            self.operations[group_index] = None
            if sequence != 1 and any(new > old for old, new in zip(operation.start_quota_bytes, target, strict=True)):
                self.completed_grant_orders[group_index] = self.next_grant_order
                self.next_grant_order += 1
            retired.append((group_index, operation))
        return retired

    def publish_next_initial_operation(self, registrations: RegistrationBook, fabric: FabricGenerationState) -> bool:
        """Publish one startup grant; allocation remains on demand inside each rank."""

        if any(operation is not None for operation in self.operations):
            return False
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            if self.command_sequences[group_index] == 0:
                self.publish_operation(group_index, self.initial_target(registrations, fabric, instance_index, dp_rank))
                return True
        return False

    def consume_demands(self, fabric: FabricGenerationState) -> None:
        """Validate new coherent vectors without discarding snapshots during a publication."""

        publications = self.channel.read_demands()
        for group_index, _, _ in self.capacity_groups(fabric):
            demand = publications[group_index]
            if demand is None:
                continue
            if demand.evaluated_sequence > self.command_sequences[group_index]:
                raise RuntimeError("elastic context state demand refers to an unpublished operation")
            active = self.active_quotas[group_index]
            required = demand.required_bytes
            if required is not None and active is not None:
                if len(required) != len(active):
                    raise RuntimeError("elastic context state demand vector differs from group membership")
                if demand.evaluated_sequence == self.applied_sequences[group_index] and any(
                    new < old for old, new in zip(active, required, strict=True)
                ):
                    raise RuntimeError("elastic context state demand requests an implicit partition shrink")
            self.demands[group_index] = demand

    def eligible_demand(self, group_index: int) -> xpool.native.ecs.QuotaDemand | None:
        """Return a current idle-group witness that fits every immutable ceiling."""

        demand = self.demands[group_index]
        active, ceiling = self.active_quotas[group_index], self.service_ceilings[group_index]
        if (
            demand is None
            or active is None
            or ceiling is None
            or self.operations[group_index] is not None
            or demand.evaluated_sequence != self.applied_sequences[group_index]
            or demand.required_bytes is None
            or demand.deadline_monotonic_ns is None
        ):
            return None
        required = demand.required_bytes
        if (
            not all(old <= new <= cap for old, new, cap in zip(active, required, ceiling, strict=True))
            or required == active
        ):
            return None
        return demand

    def ordered_borrowers(
        self, registrations: RegistrationBook, fabric: FabricGenerationState, now_ns: int
    ) -> list[int]:
        """Preserve overdue grant ordering, then predeadline work on unrelated devices."""

        candidates: list[tuple[int, int, set[int]]] = []
        for group_index, instance_index, dp_rank in self.capacity_groups(fabric):
            demand = self.eligible_demand(group_index)
            if demand is None or demand.deadline_monotonic_ns is None:
                continue
            devices = {device for device, _ in self.group_profiles(registrations, fabric, instance_index, dp_rank)}
            candidates.append((group_index, demand.deadline_monotonic_ns, devices))
        overdue = [candidate for candidate in candidates if candidate[1] <= now_ns]
        protected = set().union(*(devices for _, _, devices in overdue))
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
            (candidate for candidate in candidates if candidate[1] > now_ns and candidate[2].isdisjoint(protected)),
            key=lambda candidate: (candidate[1], candidate[0]),
        )
        return [group for group, _, _ in (*ordered_overdue, *ordered_predeadline)]

    def attempt_is_current(self, attempt: FundingAttempt) -> bool:
        """Revalidate the original sequence, vector and deadline after each donor."""

        demand = self.eligible_demand(attempt.borrower_index)
        return (
            demand is not None
            and demand.evaluated_sequence == attempt.evaluated_sequence
            and demand.required_bytes == attempt.target_quota_bytes
            and demand.deadline_monotonic_ns == attempt.deadline_monotonic_ns
        )

    def advance_attempt(
        self, registrations: RegistrationBook, fabric: FabricGenerationState, attempt: FundingAttempt
    ) -> bool:
        """Use unassigned bytes first, then shrink one donor on its short devices only."""

        if attempt.donor_index is not None:
            return False
        if not self.attempt_is_current(attempt):
            self.funding_attempt = None
            return False
        locations = {group: (instance, dp) for group, instance, dp in self.capacity_groups(fabric)}
        free = self.pool_free_bytes(registrations, fabric)
        borrower_instance, borrower_dp = locations[attempt.borrower_index]
        active = self.active_quotas[attempt.borrower_index]
        if active is None:
            raise RuntimeError("elastic context state funding borrower is uninitialized")
        required = self.growth_bytes(
            registrations, fabric, borrower_instance, borrower_dp, active, attempt.target_quota_bytes
        )
        if all(free[device] >= count for device, count in required.items()):
            self.publish_operation(attempt.borrower_index, attempt.target_quota_bytes)
            self.funding_attempt = None
            return True
        shortages = {device: count - free[device] for device, count in required.items() if count > free[device]}
        candidates: list[tuple[tuple[int, int, int, int], int, tuple[int, ...]]] = []
        for donor_index, donor_instance, donor_dp in self.capacity_groups(fabric):
            if donor_index == attempt.borrower_index or donor_index in attempt.attempted_donors:
                continue
            donor_active = self.active_quotas[donor_index]
            demand = self.demands[donor_index]
            if (
                donor_active is None
                or self.operations[donor_index] is not None
                or demand is None
                or (demand.evaluated_sequence != self.applied_sequences[donor_index])
            ):
                continue
            target = tuple(
                max(profile.floor_bytes, current - shortages.get(device, 0))
                for (device, profile), current in zip(
                    self.group_profiles(registrations, fabric, donor_instance, donor_dp), donor_active, strict=True
                )
            )
            if target == donor_active:
                continue
            has_demand = demand.required_bytes is not None
            order = self.completed_grant_orders[donor_index]
            priority = (
                int(has_demand),
                -(demand.deadline_monotonic_ns or 0) if has_demand else 0,
                -(order if order is not None else -1),
                donor_index,
            )
            candidates.append((priority, donor_index, target))
        if not candidates:
            self.funding_attempt = None
            return False
        _, donor_index, target = min(candidates)
        self.publish_operation(donor_index, target)
        attempt.donor_index = donor_index
        return True

    def note_retired_donor(self, group_index: int) -> None:
        """Release the completed donor slot before exact-witness revalidation."""

        attempt = self.funding_attempt
        if attempt is not None and attempt.donor_index == group_index:
            attempt.donor_index = None
            attempt.attempted_donors.add(group_index)

    def step(self, registrations: RegistrationBook, fabric: FabricGenerationState) -> None:
        """Consume completions and publish at most one new immutable group operation."""

        backing = self.channel.read_initial_backing()
        reports = self.channel.read_device_memory()
        if any(value is None for value in backing) or any(value is None for value in reports):
            return
        if self.pool_capacity_bytes is None:
            self.freeze_pools(registrations, fabric, backing, reports)
        for group_index, _ in self.retire_operations(registrations, fabric, self.channel.read_completions()):
            self.note_retired_donor(group_index)
        groups = self.capacity_groups(fabric)
        if any(self.command_sequences[group] == 0 for group, _, _ in groups):
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
                locations = {group: (instance, dp) for group, instance, dp in groups}
                reserved = {
                    device
                    for group in (attempt.borrower_index, attempt.donor_index)
                    for device, _ in self.group_profiles(registrations, fabric, *locations[group])
                }
                free = self.pool_free_bytes(registrations, fabric)
                for group in self.ordered_borrowers(registrations, fabric, perf_counter_ns()):
                    instance, dp = locations[group]
                    if any(
                        device in reserved for device, _ in self.group_profiles(registrations, fabric, instance, dp)
                    ):
                        continue
                    demand, active = self.eligible_demand(group), self.active_quotas[group]
                    if demand is None or demand.required_bytes is None or active is None:
                        continue
                    required = self.growth_bytes(registrations, fabric, instance, dp, active, demand.required_bytes)
                    if all(free[device] >= count for device, count in required.items()):
                        self.publish_operation(group, demand.required_bytes)
                        return
                return
        for group in self.ordered_borrowers(registrations, fabric, perf_counter_ns()):
            demand = self.eligible_demand(group)
            if demand is None or demand.required_bytes is None or demand.deadline_monotonic_ns is None:
                continue
            attempt = FundingAttempt(
                group, demand.evaluated_sequence, demand.required_bytes, demand.deadline_monotonic_ns
            )
            self.funding_attempt = attempt
            if self.advance_attempt(registrations, fabric, attempt):
                return

    def close(self) -> None:
        """Close and unlink this Generation's channel."""

        self.channel.close()
