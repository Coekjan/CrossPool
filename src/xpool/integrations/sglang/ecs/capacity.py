"""Matched TP decisions and rank-local retirement under daemon byte permission."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import isfinite

import torch
import torch.distributed
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import PostCaptureKVResize
from sglang.srt.runtime_context import get_parallel

from xpool.config import LatencySloConfig
from xpool.fabric import FabricGenerationPhase
from xpool.integrations.sglang.ecs.allocation import SglangAllocation, SglangAllocationStatus
from xpool.integrations.sglang.ecs.radix import SglangCacheEviction
from xpool.integrations.sglang.ecs.state import (
    SglangElasticContextState,
    SglangMemoryRequirement,
    SglangPrefillProposal,
    SglangReclaimPlan,
)
from xpool.native.ecs import (
    InstanceControlChannel,
    QuotaCommand,
    QuotaCompletion,
    QuotaDemand,
)
from xpool.runtime.instance import INSTANCE_STARTUP_BARRIER_TIMEOUT_S, InstanceRankError, InstanceRankRuntime

CAPACITY_COMMAND_POLL_INTERVAL_S = 0.01
CAPACITY_GENERATION_CHECK_INTERVAL_S = 0.5
capacity_reconciler_context = ContextVar["CapacityReconciler | None"]("xpool_capacity_reconciler", default=None)


@contextmanager
def capacity_reconciler_scope(reconciler: CapacityReconciler) -> Iterator[None]:
    """Lend the existing scheduler-thread owner to nested engine hooks."""

    token = capacity_reconciler_context.set(reconciler)
    try:
        yield
    finally:
        capacity_reconciler_context.reset(token)


def current_capacity_reconciler() -> CapacityReconciler:
    """Return the scheduler scope's owner; an unbound serving hook is a lifecycle error."""

    reconciler = capacity_reconciler_context.get()
    if reconciler is None:
        raise RuntimeError("xpool elastic context state hook ran outside its capacity reconciler scope")
    return reconciler


@dataclass(frozen=True, slots=True)
class DemandWitness:
    """One blocked request set evaluated under a grant, with its original deadline."""

    evaluated_sequence: int
    required_bytes: tuple[int, ...]
    deadline_monotonic_ns: int


@dataclass(slots=True)
class CapacityReconciler:
    """Own one Capacity Group rank's quota adoption, admission and physical retirement.

    Collective order follows common outer decisions. Event progress changes
    local ownership only; it never independently chooses another collective.
    """

    channel: InstanceControlChannel
    command_index: int
    state: SglangElasticContextState
    instance_rank: InstanceRankRuntime
    slo: LatencySloConfig
    command: QuotaCommand | None = None
    active_quota_bytes: tuple[int, ...] = ()
    completed_sequence: int = 0
    pending_retirement: SglangReclaimPlan | None = None
    admission_free_pages: tuple[int, ...] = ()
    deferred_eviction_pages: tuple[int, ...] = ()
    service_ceiling_bytes: tuple[int, ...] | None = None
    unresolved_demands: dict[tuple[Req, ...], DemandWitness] = field(default_factory=dict)
    published_demand: tuple[int, tuple[int, ...] | None, int | None] | None = None
    scheduler: Scheduler | None = None

    @classmethod
    def attach(
        cls,
        *,
        channel_name: str,
        group_index: int,
        partition_index: int,
        dp_group_count: int,
        state: SglangElasticContextState,
        instance_rank: InstanceRankRuntime,
        slo: LatencySloConfig,
    ) -> CapacityReconciler:
        """Attach the existing Generation mapping using its exact membership coordinates."""

        return cls(
            channel=InstanceControlChannel.attach(channel_name, group_index, partition_index, dp_group_count),
            command_index=get_parallel().attn_dp_rank,
            state=state,
            instance_rank=instance_rank,
            slo=slo,
            admission_free_pages=state.logical_free_pages(),
            deferred_eviction_pages=(0,) * len(state.components),
        )

    @property
    def tp_group(self) -> GroupCoordinator:
        """Return the engine's existing attention TP coordinator, including its CPU group."""

        parallel = get_parallel()
        group = parallel.attn_tp_group if parallel.enable_dp_attention else parallel.tp_group
        if group is None:
            raise RuntimeError("xpool elastic context state requires initialized TP membership")
        return group

    @property
    def draining(self) -> bool:
        """Keep new Prefill out while any partition has a pending smaller grant."""

        command = self.command
        return bool(
            command is not None
            and command.sequence > self.completed_sequence
            and self.active_quota_bytes
            and any(new < old for old, new in zip(self.active_quota_bytes, command.quota_bytes, strict=True))
        )

    @property
    def has_chunk_owner(self) -> bool:
        """Include the final chunk until its tentative allocation has committed."""

        attempt = SglangAllocation.current()
        return bool(
            (self.scheduler is not None and self.scheduler.chunked_req is not None)
            or (attempt is not None and not attempt.committed and attempt.prior_chunk_owner is not None)
        )

    def accept_command(self, command: QuotaCommand | None) -> None:
        """Retain the next immutable command without replacing unfinished retirement."""

        if command is None or command.sequence <= self.completed_sequence:
            return
        current = self.command
        if current is not None:
            if command.sequence < current.sequence:
                return
            if command.sequence == current.sequence:
                if command.quota_bytes != current.quota_bytes:
                    raise RuntimeError("xpool elastic context state quota changed without advancing its sequence")
                return
            if current.sequence > self.completed_sequence:
                raise RuntimeError("xpool elastic context state command supersedes an unfinished operation")
        self.command = command

    def check_generation(self) -> None:
        """Use the retained failure owner while startup waits for peer publications."""

        plan = self.instance_rank.fabric_plan
        if plan is None:
            raise InstanceRankError("elastic context state capacity requires an executable fabric generation")
        readiness = self.instance_rank.client.readiness()
        if readiness.generation != plan.generation or readiness.fabric_phase is not FabricGenerationPhase.EXECUTABLE:
            raise InstanceRankError(
                "fabric generation became unavailable during elastic context state capacity reconciliation"
            )

    def vote_ready(self, ready: bool) -> bool:
        """Enter one matched all-ready decision, even after a local clean rejection."""

        if get_parallel().attn_tp_size == 1:
            return ready
        vote = torch.tensor([int(ready)], dtype=torch.int32)
        torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN, group=self.tp_group.cpu_group)
        return bool(vote.item())

    def allocation_verdict(self, status: SglangAllocationStatus) -> SglangAllocationStatus:
        """Publish one complete batch outcome before counters, penalties or forward."""

        if get_parallel().attn_tp_size == 1:
            return status
        vote = torch.tensor([int(status)], dtype=torch.int32)
        torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MAX, group=self.tp_group.cpu_group)
        return SglangAllocationStatus(int(vote.item()))

    def sync_admission_budget(self) -> None:
        """Observe common logical free space without altering any local free-list owner."""

        free = torch.tensor(self.state.logical_free_pages(), dtype=torch.int64)
        if get_parallel().attn_tp_size > 1:
            torch.distributed.all_reduce(free, op=torch.distributed.ReduceOp.MIN, group=self.tp_group.cpu_group)
        self.admission_free_pages = tuple(free.tolist())

    def finish_retirement(self) -> None:
        """Poll local physical dependencies while retaining the same retirement record."""

        plan = self.pending_retirement
        if plan is not None:
            self.state.finish_retirement(plan)
            if plan.completed:
                self.pending_retirement = None

    def reclaim(self, release_bytes: int, allocation_pages: tuple[int, ...]) -> bool:
        """Release empty backing first, then agree on one exact logical cache effect.

        The leader considers every rank's needs. An empty-backing retirement
        completes at later boundaries before another cache operation is chosen.
        Page targets require resident delivery, not unmapped logical capacity.
        """

        if self.has_chunk_owner:
            release_bytes = 0
        if self.pending_retirement is None and release_bytes > 0:
            empty = self.state.prepare_reclaim(release_bytes)
            if empty is not None:
                self.state.apply_reclaim(empty)
                self.pending_retirement = empty
        if not self.vote_ready(self.pending_retirement is None):
            return False
        free = self.state.resident_free_pages()
        needs = torch.tensor(
            [
                *(int(need > available) for need, available in zip(allocation_pages, free, strict=True)),
                int(release_bytes > 0),
            ],
            dtype=torch.int32,
        )
        if get_parallel().attn_tp_size > 1:
            torch.distributed.all_reduce(needs, op=torch.distributed.ReduceOp.MAX, group=self.tp_group.cpu_group)
        if not bool(needs.any().item()):
            return False
        candidate: SglangCacheEviction | None = None
        if get_parallel().attn_tp_rank == 0:
            candidate = self.state.select_cache_eviction(
                tuple(bool(value) for value in needs[:-1].tolist()), release_backing=bool(needs[-1].item())
            )
        if get_parallel().attn_tp_size > 1:
            candidate = self.tp_group.broadcast_object(candidate)
        if candidate is None:
            return False
        plan = self.state.prepare_reclaim(release_bytes, cache_eviction=candidate)
        if not self.vote_ready(plan is not None):
            return False
        if plan is None:
            raise RuntimeError("xpool elastic context state cache readiness lost its prepared owner")
        self.state.apply_reclaim(plan)
        self.pending_retirement = None if plan.completed else plan
        self.sync_admission_budget()
        return True

    def begin_scheduling(self, scheduler: Scheduler | None) -> None:
        """Progress the current grant and physical work at a matched outer boundary."""

        self.finish_retirement()
        command = self.command
        if command is not None and command.sequence > self.completed_sequence:
            target = command.quota_bytes[get_parallel().attn_tp_rank]
            can_adopt = not (self.draining and self.has_chunk_owner)
            if self.vote_ready(
                can_adopt and self.pending_retirement is None and self.state.quota.charged_bytes <= target
            ):
                self.state.quota.set_limit(target)
                self.active_quota_bytes = command.quota_bytes
                self.channel.publish_completion(QuotaCompletion(command.sequence, self.state.quota.charged_bytes))
                self.completed_sequence = command.sequence
                if scheduler is not None:
                    scheduler.running_batch.batch_is_full = False
            elif scheduler is not None:
                self.reclaim(max(0, self.state.quota.charged_bytes - target), (0,) * len(self.state.components))
        if scheduler is not None:
            deferred = torch.tensor(
                self.deferred_eviction_pages or (0,) * len(self.state.components), dtype=torch.int64
            )
            if get_parallel().attn_tp_size > 1:
                torch.distributed.all_reduce(deferred, op=torch.distributed.ReduceOp.MAX, group=self.tp_group.cpu_group)
            if bool(deferred.any().item()):
                if self.reclaim(0, tuple(deferred.tolist())):
                    self.deferred_eviction_pages = (0,) * len(self.state.components)
            self.sync_admission_budget()

    def memory_ready(self, memory: SglangMemoryRequirement) -> bool:
        """Check forecast and concrete realization jointly, without mapping or eviction."""

        forecast = self.state.required_bytes(memory)
        immediate = self.state.allocation_bytes(memory.allocation_pages)
        return self.vote_ready(
            forecast is not None and immediate is not None and max(forecast, immediate) <= self.state.quota.limit_bytes
        )

    def reclaim_requirement(self, memory: SglangMemoryRequirement, resident_pages: tuple[int, ...]) -> bool:
        """Price forecast and resident delivery from current inventories before reclaiming."""

        forecast = self.state.required_bytes(memory)
        immediate = self.state.allocation_bytes(resident_pages)
        release = max(0, max(forecast or 0, immediate or 0) - self.state.quota.limit_bytes)
        return self.reclaim(release, resident_pages)

    def prepare_prefill(self, proposal: SglangPrefillProposal) -> bool:
        """Map and agree on resident capacity without reclaiming or acquiring request pages."""

        error: BaseException | None = None
        try:
            status = (
                SglangAllocationStatus.READY
                if self.state.prepare_allocation(proposal.preparation_pages)
                else SglangAllocationStatus.CAPACITY_MISS
            )
        except BaseException as failure:
            error, status = failure, SglangAllocationStatus.FAILED
        verdict = self.allocation_verdict(status)
        if verdict is SglangAllocationStatus.FAILED:
            if error is not None:
                raise error
            raise RuntimeError("xpool Prefill backing preparation failed on a TP peer")
        return verdict is SglangAllocationStatus.READY

    def coordinate_decode(self, memory: SglangMemoryRequirement) -> bool:
        """Agree on the complete Decode/subset decision outside pure repeated queries."""

        if self.memory_ready(memory):
            return True
        return self.reclaim_requirement(memory, memory.allocation_pages) and self.memory_ready(memory)

    def defer_eviction(self, params: EvictParams) -> None:
        """Retain the requested resident-free target for the next common cache boundary."""

        requested = self.state.eviction_page_targets(params)
        previous = self.deferred_eviction_pages or (0,) * len(requested)
        self.deferred_eviction_pages = tuple(max(old, target) for old, target in zip(previous, requested, strict=True))

    def deadline_ns(self, origin: float, target_ms: float) -> int:
        """Convert the authoritative scheduler timing origin to an absolute deadline."""

        if not isfinite(origin) or origin <= 0:
            raise RuntimeError("xpool elastic context state demand has no valid scheduler timing origin")
        return int((origin + target_ms / 1000) * 1_000_000_000)

    def record_requirement(
        self,
        requests: tuple[Req, ...],
        memory: SglangMemoryRequirement | None,
        deadline_monotonic_ns: int,
    ) -> None:
        """Gather absolute rank requirements, preserving one request-keyed blocked witness."""

        if memory is None:
            self.unresolved_demands = {
                key: witness
                for key, witness in self.unresolved_demands.items()
                if not any(req in key for req in requests)
            }
            return
        required = self.state.required_bytes(memory)
        value = torch.tensor(
            [-1 if required is None else max(required, self.state.quota.limit_bytes)], dtype=torch.int64
        )
        if get_parallel().attn_tp_size > 1:
            gathered = [torch.empty_like(value) for _ in range(get_parallel().attn_tp_size)]
            torch.distributed.all_gather(gathered, value, group=self.tp_group.cpu_group)
            values = tuple(int(item.item()) for item in gathered)
        else:
            values = (int(value.item()),)
        if get_parallel().attn_tp_rank != 0:
            return
        ceiling = self.service_ceiling_bytes
        if (
            any(item < 0 for item in values)
            or ceiling is None
            or any(item > cap for item, cap in zip(values, ceiling, strict=True))
            or values == self.active_quota_bytes
        ):
            self.unresolved_demands.pop(requests, None)
            return
        self.unresolved_demands[requests] = DemandWitness(self.completed_sequence, values, deadline_monotonic_ns)

    def record_prefill_requirement(self, request: Req, memory: SglangMemoryRequirement | None) -> None:
        """Use the original scheduler receipt as this Prefill witness's deadline origin."""

        deadline = (
            self.deadline_ns(request.time_stats.scheduler_recv_time, self.slo.ttft_ms)
            if (memory is not None and get_parallel().attn_tp_rank == 0)
            else 0
        )
        self.record_requirement((request,), memory, deadline)

    def record_decode_requirement(self, requests: Iterable[Req], memory: SglangMemoryRequirement | None) -> None:
        """Use the earliest next-token deadline of the authoritative Decode request set."""

        selected = tuple(requests)
        if not selected:
            raise ValueError("xpool elastic context state Decode requirement needs a nonempty request set")
        deadline = (
            min(
                self.deadline_ns(
                    req.time_stats.last_decode_finish_time or req.time_stats.last_prefill_finished_time, self.slo.tbt_ms
                )
                for req in selected
            )
            if memory is not None and get_parallel().attn_tp_rank == 0
            else 0
        )
        self.record_requirement(selected, memory, deadline)

    def finish_scheduling(self, scheduler: Scheduler) -> None:
        """Publish only unresolved current work, keeping its existing timing evidence."""

        if get_parallel().attn_tp_rank != 0 or self.completed_sequence == 0:
            return
        active = set(scheduler.waiting_queue) | set(scheduler.running_batch.reqs)
        if scheduler.chunked_req is not None:
            active.add(scheduler.chunked_req)
        self.unresolved_demands = {
            requests: witness
            for requests, witness in self.unresolved_demands.items()
            if witness.evaluated_sequence == self.completed_sequence
            and all(req in active and not req.finished() for req in requests)
        }
        witness = min(self.unresolved_demands.values(), key=lambda value: value.deadline_monotonic_ns, default=None)
        publication = (
            self.completed_sequence,
            None if witness is None else witness.required_bytes,
            None if witness is None else witness.deadline_monotonic_ns,
        )
        if publication != self.published_demand:
            self.channel.publish_demand(QuotaDemand(*publication))
            self.published_demand = publication

    def finalize_after_capture(self, model_runner: ModelRunner) -> PostCaptureKVResize:
        """Publish actual floors, adopt the startup grant and derive stable joint limits."""

        torch.cuda.synchronize(model_runner.device)
        self.channel.publish_initial_backing(self.state.quota.charged_bytes)
        deadline = time.monotonic() + INSTANCE_STARTUP_BARRIER_TIMEOUT_S
        next_check = time.monotonic()
        while time.monotonic() < deadline:
            self.accept_command(self.channel.read_commands()[self.command_index])
            if self.service_ceiling_bytes is None:
                self.service_ceiling_bytes = self.channel.service_ceiling()
            if self.command is not None:
                self.begin_scheduling(None)
            if self.completed_sequence and self.service_ceiling_bytes is not None:
                break
            now = time.monotonic()
            if now >= next_check:
                self.check_generation()
                next_check = now + CAPACITY_GENERATION_CHECK_INTERVAL_S
            time.sleep(CAPACITY_COMMAND_POLL_INTERVAL_S)
        else:
            raise InstanceRankError("timed out waiting for the initial elastic context state quota")
        reference = self.state.reference_config(self.service_ceiling_bytes[get_parallel().attn_tp_rank])
        return PostCaptureKVResize(
            max_total_num_tokens=reference.max_total_num_tokens,
            full_max_total_num_tokens=reference.full_max_total_num_tokens,
            swa_max_total_num_tokens=reference.swa_max_total_num_tokens,
            capped_max_running_requests=None,
        )

    def close(self) -> None:
        """Close this rank's channel after the lifecycle has stopped its users."""

        self.channel.close()
