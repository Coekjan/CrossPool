"""Instance-local elastic KV capacity reconciliation."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

import torch
import torch.distributed
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import PostCaptureKVResize
from sglang.srt.runtime_context import get_parallel

import xpool.native
from xpool.config import LatencySloConfig
from xpool.fabric import FabricGenerationPhase
from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.kv.radix import evict_suffix_reclaim_nodes, select_suffix_reclaim_nodes
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import INSTANCE_STARTUP_BARRIER_TIMEOUT_S, InstanceRankError, InstanceRankRuntime

KV_CAPACITY_COMMAND_POLL_INTERVAL_S = 0.01
KV_CAPACITY_GENERATION_CHECK_INTERVAL_S = 0.5


@dataclass(frozen=True, slots=True)
class DemandWitness:
    """One capacity requirement coupled to its request-local SLO evidence."""

    evaluated_sequence: int
    requested_bundles: int
    deadline_monotonic_ns: int


@dataclass(slots=True)
class CapacityReconciler:
    """Reconcile one Capacity Group partition at SGLang scheduler boundaries."""

    channel: xpool.native.kv.InstanceControlChannel
    command_index: int
    backing: KvVmmBacking
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator
    request_pool: ReqToTokenPool
    instance_rank: InstanceRankRuntime
    slo: LatencySloConfig
    command: xpool.native.kv.KvCapacityCommand | None = None
    active_bundles: int = 0
    applied_sequence: int = 0
    completed_sequence: int = 0
    pending_retirement_event: torch.cuda.Event | None = None
    unresolved_demands: dict[tuple[Req, ...], DemandWitness] = field(default_factory=dict)
    published_demand: tuple[int, int | None, int | None] | None = None

    @classmethod
    def attach(
        cls,
        *,
        channel_name: str,
        group_index: int,
        partition_index: int,
        dp_group_count: int,
        backing: KvVmmBacking,
        allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
        request_pool: ReqToTokenPool,
        instance_rank: InstanceRankRuntime,
        slo: LatencySloConfig,
    ) -> CapacityReconciler:
        """Attach the partition channel and publish its bootstrap backing."""

        channel = xpool.native.kv.InstanceControlChannel.attach(
            channel_name,
            group_index=group_index,
            partition_index=partition_index,
            dp_group_count=dp_group_count,
        )
        return cls(
            channel=channel,
            command_index=group_index % dp_group_count,
            backing=backing,
            allocator=allocator,
            request_pool=request_pool,
            instance_rank=instance_rank,
            slo=slo,
            active_bundles=backing.backed_bundles,
        )

    @property
    def draining(self) -> bool:
        """Return whether an immutable shrink is waiting to finish."""

        command = self.command
        return self.pending_retirement_event is not None or (
            command is not None
            and command.sequence > self.applied_sequence
            and command.target_bundles < self.active_bundles
        )

    def accept_command(self, command: xpool.native.kv.KvCapacityCommand | None) -> None:
        """Retain one newer immutable command."""

        if command is None:
            return
        if command.sequence <= self.completed_sequence:
            return
        current = self.command
        if current is not None and command.sequence < current.sequence:
            return
        if current is not None and command.sequence == current.sequence:
            if command.target_bundles != current.target_bundles:
                raise RuntimeError("kv capacity target changed without advancing its sequence")
            return
        if (current is not None and current.sequence > self.completed_sequence) or self.pending_retirement_event:
            raise RuntimeError("kv capacity command superseded an unfinished operation")
        self.command = command

    def check_generation(self) -> None:
        """Require the retained executable Generation while waiting for peers."""

        plan = self.instance_rank.fabric_plan
        if plan is None:
            raise InstanceRankError("kv capacity reconciliation requires an executable fabric generation")
        readiness = self.instance_rank.client.readiness()
        if readiness.generation != plan.generation or readiness.fabric_phase is not FabricGenerationPhase.EXECUTABLE:
            raise InstanceRankError("fabric generation became unavailable during kv capacity reconciliation")

    def vote_ready(self, ready: bool) -> bool:
        """Agree on one command's readiness before any TP rank changes logical capacity."""

        parallel = get_parallel()
        if parallel.attn_tp_size == 1:
            return ready
        group = parallel.attn_tp_group.cpu_group if parallel.enable_dp_attention else parallel.tp_group.cpu_group
        vote = torch.tensor([int(ready)], dtype=torch.int32)
        torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN, group=group)
        return bool(vote.item())

    def apply_command(self, scheduler: Scheduler | None, nodes: tuple[int, ...]) -> None:
        """Switch the logical prefix after the common vote, then finish local physical work."""

        command = self.command
        if command is None:
            raise RuntimeError("kv capacity switch requires a current command")
        growing = command.target_bundles >= self.active_bundles
        target_tokens = self.backing.usable_tokens(command.target_bundles)
        self.allocator.set_token_capacity(target_tokens)
        self.request_pool.reset_aux_cache_allocator()
        if (
            scheduler is not None
            and scheduler.running_batch is not None
            and command.target_bundles > self.active_bundles
        ):
            scheduler.running_batch.batch_is_full = False
        self.active_bundles = command.target_bundles
        self.applied_sequence = command.sequence
        if growing:
            self.channel.publish_completion(
                xpool.native.kv.KvCapacityCompletion(
                    sequence=command.sequence,
                    backed_bundles=self.backing.backed_bundles,
                )
            )
            self.completed_sequence = command.sequence
            return

        if scheduler is None:
            raise RuntimeError("kv capacity startup cannot apply a reclaim operation")
        evict_suffix_reclaim_nodes(scheduler.tree_cache, nodes)
        if not self.allocator.suffix_is_free(target_tokens):
            raise RuntimeError("kv capacity reclaim did not release the complete suffix")
        self.pending_retirement_event = torch.cuda.Event()
        stream = scheduler.forward_stream if scheduler.enable_overlap else scheduler.schedule_stream
        self.pending_retirement_event.record(stream)

    def finish_retirement(self) -> None:
        """Unmap an accepted reclaim after all prior GPU users complete."""

        event = self.pending_retirement_event
        command = self.command
        if event is None or command is None or not event.query():
            return
        self.backing.resize(command.target_bundles)
        self.channel.publish_completion(
            xpool.native.kv.KvCapacityCompletion(
                sequence=command.sequence,
                backed_bundles=self.backing.backed_bundles,
            )
        )
        self.pending_retirement_event = None
        self.completed_sequence = command.sequence

    def begin_scheduling(self, scheduler: Scheduler | None) -> None:
        """Reconcile at most one fixed operation before ordinary batch planning."""

        was_draining = self.draining
        self.finish_retirement()
        if was_draining and not self.draining and scheduler is not None and scheduler.running_batch is not None:
            scheduler.running_batch.batch_is_full = False
        if self.pending_retirement_event is not None:
            return
        command = self.command
        if command is None or command.sequence <= self.applied_sequence:
            return
        nodes = ()
        if command.target_bundles >= self.active_bundles:
            if self.backing.backed_bundles < command.target_bundles:
                self.backing.resize(command.target_bundles)
            ready = True
        else:
            if scheduler is None:
                raise RuntimeError("kv capacity startup cannot reclaim backing")
            selected = select_suffix_reclaim_nodes(
                scheduler.tree_cache,
                self.allocator,
                self.backing.usable_tokens(command.target_bundles),
            )
            ready = selected is not None
            nodes = selected or ()
        if self.vote_ready(ready):
            self.apply_command(scheduler, nodes)

    def deadline_ns(self, origin: float, target_ms: float) -> int:
        """Convert one scheduler-local timing origin and target to monotonic nanoseconds."""

        if not isfinite(origin) or origin <= 0:
            raise RuntimeError("kv capacity demand has no valid scheduler timing origin")
        return int((origin + target_ms / 1000) * 1_000_000_000)

    def record_prefill_requirement(self, request: Req, required_token_capacity: int) -> None:
        """Retain one blocked Prefill request and its TTFT deadline."""

        if get_parallel().attn_tp_rank == 0:
            requests = (request,)
            self.unresolved_demands[requests] = DemandWitness(
                evaluated_sequence=self.applied_sequence,
                requested_bundles=self.backing.required_bundles(required_token_capacity),
                deadline_monotonic_ns=self.deadline_ns(request.time_stats.scheduler_recv_time, self.slo.ttft_ms),
            )

    def record_decode_requirement(self, requests: Iterable[Req], required_token_capacity: int) -> None:
        """Retain one blocked Decode batch and its earliest TBT deadline."""

        if get_parallel().attn_tp_rank != 0:
            return
        request_tuple = tuple(requests)
        if not request_tuple:
            raise RuntimeError("kv capacity Decode demand requires a nonempty batch")
        deadlines = []
        for request in request_tuple:
            stats = request.time_stats
            origin = stats.last_decode_finish_time or stats.last_prefill_finished_time
            deadlines.append(self.deadline_ns(origin, self.slo.tbt_ms))
        self.unresolved_demands[request_tuple] = DemandWitness(
            evaluated_sequence=self.applied_sequence,
            requested_bundles=self.backing.required_bundles(required_token_capacity),
            deadline_monotonic_ns=min(deadlines),
        )

    def finish_scheduling(self, scheduler: Scheduler) -> None:
        """Publish the persistent demand state after ordinary scheduling."""

        if get_parallel().attn_tp_rank != 0 or self.applied_sequence == 0:
            return
        waiting = set(scheduler.waiting_queue)
        self.unresolved_demands = {
            requests: witness
            for requests, witness in self.unresolved_demands.items()
            if witness.evaluated_sequence == self.applied_sequence and all(request in waiting for request in requests)
        }
        if self.unresolved_demands:
            witness = min(self.unresolved_demands.values(), key=lambda item: item.deadline_monotonic_ns)
            evaluated_sequence = witness.evaluated_sequence
            requested_bundles = witness.requested_bundles
            deadline_monotonic_ns = witness.deadline_monotonic_ns
        else:
            evaluated_sequence = self.applied_sequence
            requested_bundles = None
            deadline_monotonic_ns = None
        publication = (evaluated_sequence, requested_bundles, deadline_monotonic_ns)
        if publication == self.published_demand:
            return
        self.channel.publish_demand(
            xpool.native.kv.KvCapacityDemand(
                evaluated_sequence=evaluated_sequence,
                requested_bundles=requested_bundles,
                deadline_monotonic_ns=deadline_monotonic_ns,
            )
        )
        self.published_demand = publication

    def finalize_after_capture(
        self,
        model_runner: ModelRunner,
    ) -> PostCaptureKVResize:
        """Release to the floor, negotiate startup capacity, and expose the service ceiling."""

        torch.cuda.synchronize(model_runner.device)
        self.active_bundles = self.backing.capacity_profile.floor_bundles
        self.allocator.set_token_capacity(self.backing.usable_tokens(self.active_bundles))
        self.backing.resize(self.active_bundles)
        self.request_pool.reset_aux_cache_allocator()
        self.channel.publish_initial_backing(self.backing.backed_bundles)
        self.channel.publish_capture_complete()

        deadline = time.monotonic() + INSTANCE_STARTUP_BARRIER_TIMEOUT_S
        next_generation_check = time.monotonic()
        ceiling = self.channel.service_ceiling()
        while time.monotonic() < deadline:
            commands = self.channel.read_commands()
            self.accept_command(commands[self.command_index])
            ceiling = ceiling or self.channel.service_ceiling()
            if self.command is not None:
                self.begin_scheduling(None)
            if self.applied_sequence != 0 and ceiling is not None:
                break
            now = time.monotonic()
            if now >= next_generation_check:
                self.check_generation()
                next_generation_check = now + KV_CAPACITY_GENERATION_CHECK_INTERVAL_S
            time.sleep(KV_CAPACITY_COMMAND_POLL_INTERVAL_S)
        else:
            raise InstanceRankError("timed out waiting for initial kv capacity activation")

        ceiling_tokens = self.backing.usable_tokens(ceiling)
        memory_pool_config = model_runner.memory_pool_config
        return PostCaptureKVResize(
            max_total_num_tokens=ceiling_tokens,
            full_max_total_num_tokens=(
                None
                if memory_pool_config is None or memory_pool_config.full_max_total_num_tokens is None
                else min(memory_pool_config.full_max_total_num_tokens, ceiling_tokens)
            ),
            swa_max_total_num_tokens=(
                None
                if memory_pool_config is None or memory_pool_config.swa_max_total_num_tokens is None
                else min(memory_pool_config.swa_max_total_num_tokens, ceiling_tokens)
            ),
            capped_max_running_requests=None,
        )

    def close(self) -> None:
        """Release this Instance process's local channel mapping."""

        self.pending_retirement_event = None
        self.channel.close()
