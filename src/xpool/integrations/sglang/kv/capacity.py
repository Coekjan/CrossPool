"""Instance-local elastic KV capacity reconciliation."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import PostCaptureKVResize
from sglang.srt.runtime_context import get_parallel

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationPhase
from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import INSTANCE_STARTUP_BARRIER_TIMEOUT_S, InstanceRankError, InstanceRankRuntime

KV_CAPACITY_COMMAND_POLL_INTERVAL_S = 0.01
KV_CAPACITY_GENERATION_CHECK_INTERVAL_S = 0.5


@dataclass(slots=True)
class CapacityReconciler:
    """Own one Instance partition's channel state and ordered command view."""

    channel: xpool.native.kv.InstanceCapacityChannel
    command_index: int
    backing: KvVmmBacking
    allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator
    request_pool: ReqToTokenPool
    command: xpool.native.kv.KvCapacityCommand | None = None
    prepared_sequence: int = 0
    pending_shrink_event: torch.cuda.Event | None = None
    unresolved_requests: set[Req] = field(default_factory=set)
    last_pressure_active_bundles: int | None = None

    @classmethod
    def attach(
        cls,
        *,
        channel_name: str,
        group_index: int,
        partition_index: int,
        group_count: int,
        backing: KvVmmBacking,
        allocator: ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator,
        request_pool: ReqToTokenPool,
    ) -> CapacityReconciler:
        """Attach the partition channel and publish its bootstrap backing."""

        channel = xpool.native.kv.InstanceCapacityChannel.attach(
            channel_name,
            group_index=group_index,
            partition_index=partition_index,
            group_count=group_count,
        )
        channel.publish_backing_report(
            xpool.native.kv.KvCapacityBackingReport(
                prepared_sequence=0,
                backed_bundles=backing.backed_bundles,
            )
        )
        return cls(
            channel=channel,
            command_index=group_index % get_global_config().atn_world_size,
            backing=backing,
            allocator=allocator,
            request_pool=request_pool,
        )

    def publish_backing_report(self, prepared_sequence: int) -> None:
        """Publish current physical backing for one command sequence."""

        self.channel.publish_backing_report(
            xpool.native.kv.KvCapacityBackingReport(
                prepared_sequence=prepared_sequence,
                backed_bundles=self.backing.backed_bundles,
            )
        )
        self.prepared_sequence = prepared_sequence

    def accept_command(self, command: xpool.native.kv.KvCapacityCommand | None) -> None:
        """Retain one newer coherent command or active-capacity advancement."""

        if command is None:
            return
        current = self.command
        if current is None or command.sequence > current.sequence:
            self.pending_shrink_event = None
            self.command = command
            return
        if command.sequence < current.sequence:
            return
        if command.target_bundles != current.target_bundles:
            raise RuntimeError("kv capacity target changed without advancing its sequence")
        if command.active_bundles > current.active_bundles:
            self.command = command

    def begin_scheduling(self, scheduler: Scheduler) -> None:
        """Advance logical admission and rank-local physical backing."""

        command = self.command
        if command is None:
            return
        allocator = self.allocator
        backing = self.backing
        active_tokens = backing.usable_tokens(command.active_bundles)
        target_tokens = backing.usable_tokens(command.target_bundles)

        if allocator.token_capacity != active_tokens:
            grew = active_tokens > allocator.token_capacity
            allocator.set_token_capacity(active_tokens)
            self.request_pool.reset_aux_cache_allocator()
            if grew and scheduler.running_batch is not None:
                scheduler.running_batch.batch_is_full = False

        suffix_is_free = allocator.suffix_is_free(target_tokens)
        if not suffix_is_free:
            scheduler.tree_cache.evict(EvictParams(num_tokens=1))
            suffix_is_free = allocator.suffix_is_free(target_tokens)

        if backing.backed_bundles < command.target_bundles:
            backing.resize(command.target_bundles)

        if self.prepared_sequence != command.sequence:
            self.publish_backing_report(command.sequence)

        if backing.backed_bundles <= command.target_bundles or not suffix_is_free:
            return
        if self.pending_shrink_event is None:
            self.pending_shrink_event = torch.cuda.Event()
            stream = scheduler.forward_stream if scheduler.enable_overlap else scheduler.schedule_stream
            self.pending_shrink_event.record(stream)
            return
        if not self.pending_shrink_event.query():
            return

        backing.resize(command.target_bundles)
        self.publish_backing_report(command.sequence)
        self.pending_shrink_event = None

    def record_pressure(self, requests: Iterable[Req]) -> None:
        """Publish one leader-local admission-pressure edge."""

        if get_parallel().attn_tp_rank != 0:
            return
        command = self.command
        if command is None:
            return
        self.unresolved_requests.update(requests)
        if self.last_pressure_active_bundles == command.active_bundles:
            return
        self.channel.publish_pressure(command.active_bundles)
        self.last_pressure_active_bundles = command.active_bundles

    def finish_scheduling(self, scheduler: Scheduler) -> None:
        """Resolve pressure after rejected requests leave the waiting queue."""

        if not self.unresolved_requests:
            return
        self.unresolved_requests.intersection_update(scheduler.waiting_queue)
        if not self.unresolved_requests:
            self.channel.publish_pressure(None)
            self.last_pressure_active_bundles = None

    def finalize_after_capture(
        self,
        model_runner: ModelRunner,
        instance_rank: InstanceRankRuntime,
    ) -> PostCaptureKVResize:
        """Publish capture completion and wait for coordinated initial capacity."""

        allocator = self.allocator
        backing = self.backing
        torch.cuda.synchronize(model_runner.device)
        floor_tokens = backing.usable_tokens(backing.floor_bundles)
        allocator.set_token_capacity(floor_tokens)
        backing.resize(backing.floor_bundles)
        self.request_pool.reset_aux_cache_allocator()
        self.publish_backing_report(0)
        self.channel.publish_capture_complete()

        deadline = time.monotonic() + INSTANCE_STARTUP_BARRIER_TIMEOUT_S
        next_generation_check = time.monotonic()
        while time.monotonic() < deadline:
            commands = self.channel.read_commands()
            self.accept_command(commands[self.command_index])
            command = self.command
            if command is not None:
                if self.prepared_sequence != command.sequence:
                    backing.resize(command.target_bundles)
                    self.publish_backing_report(command.sequence)
                parallel = get_parallel()
                covered_commands = (
                    commands
                    if parallel.enable_dp_attention
                    and not parallel.enable_dp_attention_local_control_broadcast
                    and parallel.tp_rank == 0
                    else (command,)
                )
                if command.active_bundles == command.target_bundles and all(
                    covered is not None and covered.active_bundles == covered.target_bundles
                    for covered in covered_commands
                ):
                    allocator.set_token_capacity(backing.usable_tokens(command.active_bundles))
                    self.request_pool.reset_aux_cache_allocator()
                    memory_pool_config = model_runner.memory_pool_config
                    return PostCaptureKVResize(
                        max_total_num_tokens=model_runner.max_total_num_tokens,
                        full_max_total_num_tokens=(
                            None if memory_pool_config is None else memory_pool_config.full_max_total_num_tokens
                        ),
                        swa_max_total_num_tokens=(
                            None if memory_pool_config is None else memory_pool_config.swa_max_total_num_tokens
                        ),
                        capped_max_running_requests=None,
                    )

            now = time.monotonic()
            if now >= next_generation_check:
                if instance_rank.fabric_plan is None:
                    raise InstanceRankError("kv capacity finalization requires an executable fabric generation")
                readiness = instance_rank.client.readiness()
                if (
                    readiness.generation != instance_rank.fabric_plan.generation
                    or readiness.fabric_phase is not FabricGenerationPhase.EXECUTABLE
                ):
                    raise InstanceRankError("fabric generation became unavailable during kv capacity finalization")
                next_generation_check = now + KV_CAPACITY_GENERATION_CHECK_INTERVAL_S
            time.sleep(KV_CAPACITY_COMMAND_POLL_INTERVAL_S)
        raise InstanceRankError("timed out waiting for initial kv capacity activation")

    def close(self) -> None:
        """Release this Instance process's local channel mapping."""

        self.pending_shrink_event = None
        self.channel.close()
