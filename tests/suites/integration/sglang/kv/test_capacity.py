from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed
import torch.multiprocessing
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_parallel

import xpool.integrations.sglang.kv.capacity
import xpool.native
from tests.harness.support.sglang.fakes import ServerArgs
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.config import LatencySloConfig
from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import InstanceRankRuntime

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)
TEST_SLO = LatencySloConfig(ttft_ms=1000, tbt_ms=50)


class FakeBacking:
    """Minimal compound backing used by reconciliation tests."""

    capacity_profile = SimpleNamespace(floor_bundles=1)

    def __init__(self, backed_bundles: int = 2) -> None:
        self.backed_bundles = backed_bundles
        self.resize_calls: list[int] = []

    def usable_tokens(self, bundle_count: int) -> int:
        return bundle_count * 4

    def required_bundles(self, token_count: int) -> int:
        return (token_count + 3) // 4

    def resize(self, bundle_count: int) -> None:
        self.resize_calls.append(bundle_count)
        self.backed_bundles = bundle_count


class FakeControlChannel:
    """Single-partition in-memory Control Channel."""

    def __init__(self, command: xpool.native.kv.KvCapacityCommand | None = None, ceiling: int = 4) -> None:
        self.commands = [command]
        self.ceiling = ceiling
        self.initial_backing: list[int] = []
        self.capture_complete = False
        self.completions: list[xpool.native.kv.KvCapacityCompletion] = []
        self.demands: list[xpool.native.kv.KvCapacityDemand] = []

    def publish_initial_backing(self, bundles: int) -> None:
        self.initial_backing.append(bundles)

    def publish_capture_complete(self) -> None:
        self.capture_complete = True

    def service_ceiling(self) -> int:
        return self.ceiling

    def read_commands(self) -> list[xpool.native.kv.KvCapacityCommand | None]:
        return self.commands

    def publish_completion(self, completion: xpool.native.kv.KvCapacityCompletion) -> None:
        self.completions.append(completion)

    def publish_demand(self, demand: xpool.native.kv.KvCapacityDemand) -> None:
        self.demands.append(demand)


def make_reconciler(
    channel: FakeControlChannel,
    backing: FakeBacking,
    allocator: object,
    request_pool: object,
    *,
    command: xpool.native.kv.KvCapacityCommand | None = None,
    active_bundles: int = 0,
    applied_sequence: int = 0,
    completed_sequence: int = 0,
) -> CapacityReconciler:
    return CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceControlChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, backing),
        allocator=cast(ElasticTokenToKVPoolAllocator, allocator),
        request_pool=cast(ReqToTokenPool, request_pool),
        instance_rank=cast(InstanceRankRuntime, object()),
        slo=TEST_SLO,
        command=command,
        active_bundles=active_bundles,
        applied_sequence=applied_sequence,
        completed_sequence=completed_sequence,
    )


def run_tp2_readiness_vote(rank: int, rendezvous_uri: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous_uri,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        get_context().set_server_args(ServerArgs(model_path="dummy", tp_size=2, dp_size=1, enable_dp_attention=False))
        reconciler = make_reconciler(FakeControlChannel(), FakeBacking(), object(), object())
        with get_parallel().override(
            attn_tp_size=2,
            tp_group=SimpleNamespace(cpu_group=torch.distributed.group.WORLD),
        ):
            assert not reconciler.vote_ready(rank == 0)
            assert reconciler.vote_ready(True)
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_readiness_vote_requires_every_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5]))
    torch.multiprocessing.spawn(
        run_tp2_readiness_vote,
        args=((tmp_path / "readiness-vote").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_post_capture_finalization_publishes_floor_once_then_activates_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=3)
    channel = FakeControlChannel(command)
    backing = FakeBacking()
    capacities: list[int] = []
    allocator = SimpleNamespace(set_token_capacity=capacities.append)
    resets: list[None] = []
    request_pool = SimpleNamespace(reset_aux_cache_allocator=lambda: resets.append(None))
    model_runner = SimpleNamespace(device="cuda:0", memory_pool_config=None)
    reconciler = make_reconciler(channel, backing, allocator, request_pool)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        result = reconciler.finalize_after_capture(cast(ModelRunner, model_runner))

    assert backing.resize_calls == [1, 3]
    assert channel.initial_backing == [1]
    assert channel.capture_complete
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(1, 3)]
    assert capacities == [4, 12]
    assert resets == [None, None]
    assert reconciler.applied_sequence == reconciler.completed_sequence == 1
    assert result.max_total_num_tokens == 16


def test_reclaim_drains_until_the_exact_suffix_becomes_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)
    channel = FakeControlChannel(command)
    backing = FakeBacking(8)
    reconciler = make_reconciler(
        channel,
        backing,
        SimpleNamespace(),
        SimpleNamespace(),
        command=command,
        active_bundles=8,
        applied_sequence=1,
        completed_sequence=1,
    )
    probes: list[int] = []

    def select_nodes(tree_cache: object, allocator: object, token_capacity: int) -> tuple[int, ...] | None:
        probes.append(token_capacity)
        return None if len(probes) == 1 else (3, 2)

    monkeypatch.setattr(xpool.integrations.sglang.kv.capacity, "select_suffix_reclaim_nodes", select_nodes)
    scheduler = cast(Scheduler, SimpleNamespace(chunked_req=None, tree_cache=object()))

    monkeypatch.setattr(CapacityReconciler, "vote_ready", lambda self, ready: False)
    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.begin_scheduling(scheduler)
        assert reconciler.draining
        reconciler.begin_scheduling(scheduler)

    assert probes == [4, 4]
    assert channel.completions == []
    assert backing.resize_calls == []
    assert reconciler.active_bundles == 8
    assert reconciler.applied_sequence == 1
    assert reconciler.completed_sequence == 1


def test_growth_maps_before_exposing_capacity_and_completes_at_the_boundary() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=3)
    channel = FakeControlChannel(command)
    backing = FakeBacking(2)
    capacities: list[int] = []
    running_batch = SimpleNamespace(batch_is_full=True)
    reconciler = make_reconciler(
        channel,
        backing,
        SimpleNamespace(set_token_capacity=capacities.append),
        SimpleNamespace(reset_aux_cache_allocator=lambda: None),
        command=command,
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )
    scheduler = cast(Scheduler, SimpleNamespace(chunked_req=object(), running_batch=running_batch))

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.begin_scheduling(scheduler)

    assert backing.resize_calls == [3]
    assert capacities == [12]
    assert not running_batch.batch_is_full
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(2, 3)]
    assert reconciler.active_bundles == 3
    assert reconciler.applied_sequence == reconciler.completed_sequence == 2


def test_accepted_reclaim_defers_unmap_and_completion_until_cuda_retires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)
    channel = FakeControlChannel(command)
    backing = FakeBacking(8)
    capacities: list[int] = []
    allocator = SimpleNamespace(
        set_token_capacity=capacities.append,
        suffix_is_free=lambda token_capacity: True,
    )
    evictions: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.capacity,
        "select_suffix_reclaim_nodes",
        lambda tree_cache, selected_allocator, token_capacity: (3, 2),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.capacity,
        "evict_suffix_reclaim_nodes",
        lambda tree_cache, nodes: evictions.append(nodes),
    )

    class Event:
        ready = False

        def record(self, stream: object) -> None:
            assert stream == "execution"

        def query(self) -> bool:
            return self.ready

    event = Event()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    running_batch = SimpleNamespace(batch_is_full=True)
    reconciler = make_reconciler(
        channel,
        backing,
        allocator,
        SimpleNamespace(reset_aux_cache_allocator=lambda: None),
        command=command,
        active_bundles=8,
        applied_sequence=1,
        completed_sequence=1,
    )
    scheduler = cast(
        Scheduler,
        SimpleNamespace(
            chunked_req=None,
            tree_cache=object(),
            running_batch=running_batch,
            enable_overlap=False,
            schedule_stream="execution",
            forward_stream="overlap",
        ),
    )

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.begin_scheduling(scheduler)
        assert channel.completions == []
        event.ready = True
        reconciler.begin_scheduling(scheduler)

    assert capacities == [4]
    assert evictions == [(3, 2)]
    assert backing.resize_calls == [1]
    assert [(item.sequence, item.backed_bundles) for item in channel.completions] == [(2, 1)]
    assert reconciler.applied_sequence == reconciler.completed_sequence == 2
    assert not running_batch.batch_is_full


class TimedRequest(SimpleNamespace):
    """Hashable request stub carrying SGLang scheduler timestamps."""

    __hash__ = object.__hash__


def test_prefill_demand_uses_scheduler_receipt_deadline_until_the_request_leaves() -> None:
    channel = FakeControlChannel()
    backing = FakeBacking()
    request = cast(
        Req,
        TimedRequest(
            time_stats=SimpleNamespace(
                scheduler_recv_time=10.0,
                last_decode_finish_time=0.0,
                last_prefill_finished_time=0.0,
            )
        ),
    )
    reconciler = make_reconciler(
        channel,
        backing,
        object(),
        object(),
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.record_prefill_requirement(request, 13)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request])))
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request])))
        reconciler.applied_sequence = 2
        reconciler.record_prefill_requirement(request, 9)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request])))
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[])))

    assert [
        (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns)
        for demand in channel.demands
    ] == [
        (1, 4, 11_000_000_000),
        (2, 3, 11_000_000_000),
        (2, None, None),
    ]


def test_decode_batch_keeps_its_target_with_earliest_request_deadline() -> None:
    channel = FakeControlChannel()
    requests = (
        cast(
            Req,
            TimedRequest(
                time_stats=SimpleNamespace(
                    scheduler_recv_time=1.0,
                    last_decode_finish_time=20.0,
                    last_prefill_finished_time=19.0,
                )
            ),
        ),
        cast(
            Req,
            TimedRequest(
                time_stats=SimpleNamespace(
                    scheduler_recv_time=2.0,
                    last_decode_finish_time=0.0,
                    last_prefill_finished_time=18.0,
                )
            ),
        ),
    )
    reconciler = make_reconciler(
        channel,
        FakeBacking(),
        object(),
        object(),
        active_bundles=2,
        applied_sequence=1,
        completed_sequence=1,
    )

    with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
        reconciler.record_decode_requirement(requests, 17)
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=list(requests))))

    demand = channel.demands[-1]
    assert (demand.evaluated_sequence, demand.requested_bundles, demand.deadline_monotonic_ns) == (
        1,
        5,
        18_050_000_000,
    )


def test_new_command_cannot_supersede_an_unfinished_operation() -> None:
    command = xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1)
    reconciler = make_reconciler(
        FakeControlChannel(command),
        FakeBacking(),
        object(),
        object(),
        command=command,
        applied_sequence=1,
        completed_sequence=1,
    )

    with pytest.raises(RuntimeError, match="superseded"):
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=3, target_bundles=4))
    with pytest.raises(RuntimeError, match="target changed"):
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=2))
