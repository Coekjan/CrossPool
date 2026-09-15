from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_parallel

import xpool.native
from tests.harness.support.sglang.runtime import published_sglang_config
from xpool.fabric import FabricGenerationId, FabricGenerationPhase
from xpool.integrations.sglang.kv.allocator import ElasticTokenToKVPoolAllocator
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.runtime.instance import InstanceRankRuntime

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


class FakeBacking:
    """Minimal backing state needed by initial capacity reconciliation."""

    floor_bundles = 1

    def __init__(self) -> None:
        self.backed_bundles = 2
        self.resize_calls: list[int] = []

    def usable_tokens(self, bundle_count: int) -> int:
        return bundle_count * 4

    def resize(self, bundle_count: int) -> None:
        self.resize_calls.append(bundle_count)
        self.backed_bundles = bundle_count


class FakeCapacityChannel:
    """In-memory channel seam for one immediately activated command."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, int]] = []
        self.capture_complete = False
        self.pressure: list[int | None] = []

    def publish_backing_report(self, report: xpool.native.kv.KvCapacityBackingReport) -> None:
        self.reports.append((report.prepared_sequence, report.backed_bundles))

    def publish_capture_complete(self) -> None:
        self.capture_complete = True

    def read_commands(self) -> list[xpool.native.kv.KvCapacityCommand]:
        return [xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=3, active_bundles=3)]

    def publish_pressure(self, active_bundles: int | None) -> None:
        self.pressure.append(active_bundles)


def test_post_capture_finalization_releases_to_floor_then_activates_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = FakeBacking()
    channel = FakeCapacityChannel()
    operations: list[str] = []
    allocator_capacities: list[int] = []
    request_resets: list[None] = []
    original_resize = backing.resize
    original_publish_backing_report = channel.publish_backing_report
    original_publish_capture_complete = channel.publish_capture_complete

    def resize(bundle_count: int) -> None:
        operations.append(f"resize:{bundle_count}")
        original_resize(bundle_count)

    def set_token_capacity(token_count: int) -> None:
        operations.append(f"capacity:{token_count}")
        allocator_capacities.append(token_count)

    def publish_backing_report(report: xpool.native.kv.KvCapacityBackingReport) -> None:
        operations.append(f"report:{report.backed_bundles}")
        original_publish_backing_report(report)

    def publish_capture_complete() -> None:
        operations.append("capture-complete")
        original_publish_capture_complete()

    monkeypatch.setattr(backing, "resize", resize)
    monkeypatch.setattr(channel, "publish_backing_report", publish_backing_report)
    monkeypatch.setattr(channel, "publish_capture_complete", publish_capture_complete)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: operations.append(f"synchronize:{device}"))
    allocator = SimpleNamespace(set_token_capacity=set_token_capacity)
    request_pool = SimpleNamespace(reset_aux_cache_allocator=lambda: request_resets.append(None))
    model_runner = SimpleNamespace(
        device="cuda:0",
        memory_pool_config=None,
        max_total_num_tokens=64,
    )
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, backing),
        allocator=cast(ElasticTokenToKVPoolAllocator, allocator),
        request_pool=cast(ReqToTokenPool, request_pool),
    )

    result = reconciler.finalize_after_capture(
        cast(ModelRunner, model_runner),
        cast(InstanceRankRuntime, object()),
    )

    assert operations[:5] == ["synchronize:cuda:0", "capacity:4", "resize:1", "report:1", "capture-complete"]
    assert backing.resize_calls == [1, 3]
    assert channel.reports == [(0, 1), (1, 3)]
    assert channel.capture_complete
    assert allocator_capacities == [4, 12]
    assert request_resets == [None, None]
    assert result.max_total_num_tokens == 64
    assert result.capped_max_running_requests is None


@pytest.mark.parametrize(("local_control", "expected_reads"), [(False, 2), (True, 1)])
def test_post_capture_finalization_waits_for_the_broadcast_source_scope(
    monkeypatch: pytest.MonkeyPatch,
    local_control: bool,
    expected_reads: int,
) -> None:
    backing = FakeBacking()

    class Channel(FakeCapacityChannel):
        reads = 0

        def read_commands(self) -> list[xpool.native.kv.KvCapacityCommand]:
            self.reads += 1
            peer_active = 2 if self.reads > 1 else 1
            return [
                xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=3, active_bundles=3),
                xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=2, active_bundles=peer_active),
            ]

    channel = Channel()
    allocator = SimpleNamespace(set_token_capacity=lambda capacity: None)
    request_pool = SimpleNamespace(reset_aux_cache_allocator=lambda: None)
    model_runner = SimpleNamespace(
        device="cuda:0",
        memory_pool_config=None,
        max_total_num_tokens=64,
    )
    generation = FabricGenerationId(high=1, low=1)
    instance_rank = SimpleNamespace(
        fabric_plan=SimpleNamespace(generation=generation),
        client=SimpleNamespace(
            readiness=lambda: SimpleNamespace(
                generation=generation,
                fabric_phase=FabricGenerationPhase.EXECUTABLE,
            )
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, backing),
        allocator=cast(ElasticTokenToKVPoolAllocator, allocator),
        request_pool=cast(ReqToTokenPool, request_pool),
    )

    with (
        get_context().override_server_args(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=local_control,
        ),
        get_parallel().override(tp_rank=0),
    ):
        reconciler.finalize_after_capture(
            cast(ModelRunner, model_runner),
            cast(InstanceRankRuntime, instance_rank),
        )

    assert channel.reads == expected_reads


def test_service_shrink_waits_for_allocator_suffix_and_cuda_event(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator = ElasticTokenToKVPoolAllocator(8, torch.float16, "cpu", cast(KVCache, object()), False)
    allocated = allocator.alloc(8)
    assert allocated is not None
    backing = FakeBacking()
    backing.backed_bundles = 8
    channel = FakeCapacityChannel()

    class Event:
        ready = False

        def record(self, stream: object) -> None:
            assert stream == "execution"

        def query(self) -> bool:
            return self.ready

    event = Event()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)

    class TreeCache:
        calls = 0

        def evict(self, params: object) -> None:
            self.calls += 1
            if self.calls == 2:
                allocator.free(allocated[4:])

    resets: list[None] = []
    request_pool = SimpleNamespace(reset_aux_cache_allocator=lambda: resets.append(None))
    scheduler = cast(
        Scheduler,
        SimpleNamespace(
            tree_cache=TreeCache(),
            running_batch=SimpleNamespace(batch_is_full=True),
            enable_overlap=False,
            schedule_stream="execution",
            forward_stream="forward",
        ),
    )
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, backing),
        allocator=allocator,
        request_pool=cast(ReqToTokenPool, request_pool),
        command=xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1, active_bundles=1),
    )

    reconciler.begin_scheduling(scheduler)
    assert allocator.token_capacity == 4
    assert backing.resize_calls == []
    assert channel.reports == [(2, 8)]
    assert reconciler.pending_shrink_event is None

    reconciler.begin_scheduling(scheduler)
    assert backing.resize_calls == []
    assert reconciler.pending_shrink_event is event

    event.ready = True
    reconciler.begin_scheduling(scheduler)
    assert backing.resize_calls == [1]
    assert channel.reports == [(2, 8), (2, 1)]
    assert reconciler.pending_shrink_event is None


def test_new_command_retires_pending_reclamation_before_activation_advances() -> None:
    pending = cast(torch.cuda.Event, object())
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, FakeCapacityChannel()),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
        command=xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1, active_bundles=1),
        pending_shrink_event=pending,
    )

    reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=3, target_bundles=4, active_bundles=1))
    assert reconciler.pending_shrink_event is None
    assert reconciler.command is not None
    assert (
        reconciler.command.sequence,
        reconciler.command.target_bundles,
        reconciler.command.active_bundles,
    ) == (3, 4, 1)

    reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=2, target_bundles=1, active_bundles=1))
    reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=3, target_bundles=4, active_bundles=4))
    assert reconciler.command is not None
    assert (
        reconciler.command.sequence,
        reconciler.command.target_bundles,
        reconciler.command.active_bundles,
    ) == (3, 4, 4)
    with pytest.raises(RuntimeError, match="target changed"):
        reconciler.accept_command(xpool.native.kv.KvCapacityCommand(sequence=3, target_bundles=3, active_bundles=3))


def test_pressure_is_deduplicated_until_rejected_requests_leave_waiting_queue() -> None:
    channel = FakeCapacityChannel()
    request = cast(Req, object())
    reconciler = CapacityReconciler(
        channel=cast(xpool.native.kv.InstanceCapacityChannel, channel),
        command_index=0,
        backing=cast(KvVmmBacking, object()),
        allocator=cast(ElasticTokenToKVPoolAllocator, object()),
        request_pool=cast(ReqToTokenPool, object()),
        command=xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=3, active_bundles=2),
    )

    with get_parallel().override(attn_tp_rank=0):
        reconciler.record_pressure([request])
        reconciler.record_pressure([request])
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[request])))
        reconciler.finish_scheduling(cast(Scheduler, SimpleNamespace(waiting_queue=[])))

    assert channel.pressure == [2, None]
