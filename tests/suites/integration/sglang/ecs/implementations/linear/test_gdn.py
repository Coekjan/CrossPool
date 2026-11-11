from array import array
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import torch
import torch.distributed
import torch.multiprocessing
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import track_mamba_states_all_layers
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType, Mamba2StateShape
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.qwen3_5 import Qwen3_5Config
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner_components.layer_setup import ModelLayerInfo
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import SpecAuxHiddenStateConfig
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from tests.suites.integration.sglang.ecs.support import install_vmm_driver

import xpool.integrations.sglang.ecs.state_vmm
import xtest
from xkit.results import write_json
from xpool import ecs
from xpool.config import LatencySloConfig
from xpool.integrations.sglang.ecs.allocation import SglangCapacityMiss
from xpool.integrations.sglang.ecs.capacity import CapacityReconciler, capacity_reconciler_scope
from xpool.integrations.sglang.ecs.construction import state_adapter_scope
from xpool.integrations.sglang.ecs.implementations.linear.gdn import (
    ElasticMambaPool,
    ElasticMambaSlotAllocator,
    GdnStateAdapter,
    prepared_state_backing,
    state_hooks,
    state_layouts,
)
from xpool.integrations.sglang.ecs.radix import StateCacheProposal
from xpool.integrations.sglang.ecs.state_vmm import StateVmmBacking
from xpool.integrations.sglang.hooks.ecs import ElasticContextStateHookSet, complete_allocation
from xpool.native.ecs import InstanceControlChannel
from xpool.runtime.instance import InstanceRankRuntime
from xtest.harness.native.case import run_native_case
from xtest.harness.support.sglang.fakes import server_args
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


def cache_params(*, full_size: bool = False) -> Mamba2CacheParams:
    return Mamba2CacheParams(
        layers=list(range(24 if full_size else 2)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        shape=Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096 if full_size else 8,
            n_groups=16 if full_size else 2,
            num_heads=32 if full_size else 2,
            head_dim=128 if full_size else 4,
            state_size=128 if full_size else 4,
            conv_kernel=4,
        ),
    )


@pytest.fixture
def state_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[ElasticMambaPool]:
    install_vmm_driver(monkeypatch)
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.state_vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    params = cache_params()
    layouts = state_layouts(params, 8, len(params.layers), 16)
    backing = StateVmmBacking(layouts, ecs.MemoryQuota(1 << 20), device=torch.device("cuda:0"))
    token = prepared_state_backing.set(backing)
    try:
        yield ElasticMambaPool(
            size=8, spec_state_size=0, cache_params=params, mamba_layer_ids=params.layers, device="cpu"
        )
    finally:
        prepared_state_backing.reset(token)
        backing.close()


def test_slot_allocation_delivers_backed_ids_and_preserves_ownership_on_quota_miss(
    state_pool: ElasticMambaPool,
) -> None:
    backing = state_pool.backing
    slots = ElasticMambaSlotAllocator(8, "cpu", backing)
    before = slots.plan_slots(2)
    assert before is not None
    backing.quota.set_limit(backing.backed_bytes)
    assert not backing.try_map(backing.coverage(before))
    assert slots.alloc(1) is None
    backing.quota.set_limit(1 << 20)
    assert backing.try_map(backing.coverage(before))
    slots.publish_resident()
    acquired = slots.alloc(2)
    assert acquired is not None
    epochs = dict(slots.owned)
    assert len(epochs) == 2
    assert backing.empty_granules(tuple(slots.owned)) == ()
    slots.free(acquired[:1])
    returned = slots.alloc(1)
    assert returned is not None
    assert slots.owned[int(returned.item())] > max(epochs.values())
    with pytest.raises(RuntimeError, match="ownership"):
        slots.free(torch.tensor([7]))


def test_real_pool_clear_and_copy_preserve_source_and_unrelated_state(state_pool: ElasticMambaPool) -> None:
    assert state_pool.backing.try_map(state_pool.backing.coverage((1, 3, 6)))
    for tensor in (*state_pool.mamba_cache.conv, state_pool.mamba_cache.temporal):
        tensor[:, 1] = 7
        tensor[:, 3] = 13
        tensor[:, 6] = 19
    state_pool.copy_from(torch.tensor([1]), torch.tensor([3]))
    for tensor in (*state_pool.mamba_cache.conv, state_pool.mamba_cache.temporal):
        assert torch.all(tensor[:, 1] == 7)
        assert torch.all(tensor[:, 3] == 7)
        assert torch.all(tensor[:, 6] == 19)
    state_pool.clear_slots(torch.tensor([3]))
    for tensor in (*state_pool.mamba_cache.conv, state_pool.mamba_cache.temporal):
        assert torch.all(tensor[:, 3] == 0)
        assert torch.all(tensor[:, 1] == 7)


@xtest.requirements(device_count=1)
def test_real_state_tracking_and_graph_replay_survive_sparse_remap() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    params = cache_params(full_size=True)
    layouts = state_layouts(params, 16, 24, StateVmmBacking.granularity(device))
    quota = ecs.MemoryQuota(sum(layout.reserved_bytes for layout in layouts))
    backing = StateVmmBacking(layouts, quota, device=device)
    token = prepared_state_backing.set(backing)
    try:
        pool = ElasticMambaPool(
            size=16, spec_state_size=0, cache_params=params, mamba_layer_ids=params.layers, device=str(device)
        )
        assert backing.try_map(backing.coverage((1, 7, 13)))
        for tensor in (*pool.mamba_cache.conv, pool.mamba_cache.temporal):
            tensor[:, 1] = 7
            tensor[:, 13] = 19
        source = torch.tensor([1], dtype=torch.int64, device=device)
        destination = torch.tensor([7], dtype=torch.int64, device=device)
        mask = torch.tensor([True], device=device)
        pool.copy_from(source, destination)
        track_mamba_states_all_layers(pool.mamba_cache.conv[0], pool.mamba_cache.temporal, source, mask, destination, 1)
        torch.cuda.synchronize(device)
        pointers = tuple(t.data_ptr() for t in backing.tensors)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            track_mamba_states_all_layers(
                pool.mamba_cache.conv[0], pool.mamba_cache.temporal, source, mask, destination, 1
            )
        backing.unmap(backing.empty_granules((1, 13)))
        assert backing.try_map(backing.coverage((7,)))
        source.fill_(13)
        graph.replay()
        torch.cuda.synchronize(device)
        for tensor in (*pool.mamba_cache.conv, pool.mamba_cache.temporal):
            assert torch.all(tensor[:, 7] == 19)
            assert torch.all(tensor[:, 1] == 7)
        assert pointers == tuple(t.data_ptr() for t in backing.tensors)
    finally:
        prepared_state_backing.reset(token)
        torch.cuda.synchronize(device)
        backing.close()
    assert quota.charged_bytes == 0


@xtest.requirements(device_count=1)
def test_physical_mapping_amplification_for_original_state_strides(task_artifact_dir: Path | None) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    params = cache_params(full_size=True)
    observations = []
    for recommended in (False, True):
        granularity = StateVmmBacking.granularity(device, recommended=recommended)
        for capacity in (16, 128):
            layouts = state_layouts(params, capacity, 24, granularity)
            for units in ((0, 1), (0, capacity // 2), (0, 1, capacity // 2, capacity - 1)):
                physical = sum(len(layout.coverage(units)) * granularity for layout in layouts)
                logical = len(units) * params.mamba_cache_per_req
                assert physical >= logical
                observations.append(
                    dict(
                        recommended=recommended,
                        granularity_bytes=granularity,
                        capacity=capacity,
                        units=units,
                        physical_bytes=physical,
                        logical_bytes=logical,
                        amplification=physical / logical,
                    )
                )
    if task_artifact_dir is not None:
        write_json(task_artifact_dir / "state-mapping-amplification.json", observations)


@xtest.requirements(device_count=1)
def test_original_hybrid_factory_and_state_receipt_rollback(tmp_path: Path) -> None:
    run_native_case(run_hybrid_factory, tmp_path / "model", workdir=tmp_path / "child")


def run_hybrid_factory(model_path: Path) -> None:
    build_hybrid_case(model_path)


@xtest.requirements(device_count=2)
def test_tp_hybrid_state_matches_logical_eviction_and_rolls_back_after_cow(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        run_hybrid_rank,
        args=(tmp_path, (tmp_path / "gloo").as_uri()),
        nprocs=2,
        join=True,
    )


def run_hybrid_rank(rank: int, workdir: Path, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    group = object.__new__(GroupCoordinator)
    group.cpu_group = torch.distributed.group.WORLD
    group.rank = group.rank_in_group = rank
    group.ranks, group.world_size, group.mq_broadcaster = [0, 1], 2, None
    try:
        build_hybrid_case(workdir / str(rank), group=group)
    finally:
        torch.distributed.destroy_process_group()


def build_hybrid_case(model_path: Path, *, group: GroupCoordinator | None = None) -> None:
    for hook in (*ElasticContextStateHookSet().hooks(), *state_hooks()):
        HookRegistry.register(hook.target, hook.handler, hook.kind)
    HookRegistry.apply_hooks()
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=dict(
            num_hidden_layers=4,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            max_position_embeddings=1024,
            dtype="bfloat16",
        ),
    )
    config.save_pretrained(model_path)
    device = torch.device("cuda", torch.cuda.current_device())
    rank = 0 if group is None else group.rank
    width = 1 if group is None else 2
    with (
        state_adapter_scope(GdnStateAdapter),
        get_context().override_server_args(
            model_path=str(model_path),
            page_size=4,
            max_mamba_cache_size=16,
            max_running_requests=2,
            mamba_radix_cache_strategy="extra_buffer",
            cuda_graph_config=server_args().cuda_graph_config,
        ),
        get_parallel().override(attn_tp_size=width, attn_tp_rank=rank, tp_group=group),
    ):
        model = ModelConfig(str(model_path), dtype="bfloat16")
        configurator = KVCacheConfigurator(
            server_args=get_server_args(),
            model_dtype=torch.bfloat16,
            page_size=4,
            device=str(device),
            gpu_id=device.index,
            ps=ParallelState.trivial(tp_rank=rank, tp_size=width, attn_tp_rank=rank, attn_tp_size=width, gpu_id=rank),
            pp_group=None,
            model=torch.nn.Module(),
            model_config=model,
            kv_cache_dtype=torch.bfloat16,
            sliding_window_size=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            is_draft_worker=False,
            post_capture_kv_active=True,
            spec_aux_config=SpecAuxHiddenStateConfig(),
            is_hybrid_swa=False,
            is_hybrid_swa_compress=False,
            use_mla_backend=False,
            layer_info=ModelLayerInfo(start_layer=0, end_layer=4, num_effective_layers=4),
            forward_stream=torch.cuda.current_stream(),
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            memory_pool_config=None,
        )
        sizes = configurator._derive_pool_sizes(
            # Retain at least one unpinned granule even at attention TP=2 so
            # the live-source shrink below can release actual physical bytes.
            config=MemoryPoolConfig(max_total_num_tokens=16_384, max_running_requests=2)
        )
        pools = configurator._init_pools(sizes=sizes, req_to_token_pool=None, token_to_kv_pool_allocator=None)
        state = GdnStateAdapter.create(
            pools.token_to_kv_pool, pools.token_to_kv_pool_allocator, pools.req_to_token_pool, configurator=configurator
        )
        try:
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=state.request_pool,
                    token_to_kv_pool_allocator=state.kv.allocator,
                    page_size=4,
                    tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                    enable_mamba_extra_buffer=True,
                )
            )
            state.bind_cache(cache)
            state.quota.set_limit(state.profile.reservation_bytes)
            assert state.prepare_allocation((1, 4))
            request = Req("cold", "", array("q", [1, 2, 3, 4]), SamplingParams(max_new_tokens=2))
            with state.allocation() as attempt:
                attempt.watch((request,))
                state.commit_prefill(request)
                rows = state.request_pool.alloc([request])
                assert rows is not None
                attempt.record_rows(rows)
                assert len(state.slots.owned) == 4
                attempt.rollback()
                attempt.rollback()
            assert not state.slots.owned and not state.leases
            assert request.kv.mamba_pool_idx is None and request.kv.req_pool_idx is None
            assert state.request_pool.available_size() == 2
            assert state.profile.floor_bytes <= state.quota.charged_bytes <= state.quota.limit_bytes
            state.slots.free_slots = state.slots.free_slots.roll(rank * 3)
            exercise_cache_lifecycle(state, cache, group=group)
        finally:
            torch.cuda.synchronize(device)
            state.close()
        assert state.quota.charged_bytes == 0


def exercise_cache_lifecycle(
    state: GdnStateAdapter, cache: UnifiedRadixCache, *, group: GroupCoordinator | None = None
) -> None:
    pool = state.request_pool
    device = pool.mamba_allocator.device
    tokens = array("q", [(1 << 40) + i for i in range(512)])
    request = Req("checkpoint-owner", "", tokens, SamplingParams(max_new_tokens=16))
    request.full_untruncated_fill_ids = tokens
    request.set_extend_range(0, 512)
    assert state.prepare_allocation((128, 4))
    with state.allocation() as attempt:
        attempt.watch((request,))
        state.commit_prefill(request)
        rows = pool.alloc([request])
        assert rows is not None
        attempt.record_rows(rows)
        locations = state.kv.allocator.alloc(512)
        assert locations is not None
        pool.req_to_token[rows[0], :512] = locations
        request.kv.kv_allocated_len = request.kv.kv_committed_len = 512
        attempt.commit()
    tracking = request.kv.mamba_ping_pong_track_buffer
    assert tracking is not None
    request.kv.mamba_last_track_idx = 0
    request.kv.mamba_last_track_seqlen = 512
    checkpoint = tracking[0].view(1).clone()
    for tensor in state.backing.tensors:
        tensor[:, checkpoint] = 7
    owner = cache.components[ComponentType.MAMBA]
    assert isinstance(owner, MambaComponent)
    insert = InsertParams(key=RadixKey(tokens), value=locations, prev_prefix_len=0)
    assert owner.prepare_for_caching_req(request, insert, 512, False) == 512
    assert request not in state.leases
    result = cache.insert(insert)
    owner.cleanup_after_caching_req(request, False, result, insert)
    assert not result.mamba_exist and cache.mamba_evictable_size() == 1
    assert not torch.equal(tracking[0].view(1), checkpoint)

    # A duplicate donation consumes its replacement and returns the unused
    # checkpoint exactly once, while the original cached state stays intact.
    assert state.prepare_allocation((0, 1))
    state.commit_prefill(request)
    request.kv.mamba_last_track_seqlen = 512
    duplicate = InsertParams(key=RadixKey(tokens), value=locations, prev_prefix_len=512)
    owner.prepare_for_caching_req(request, duplicate, 512, False)
    duplicate_result = cache.insert(duplicate)
    owner.cleanup_after_caching_req(request, False, duplicate_result, duplicate)
    assert duplicate_result.mamba_exist
    assert len(state.slots.owned) == 4 and cache.mamba_evictable_size() == 1

    # A continuation's prior active/tracking owners survive a rejected
    # transition, including in-place cursor/tensor mutations before publication.
    prior_owners = dict(state.slots.owned)
    prior_tracking = tracking.clone()
    prior_cursor = request.kv.mamba_next_track_idx
    assert prior_cursor is not None
    with state.allocation() as attempt:
        attempt.watch((request,))
        assert state.prepare_allocation((0, 1))
        state.commit_prefill(request)
        pool.set_mamba_ping_pong_slot(request, 0, state.leases[request][0])
        request.kv.mamba_next_track_idx = 1 - prior_cursor
        request.kv.mamba_last_track_seqlen = 999
        attempt.rollback()
    assert state.slots.owned == prior_owners and not state.leases
    restored_tracking = request.kv.mamba_ping_pong_track_buffer
    assert restored_tracking is not None and torch.equal(restored_tracking, prior_tracking)
    assert torch.equal(pool.req_index_to_mamba_ping_pong_track_buffer_mapping[request.kv.req_pool_idx], prior_tracking)
    assert request.kv.mamba_next_track_idx == prior_cursor and request.kv.mamba_last_track_seqlen is None

    fork = Req("private-copy", "", tokens + array("q", [9]), SamplingParams(max_new_tokens=16))
    before = dict(state.slots.owned)
    fork.init_next_round_input(cache, cow_mamba=True)
    assert len(fork.prefix_indices) == 512 and not fork.kv.holds_mamba
    assert state.slots.owned == before
    with state.allocation() as attempt:
        attempt.watch((fork,))
        lock = cache.inc_lock_ref(fork.last_node)
        attempt.record_lock(lock.to_dec_params())
        assert state.prepare_allocation((0, 4))
        state.commit_prefill(fork)
        rows = pool.alloc([fork])
        assert rows is not None
        attempt.record_rows(rows)
        assert fork.kv.mamba_cow_src_index is not None and fork.kv.mamba_pool_idx is not None
        pool.copy_mamba_state(fork.kv.mamba_cow_src_index.view(1), fork.kv.mamba_pool_idx.view(1))
        for tensor in state.backing.tensors:
            assert torch.all(tensor[:, fork.kv.mamba_pool_idx] == 7)
            tensor[:, fork.kv.mamba_pool_idx] = 19
            assert torch.all(tensor[:, checkpoint] == 7)
        # Shrinking confirmed empty backing preserves both the locked source
        # and private continuation numerically, even with asymmetric local IDs.
        prior_limit = state.quota.limit_bytes
        owned = dict(state.slots.owned)
        kv_backing = state.kv.components[0].allocator.backing
        assert kv_backing.try_map((kv_backing.layout.profile.bundle_capacity - 1,))
        retirement = state.prepare_reclaim(state.quota.charged_bytes)
        assert retirement is not None
        state.apply_reclaim(retirement)
        torch.cuda.synchronize(device)
        assert state.finish_retirement(retirement) > 0 and retirement.completed
        state.quota.set_limit(state.quota.charged_bytes)
        assert state.slots.owned == owned
        for tensor in state.backing.tensors:
            assert torch.all(tensor[:, fork.kv.mamba_pool_idx] == 19)
            assert torch.all(tensor[:, checkpoint] == 7)
        state.quota.set_limit(prior_limit)
        if group is None:
            attempt.rollback()
        else:
            reconciler = CapacityReconciler(
                Mock(spec=InstanceControlChannel),
                0,
                state,
                Mock(spec=InstanceRankRuntime),
                LatencySloConfig(ttft_ms=1000, tbt_ms=50),
            )
            batch = ScheduleBatch(
                reqs=[fork],
                extend_num_tokens=1,
                seq_lens_cpu=torch.tensor([513]),
                prefix_lens=[512],
            )

            def allocate() -> torch.Tensor:
                locations = state.kv.allocator.alloc(4)
                assert locations is not None
                if group.rank == 1:
                    raise SglangCapacityMiss("injected post-COW, post-KV miss")
                return locations

            with capacity_reconciler_scope(reconciler), pytest.raises(SglangCapacityMiss):
                complete_allocation(allocate, batch, token_per_req=None)
            assert attempt.rolled_back and not attempt.committed
        attempt.rollback()
    assert state.slots.owned == before and fork.kv.mamba_pool_idx is None

    request.kv.cache_protected_len = 512
    cache.cache_finished_req(request, is_insert=False, kv_len_to_handle=512)
    pool.free(request)
    assert len(state.slots.owned) == 1 and not state.leases

    # An internal checkpoint is evicted as a state-only effect. Full leaf
    # eviction later cascades through its own checkpoint on every rank.
    assert state.prepare_allocation((1, 1))
    tail = state.kv.allocator.alloc(4)
    child_state = state.slots.alloc(1)
    assert tail is not None and child_state is not None
    cache.insert(
        InsertParams(
            key=RadixKey(tokens + array("q", [1, 2, 3, 4])),
            value=torch.cat((locations, tail)),
            prev_prefix_len=512,
            mamba_value=child_state,
        )
    )
    candidate = state.select_cache_eviction((False, True), release_backing=False)
    assert candidate is not None and candidate.scope == "resource" and candidate.pool_index == 1
    assert candidate.key.tokens == tuple(tokens) and len(candidate.effects) == 1
    if group is not None:
        candidate = cast(StateCacheProposal, group.broadcast_object(candidate if group.rank == 0 else None))
    plan = state.prepare_reclaim(0, cache_eviction=candidate)
    assert plan is not None
    state.apply_reclaim(plan)
    assert cache.full_evictable_size() == 516 and cache.mamba_evictable_size() == 1
    leaf = state.select_cache_eviction((False, True), release_backing=False)
    assert leaf is not None and leaf.scope == "endpoint" and len(leaf.effects) == 2
    if group is not None:
        leaf = cast(StateCacheProposal, group.broadcast_object(leaf if group.rank == 0 else None))
    plan = state.prepare_reclaim(0, cache_eviction=leaf)
    assert plan is not None
    state.apply_reclaim(plan)
    assert cache.full_evictable_size() == 512 and cache.mamba_evictable_size() == 0
    assert not state.slots.owned
    backing = state.kv.components[0].allocator.backing
    assert backing.try_map((backing.layout.profile.bundle_capacity - 1,))
    retirement = state.prepare_reclaim(state.quota.charged_bytes - state.profile.floor_bytes)
    if retirement is None:
        assert state.quota.charged_bytes == state.profile.floor_bytes
    else:
        state.apply_reclaim(retirement)
        torch.cuda.synchronize(device)
        state.finish_retirement(retirement)
        assert retirement.completed and state.quota.charged_bytes >= state.profile.floor_bytes
