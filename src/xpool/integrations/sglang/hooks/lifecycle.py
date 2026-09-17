"""CrossPool lifecycle hooks for SGLang model runners."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Concatenate

import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import Backend, PhaseConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_exec, get_schedule, get_serving

from xpool import bootstrap, devkit
from xpool.config import get_global_config
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.integrations.sglang.adapter import (
    SglangInstanceRankBinding,
    SglangInstanceRankRuntime,
    SglangShimAdapter,
    model_runner_architectures,
)
from xpool.integrations.sglang.hooks.registry import SglangHook, SglangHookSet
from xpool.integrations.sglang.kv.allocator import (
    ElasticPagedTokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool
from xpool.integrations.sglang.registry import MODELS_PACKAGE, discover_sglang_model_adapters
from xpool.integrations.sglang.server_args import validate_sglang_server_args
from xpool.integrations.sglang.shim import iter_ffn_shims
from xpool.native import RuntimeRole
from xpool.runtime.instance import InstanceRankRuntime
from xpool.runtime.transport import InstanceRankTransportProfile
from xpool.service.wire import ServingListener

MODEL_RUNNER_LOAD_MODEL = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
MODEL_RUNNER_ALLOC_MEMORY_POOL = "sglang.srt.model_executor.model_runner.ModelRunner.alloc_memory_pool"
SCHEDULER_GET_INIT_INFO = "sglang.srt.managers.scheduler.Scheduler.get_init_info"
SCHEDULER_RELEASE_HOST_RESOURCES = "sglang.srt.managers.scheduler.Scheduler.release_host_resources"
SGLANG_DEVKIT_PACKAGE = "xpool.integrations.sglang.devkit"
logger = logging.getLogger(__name__)


class LifecycleHookSet(SglangHookSet):
    """Model-adapter and common runner lifecycle hooks."""

    def hooks(self) -> tuple[SglangHook, ...]:
        adapters = discover_sglang_model_adapters(MODELS_PACKAGE)
        hooks = [hook for adapter in adapters for hook in adapter.hooks()]
        hooks.extend(
            (
                SglangHook(
                    MODEL_RUNNER_LOAD_MODEL,
                    partial(around_model_runner_load_model, adapters),
                    HookType.AROUND,
                ),
                SglangHook(
                    MODEL_RUNNER_ALLOC_MEMORY_POOL,
                    after_model_runner_alloc_memory_pool,
                    HookType.AFTER,
                ),
                SglangHook(
                    SCHEDULER_GET_INIT_INFO,
                    after_scheduler_get_init_info,
                    HookType.AFTER,
                ),
                SglangHook(
                    SCHEDULER_RELEASE_HOST_RESOURCES,
                    around_scheduler_release_host_resources,
                    HookType.AROUND,
                ),
            )
        )
        return tuple(hooks)


def around_model_runner_load_model[**P, R](
    adapters: Sequence[SglangShimAdapter],
    original_fn: Callable[Concatenate[ModelRunner, P], R],
    model_runner: ModelRunner,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run adapter lifecycle checks around SGLang model loading.

    Args:
        adapters: Model adapters installed by the CrossPool SGLang plugin.
        original_fn: Original SGLang ``ModelRunner.load_model`` callable.
        model_runner: SGLang model runner being loaded.
        *args: Positional arguments forwarded to the original load function.
        **kwargs: Keyword arguments forwarded to the original load function.

    Returns:
        Return value from the original SGLang load function.

    Raises:
        ConfigError: If the process-global CrossPool config is unavailable or model
            binding policy derivation fails.
        OSError: If the matched model ``config.json`` cannot be opened.
        RuntimeError: If no adapter matches a configured model, server arguments
            are unsupported, SGLang TP/DP settings do not match CrossPool config,
            binding fails, or post-load validation fails.

    Side Effects:
        Binds CrossPool instance/model identity, invokes model loading, stamps
        every FFN shim with identity, and runs adapter postconditions.
    """

    validate_sglang_server_args()

    matching_adapters = tuple(adapter for adapter in adapters if adapter.matches(model_runner))
    if not matching_adapters:
        architectures = ", ".join(sorted(model_runner_architectures(model_runner))) or "<unknown>"
        raise RuntimeError(
            f"no xpool adapter matches SGLang model architecture ({architectures}); add a model adapter under "
            "xpool.integrations.sglang.models or remove the [[models]] entry from XPOOL_CONFIG."
        )
    if len(matching_adapters) != 1:
        names = ", ".join(adapter.name for adapter in matching_adapters)
        raise RuntimeError(f"xpool requires exactly one model adapter match, got {len(matching_adapters)}: {names}")
    adapter = matching_adapters[0]
    binding = SglangInstanceRankBinding.resolve(
        model_runner,
        supports_dp_attention=adapter.supports_dp_attention,
    )
    binding.validate_server_args()
    bootstrap.init(int(binding.cuda_device), RuntimeRole.INSTANCE)
    devkit.install()
    devkit.install(SGLANG_DEVKIT_PACKAGE)
    adapter.validate_before_load(model_runner)

    runtime = SglangInstanceRankRuntime.attach(model_runner, binding)
    logger.info(
        "bound instance=%s rank=%s device=%s pid=%s",
        binding.instance_id,
        binding.worker_rank,
        binding.cuda_device,
        os.getpid(),
    )
    try:
        adapter.bind_runtime(model_runner)

        result = original_fn(model_runner, *args, **kwargs)

        binding.bind_shim_runtime(model_runner)
        adapter.validate_after_load(model_runner)
    except Exception:
        runtime.detach(model_runner)
        raise
    return result


def after_model_runner_alloc_memory_pool[R](
    result: R,
    model_runner: ModelRunner,
    memory_pool_config: MemoryPoolConfig | None = None,
) -> R:
    """Start CrossPool transport after SGLang resolves memory-pool concurrency.

    Args:
        result: Return value from SGLang's original ``alloc_memory_pool`` call.
        model_runner: Loaded runner with an applied memory-pool configuration.
        memory_pool_config: Optional SGLang memory-pool configuration forwarded
            to the original method; already consumed before this hook runs.

    Returns:
        The original ``alloc_memory_pool`` return value unchanged.

    Raises:
        RuntimeError: If the load hook did not attach a binding, transport
            geometry cannot be derived, or instance runtime startup fails.

    Side Effects:
        Initializes and attaches the process-global CrossPool instance transport
        runtime.
    """

    runtime = SglangInstanceRankRuntime.require(model_runner)
    binding = runtime.binding
    try:
        ffn_profile = derive_instance_ffn_profile(model_runner, binding)
        transport = derive_instance_rank_transport_profile(binding, ffn_profile)
        pool = model_runner.token_to_kv_pool
        if not isinstance(pool, ElasticMHATokenToKVPool | ElasticMLATokenToKVPool):
            raise RuntimeError("xpool elastic kv pool replacement was not applied")
        allocator = model_runner.token_to_kv_pool_allocator
        if not isinstance(allocator, ElasticTokenToKVPoolAllocator | ElasticPagedTokenToKVPoolAllocator):
            raise RuntimeError("xpool elastic kv allocator replacement was not applied")
        request_pool = model_runner.req_to_token_pool
        if not isinstance(request_pool, ReqToTokenPool):
            raise RuntimeError("xpool SGLang request pool is unavailable after memory-pool allocation")
        runtime.instance_rank = InstanceRankRuntime.start(
            instance_id=binding.instance_id,
            rank=binding.worker_rank,
            transport=transport,
            ffn_profile=ffn_profile,
            kv_capacity=pool.backing.partition_profile(),
        )
        plan = runtime.instance_rank.wait_for_fabric_executable()
        channel_ref = runtime.instance_rank.client.kv_control_channel(plan.generation)
        if channel_ref.generation != plan.generation:
            raise RuntimeError("xpool daemon returned a kv control channel for a different fabric generation")
        config = get_global_config()
        model = config.models[binding.instance_index]
        runtime.kv_capacity = CapacityReconciler.attach(
            channel_name=channel_ref.name,
            group_index=binding.instance_index * config.atn_world_size + binding.atn_dp_rank,
            partition_index=binding.instance_index * config.atn_world_size + binding.worker_rank,
            dp_group_count=binding.atn_dp_size,
            backing=pool.backing,
            allocator=allocator,
            request_pool=request_pool,
            instance_rank=runtime.instance_rank,
            slo=model.slo or config.scheduler.slo,
        )
        runtime.instance_rank.attach_arena_from_daemon()
        logger.info(
            "transport attached instance=%s rank=%s device=%s pid=%s",
            binding.instance_id,
            binding.worker_rank,
            binding.cuda_device,
            os.getpid(),
        )
        runtime.instance_rank.start_failure_monitor()
    except Exception:
        runtime.detach(model_runner)
        raise
    return result


def after_scheduler_get_init_info[R](
    result: R,
    scheduler: Scheduler,
) -> R:
    """Publish CrossPool readiness during SGLang's scheduler startup handshake.

    Args:
        result: Return value from SGLang's original ``get_init_info`` method.
        scheduler: Fully constructed Scheduler publishing its initialization information.

    Returns:
        The original ``get_init_info`` return value unchanged.

    Raises:
        RuntimeError: If startup did not retain the executable plan.

    Side Effects:
        Publishes this rank's initialized barrier to the daemon and waits for
        System Ready before SGLang sends its scheduler handshake to the parent.
    """

    runtime = SglangInstanceRankRuntime.require(scheduler.tp_worker.model_runner)
    if runtime.instance_rank is None:
        raise RuntimeError("xpool Scheduler.get_init_info hook requires a started Instance-rank runtime")
    runtime.instance_rank.publish_initialized(
        ServingListener(
            host=get_serving().host,
            port=get_serving().port,
        )
    )
    runtime.instance_rank.wait_for_ready()
    binding = runtime.binding
    logger.info(
        "ready instance=%s rank=%s device=%s pid=%s generation=%s",
        binding.instance_id,
        binding.worker_rank,
        binding.cuda_device,
        os.getpid(),
        runtime.instance_rank.fabric_plan.generation.format()
        if runtime.instance_rank.fabric_plan is not None
        else "unknown",
    )
    return result


def around_scheduler_release_host_resources[R](
    original_fn: Callable[[Scheduler], R],
    scheduler: Scheduler,
) -> R:
    """Release runner-owned CrossPool resources after graceful SGLang teardown."""

    model_runner = scheduler.tp_worker.model_runner
    runtime = SglangInstanceRankRuntime.require(model_runner)
    try:
        return original_fn(scheduler)
    finally:
        try:
            torch.cuda.synchronize(model_runner.device)
        finally:
            runtime.detach(model_runner)


def derive_instance_ffn_profile(
    model_runner: ModelRunner,
    binding: SglangInstanceRankBinding,
) -> InstanceFfnProfile:
    """Derive the complete FFN executor ffn_profile from resolved SGLang state.

    Args:
        model_runner: Loaded runner with memory-pool concurrency and installed
            FFN shims.
        binding: Validated CrossPool instance and parallel identity.

    Returns:
        Strict rank-independent ffn_profile registered with the daemon.

    Raises:
        RuntimeError: If dtype, layers, bucket ceilings, or model config bytes
            cannot be resolved without guessing.
        OSError: If the model's config file cannot be read.
    """

    dtype = model_runner.model_config.dtype
    if dtype not in {torch.bfloat16, torch.float16}:
        raise RuntimeError(f"xpool FFN ffn_profile does not support SGLang dtype {dtype!r}")
    payload_dtype = dtype

    model = model_runner.model
    if model is None:
        raise RuntimeError("xpool cannot derive an FFN ffn_profile before SGLang installs the model")
    shims = tuple(sorted(iter_ffn_shims(model), key=lambda shim: shim.layer_id))
    if not shims:
        raise RuntimeError("xpool cannot derive an FFN ffn_profile without installed FFN shims")
    hidden_sizes = {shim.hidden_size for shim in shims}
    if len(hidden_sizes) != 1:
        raise RuntimeError(f"xpool FFN shims disagree on hidden size: {sorted(hidden_sizes)}")
    layers = tuple(InstanceFfnLayerProfile(layer_id=shim.layer_id, kind=shim.layer_kind) for shim in shims)

    max_running_requests = model_runner.max_running_requests
    max_prefill_tokens = get_schedule().max_prefill_tokens
    if not isinstance(max_running_requests, int) or isinstance(max_running_requests, bool) or max_running_requests <= 0:
        raise RuntimeError("xpool cannot derive positive eager decode rows from ModelRunner.max_running_requests")
    if not isinstance(max_prefill_tokens, int) or isinstance(max_prefill_tokens, bool) or max_prefill_tokens <= 0:
        raise RuntimeError(
            "xpool cannot derive positive eager prefill rows from "
            "resolved SGLang runtime configuration max_prefill_tokens"
        )
    cuda_graph_config = get_exec().graph.cuda_graph_config
    if cuda_graph_config is None:
        raise RuntimeError("xpool cannot derive FFN row capacities before SGLang resolves cuda_graph_config")
    max_decode_rows = resolved_graph_capacity(
        eager_capacity=max_running_requests,
        phase_config=cuda_graph_config.decode,
        label="decode CUDA graph",
    )
    max_prefill_rows = resolved_graph_capacity(
        eager_capacity=max_prefill_tokens,
        phase_config=cuda_graph_config.prefill,
        label="prefill CUDA graph",
    )

    model_config_path = binding.model_path / "config.json"
    return InstanceFfnProfile(
        model_config_digest=hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
        payload_dtype=payload_dtype,
        hidden_size=hidden_sizes.pop(),
        layers=layers,
        decode_payload_row_capacity=max_decode_rows,
        prefill_payload_row_capacity=max_prefill_rows,
        group_sum_complete_admitted=binding.atn_dp_size > 1,
    )


def resolved_graph_capacity(
    *,
    eager_capacity: int,
    phase_config: PhaseConfig,
    label: str,
) -> int:
    """Resolve one ffn_profile capacity from eager and enabled graph geometry.

    Disabled graph paths contribute no graph capacity. Enabled paths accept the
    concrete bucket list and resolved maximum exposed by SGLang's phase config.
    """

    if phase_config.backend == Backend.DISABLED:
        return eager_capacity
    candidates = [eager_capacity]
    if phase_config.bs is not None:
        if not isinstance(phase_config.bs, (list, tuple)):
            raise RuntimeError(f"xpool cannot derive positive {label} buckets from resolved SGLang graph configuration")
        for row in phase_config.bs:
            if not isinstance(row, int) or isinstance(row, bool) or row <= 0:
                raise RuntimeError(
                    f"xpool cannot derive positive {label} buckets from resolved SGLang graph configuration"
                )
            candidates.append(row)
    if phase_config.max_bs is not None:
        if (
            not isinstance(phase_config.max_bs, int)
            or isinstance(phase_config.max_bs, bool)
            or phase_config.max_bs <= 0
        ):
            raise RuntimeError(
                f"xpool cannot derive a positive {label} maximum from resolved SGLang graph configuration"
            )
        candidates.append(phase_config.max_bs)
    return max(candidates)


def derive_instance_rank_transport_profile(
    binding: SglangInstanceRankBinding,
    ffn_profile: InstanceFfnProfile,
) -> InstanceRankTransportProfile:
    """Derive daemon registration transport attributes for one SGLang rank.

    Args:
        binding: Validated CrossPool attention TP/DP rank binding.
        ffn_profile: Rank-independent hidden geometry and row coverage.

    Returns:
        Transport geometry covering every eager and captured ffn_profile shape.
    """

    return InstanceRankTransportProfile(
        hidden_size=ffn_profile.hidden_size,
        payload_row_capacity=max(
            ffn_profile.decode_payload_row_capacity,
            ffn_profile.prefill_payload_row_capacity,
        ),
        atn_tp_rank=binding.atn_tp_rank,
        atn_tp_size=binding.atn_tp_size,
        atn_dp_rank=binding.atn_dp_rank,
        atn_dp_size=binding.atn_dp_size,
    )
