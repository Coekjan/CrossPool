"""SGLang plugin entry point for xpool."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import Concatenate

import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs

from xpool import bootstrap, devkit
from xpool.config import init_global_config
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.integrations.sglang.adapter import (
    SglangInstanceRankBinding,
    SglangInstanceRankRuntime,
    SglangShimAdapter,
    model_runner_architectures,
)
from xpool.integrations.sglang.registry import MODELS_PACKAGE, discover_sglang_model_adapters
from xpool.integrations.sglang.server_args import validate_sglang_server_args
from xpool.integrations.sglang.shim import iter_ffn_shims
from xpool.native import RuntimeRole
from xpool.runtime.instance import InstanceRankRuntime
from xpool.runtime.transport import InstanceRankTransportProfile

MODEL_RUNNER_LOAD_MODEL = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
MODEL_RUNNER_INITIALIZE = "sglang.srt.model_executor.model_runner.ModelRunner.initialize"
MODEL_RUNNER_INIT_MEMORY_POOL = "sglang.srt.model_executor.model_runner.ModelRunner.init_memory_pool"
XPOOL_REQUIRED_HOOK_TARGETS: set[str] = set()
SGLANG_DEVKIT_PACKAGE = "xpool.integrations.sglang.devkit"
logger = logging.getLogger(__name__)


def install() -> None:
    """Install xpool hooks in an SGLang process.

    Side Effects:
        Imports and instantiates all repo-owned SGLang model adapters, registers
        their hooks in SGLang's global ``HookRegistry``, and wraps
        ``ModelRunner.load_model`` for adapter validation, and wraps
        ``ModelRunner.init_memory_pool`` to start transport after SGLang
        resolves request concurrency.

    Raises:
        SystemExit: If config initialization, adapter discovery, or hook
            registration fails. SGLang catches ordinary plugin exceptions, so
            xpool uses ``SystemExit`` for fatal fail-closed startup behavior.
    """

    try:
        init_global_config()
        adapters = discover_sglang_model_adapters(MODELS_PACKAGE)
        required_targets: set[str] = set()
        for adapter in adapters:
            for hook in adapter.hooks():
                HookRegistry.register(hook.target, hook.handler, hook.kind)
                required_targets.add(hook.target)
        HookRegistry.register(
            MODEL_RUNNER_LOAD_MODEL,
            partial(around_model_runner_load_model, adapters),
            HookType.AROUND,
        )
        HookRegistry.register(
            MODEL_RUNNER_INIT_MEMORY_POOL,
            after_model_runner_init_memory_pool,
            HookType.AFTER,
        )
        HookRegistry.register(
            MODEL_RUNNER_INITIALIZE,
            after_model_runner_initialize,
            HookType.AFTER,
        )
        required_targets.update(
            (
                MODEL_RUNNER_LOAD_MODEL,
                MODEL_RUNNER_INIT_MEMORY_POOL,
                MODEL_RUNNER_INITIALIZE,
            )
        )
        XPOOL_REQUIRED_HOOK_TARGETS.update(required_targets)
        install_apply_hooks_guard()
    except Exception as exc:
        raise SystemExit(f"xpool SGLang plugin failed to install: {exc}") from exc


def install_apply_hooks_guard() -> None:
    """Install a fatal postcondition check around SGLang hook application.

    SGLang catches ordinary exceptions while applying plugin hooks and then
    continues startup. XPool cannot serve correctly if any required hook is
    missing, so the guard raises ``SystemExit`` after SGLang's best-effort hook
    application leaves one of xpool's targets unpatched.

    Side Effects:
        Replaces ``HookRegistry.apply_hooks`` with a classmethod wrapper once.
    """

    original_apply_hooks = getattr(HookRegistry, "apply_hooks", None)
    if original_apply_hooks is None or getattr(HookRegistry, "xpool_apply_hooks_guarded", False):
        return

    def guarded_apply_hooks(registry: type[object]) -> object:
        result = original_apply_hooks()
        verify_required_hooks_applied()
        return result

    setattr(HookRegistry, "apply_hooks", classmethod(guarded_apply_hooks))
    setattr(HookRegistry, "xpool_apply_hooks_guarded", True)


def verify_required_hooks_applied() -> None:
    """Raise fatally if SGLang skipped any xpool-required hook target.

    Raises:
        SystemExit: If SGLang's private patched-target registry is missing or
            any target registered by the xpool plugin was not patched.

    Side Effects:
        None.
    """

    patched_targets = getattr(HookRegistry, "_patched", None)
    if not isinstance(patched_targets, set):
        raise SystemExit("xpool SGLang plugin cannot verify HookRegistry patched targets")
    missing_targets = XPOOL_REQUIRED_HOOK_TARGETS.difference(str(target) for target in patched_targets)
    if missing_targets:
        joined = ", ".join(sorted(missing_targets))
        raise SystemExit(f"xpool SGLang plugin failed to apply required hooks: {joined}")


def around_model_runner_load_model[**P, R](
    adapters: Sequence[SglangShimAdapter],
    original_fn: Callable[Concatenate[ModelRunner, P], R],
    model_runner: ModelRunner,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run adapter lifecycle checks around SGLang model loading.

    Args:
        adapters: Model adapters installed by the xpool SGLang plugin.
        original_fn: Original SGLang ``ModelRunner.load_model`` callable.
        model_runner: SGLang model runner being loaded.
        *args: Positional arguments forwarded to the original load function.
        **kwargs: Keyword arguments forwarded to the original load function.

    Returns:
        Return value from the original SGLang load function.

    Raises:
        ConfigError: If the process-global xpool config is unavailable or model
            binding policy derivation fails.
        OSError: If the matched model ``config.json`` cannot be opened.
        RuntimeError: If no adapter matches a configured model, server arguments
            are unsupported, SGLang TP/DP settings do not match xpool config,
            binding fails, or post-load validation fails.

    Side Effects:
        Binds xpool instance/model identity, invokes model loading, stamps
        every FFN shim with identity, and runs adapter postconditions.
    """

    server_args = model_runner.server_args
    validate_sglang_server_args(server_args)

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
        server_args,
        supports_dp_attention=adapter.supports_dp_attention,
    )
    binding.validate_server_args(server_args)
    bootstrap.init(int(binding.cuda_device), RuntimeRole.INSTANCE)
    devkit.install()
    devkit.install(SGLANG_DEVKIT_PACKAGE)
    adapter.validate_before_load(model_runner)

    runtime = SglangInstanceRankRuntime.attach(model_runner, binding)
    try:
        adapter.bind_runtime(model_runner)

        result = original_fn(model_runner, *args, **kwargs)

        binding.bind_shim_runtime(model_runner)
        adapter.validate_after_load(model_runner)
    except Exception:
        runtime.detach(model_runner)
        raise
    return result


def after_model_runner_init_memory_pool[R](
    result: R,
    model_runner: ModelRunner,
    pre_model_load_memory: int,
) -> R:
    """Start xpool transport after SGLang resolves memory-pool concurrency.

    Args:
        result: Return value from SGLang's original ``init_memory_pool`` call.
        model_runner: Loaded runner with an applied memory-pool configuration.
        pre_model_load_memory: SGLang memory sample forwarded to the original
            method; already consumed before this hook runs.

    Returns:
        The original ``init_memory_pool`` return value unchanged.

    Raises:
        RuntimeError: If the load hook did not attach a binding, transport
            geometry cannot be derived, or instance runtime startup fails.

    Side Effects:
        Initializes and attaches the process-global xpool instance transport
        runtime.
    """

    runtime = SglangInstanceRankRuntime.require(model_runner)
    binding = runtime.binding
    try:
        server_args = model_runner.server_args
        ffn_profile = derive_instance_ffn_profile(model_runner, binding, server_args)
        transport = derive_instance_rank_transport_profile(binding, ffn_profile)
        runtime.instance_rank = InstanceRankRuntime.start(
            instance_id=binding.instance_id,
            rank=binding.worker_rank,
            transport=transport,
            ffn_profile=ffn_profile,
        )
        runtime.instance_rank.wait_for_fabric_executable()
        runtime.instance_rank.attach_arena_from_daemon()
        runtime.instance_rank.start_failure_monitor()
    except Exception:
        runtime.detach(model_runner)
        raise
    return result


def after_model_runner_initialize[R](
    result: R,
    model_runner: ModelRunner,
    pre_model_load_memory: float,
) -> R:
    """Publish SGLang's post-initialize barrier for production execution.

    Args:
        result: Return value from SGLang's original ``initialize`` method.
        model_runner: Runner whose model and CUDA graphs are fully initialized.
        pre_model_load_memory: Memory sample already consumed by SGLang.

    Returns:
        The original ``initialize`` return value unchanged.

    Raises:
        RuntimeError: If production startup did not retain the executable plan.

    Side Effects:
        Publishes this rank's initialized barrier to the daemon and waits for
        generation-wide readiness.
    """

    runtime = SglangInstanceRankRuntime.require(model_runner)
    if runtime.instance_rank is None:
        raise RuntimeError("xpool ModelRunner.initialize hook requires a started Instance-rank runtime")
    runtime.instance_rank.publish_initialized()
    runtime.instance_rank.wait_for_ready()
    return result


def derive_instance_ffn_profile(
    model_runner: ModelRunner,
    binding: SglangInstanceRankBinding,
    server_args: ServerArgs,
) -> InstanceFfnProfile:
    """Derive the complete FFN executor ffn_profile from resolved SGLang state.

    Args:
        model_runner: Loaded runner with memory-pool concurrency and installed
            FFN shims.
        binding: Validated xpool instance and parallel identity.
        server_args: Resolved SGLang graph and eager ffn_profile settings.

    Returns:
        Strict rank-independent ffn_profile registered with the daemon.

    Raises:
        RuntimeError: If dtype, layers, bucket ceilings, or model config bytes
            cannot be resolved without guessing.
        OSError: If the model's config file cannot be read.
    """

    dtype = getattr(model_runner.model_config, "dtype", None)
    if dtype not in {torch.bfloat16, torch.float16}:
        raise RuntimeError(f"xpool FFN ffn_profile does not support SGLang dtype {dtype!r}")
    payload_dtype = dtype

    model = getattr(model_runner, "model", None)
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
    max_prefill_tokens = getattr(server_args, "max_prefill_tokens", None)
    if not isinstance(max_running_requests, int) or isinstance(max_running_requests, bool) or max_running_requests <= 0:
        raise RuntimeError("xpool cannot derive positive eager decode rows from ModelRunner.max_running_requests")
    if not isinstance(max_prefill_tokens, int) or isinstance(max_prefill_tokens, bool) or max_prefill_tokens <= 0:
        raise RuntimeError("xpool cannot derive positive eager prefill rows from ServerArgs.max_prefill_tokens")
    max_decode_rows = resolved_graph_capacity(
        eager_capacity=max_running_requests,
        enabled=not server_args.disable_cuda_graph,
        buckets=getattr(server_args, "cuda_graph_bs", None),
        maximum=getattr(server_args, "cuda_graph_max_bs", None),
        label="decode CUDA graph",
    )
    max_prefill_rows = resolved_graph_capacity(
        eager_capacity=max_prefill_tokens,
        enabled=not server_args.disable_piecewise_cuda_graph,
        buckets=getattr(server_args, "piecewise_cuda_graph_tokens", None),
        maximum=getattr(server_args, "piecewise_cuda_graph_max_tokens", None),
        label="piecewise prefill CUDA graph",
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
    enabled: bool,
    buckets: object,
    maximum: object,
    label: str,
) -> int:
    """Resolve one ffn_profile capacity from eager and enabled graph geometry.

    Disabled graph paths contribute no capacity even when SGLang retains stale
    bucket fields. Enabled paths accept the concrete bucket list and resolved
    maximum exposed by the pinned ``ServerArgs`` object.
    """

    if not enabled:
        return eager_capacity
    candidates = [eager_capacity]
    if buckets is not None:
        if not isinstance(buckets, (list, tuple)):
            raise RuntimeError(f"xpool cannot derive positive {label} buckets from resolved ServerArgs")
        for row in buckets:
            if not isinstance(row, int) or isinstance(row, bool) or row <= 0:
                raise RuntimeError(f"xpool cannot derive positive {label} buckets from resolved ServerArgs")
            candidates.append(row)
    if maximum is not None:
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise RuntimeError(f"xpool cannot derive a positive {label} maximum from resolved ServerArgs")
        candidates.append(maximum)
    return max(candidates)


def derive_instance_rank_transport_profile(
    binding: SglangInstanceRankBinding,
    ffn_profile: InstanceFfnProfile,
) -> InstanceRankTransportProfile:
    """Derive daemon registration transport attributes for one SGLang rank.

    Args:
        binding: Validated xpool attention TP/DP rank binding.
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
