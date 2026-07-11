"""SGLang plugin entry point for xpool."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import Concatenate, cast

import torch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs

from xpool import bootstrap, devkit
from xpool.abi import RuntimeRole
from xpool.config import get_global_config, init_global_config
from xpool.integrations.sglang.adapter import (
    SglangModelAdapter,
    XpoolModelBinding,
    attach_model_binding,
    clear_model_binding,
    inject_shim_identity,
    model_runner_architectures,
    resolve_model_binding,
)
from xpool.integrations.sglang.registry import MODELS_PACKAGE, discover_sglang_model_adapters
from xpool.integrations.sglang.server_args import validate_sglang_server_args
from xpool.integrations.sglang.shim import iter_ffn_shims
from xpool.runtime.instance import init_instance
from xpool.runtime.transport import InstanceTransportAttributes

MODEL_RUNNER_LOAD_MODEL = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
MODEL_RUNNER_INIT_MEMORY_POOL = "sglang.srt.model_executor.model_runner.ModelRunner.init_memory_pool"
XPOOL_REQUIRED_HOOK_TARGETS: set[str] = set()
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
        required_targets.update((MODEL_RUNNER_LOAD_MODEL, MODEL_RUNNER_INIT_MEMORY_POOL))
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
    adapters: Sequence[SglangModelAdapter],
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
        Binds xpool instance/model identity, invokes SGLang model loading, stamps
        every FFN shim with identity, and runs adapter postconditions.
    """

    server_args = model_runner_server_args(model_runner)
    validate_sglang_server_args(server_args)

    matching_adapters = tuple(adapter for adapter in adapters if adapter.matches(model_runner))
    binding = resolve_model_binding(model_runner)
    if not matching_adapters:
        architectures = ", ".join(sorted(model_runner_architectures(model_runner))) or "<unknown>"
        raise RuntimeError(
            f"xpool config owns SGLang model path {binding.model_path} but no xpool adapter "
            f"matches its architecture ({architectures}); add a model adapter under "
            "xpool.integrations.sglang.models or remove the [[models]] entry from XPOOL_CONFIG."
        )
    validate_sglang_parallel_args(server_args, binding)
    bootstrap.init(int(binding.cuda_device), RuntimeRole.INSTANCE)
    devkit.install()
    for adapter in matching_adapters:
        adapter.validate_before_load(model_runner)

    attach_model_binding(model_runner, binding)
    try:
        for adapter in matching_adapters:
            adapter.bind_runtime(model_runner)

        result = original_fn(model_runner, *args, **kwargs)

        inject_shim_identity(model_runner, binding)
        for adapter in matching_adapters:
            adapter.validate_after_load(model_runner)
    except Exception:
        clear_model_binding(model_runner, binding)
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
        runtime unless direct shim loopback is enabled.
    """

    binding = getattr(model_runner, "xpool_model_binding", None)
    if not isinstance(binding, XpoolModelBinding):
        raise RuntimeError("xpool ModelRunner.init_memory_pool hook requires an attached model binding")
    config = get_global_config()
    if config.debug.shim_loopback.enable:
        return result
    try:
        server_args = model_runner_server_args(model_runner)
        transport = derive_transport_attributes(model_runner, binding, server_args)
        init_instance(
            instance_id=binding.instance_id,
            rank=binding.sglang_rank,
            transport=transport,
        )
    except Exception:
        clear_model_binding(model_runner, binding)
        raise
    return result


def validate_sglang_parallel_args(server_args: ServerArgs, binding: XpoolModelBinding) -> None:
    """Validate SGLang TP/DP launch dimensions against xpool config.

    Args:
        server_args: Resolved SGLang server arguments from the active model runner.
        binding: xpool model binding derived from ``XPOOL_CONFIG``.

    Raises:
        RuntimeError: If SGLang ``tp_size`` or ``dp_size`` does not match the
    dimensions and rank placement xpool derives for the bound instance.
    """

    for label, actual, expected in (
        ("tp_size", server_args.tp_size, binding.sglang_tp_size),
        ("dp_size", server_args.dp_size, binding.sglang_dp_size),
        ("base_gpu_id", server_args.base_gpu_id, binding.sglang_base_gpu_id),
        ("gpu_id_step", server_args.gpu_id_step, binding.sglang_gpu_id_step),
        ("enable_dp_attention", server_args.enable_dp_attention, binding.enable_dp_atn),
    ):
        if actual != expected:
            raise RuntimeError(
                f"xpool config expects SGLang {label}={expected} for {binding.instance_id}, got {actual}"
            )
    expected_cuda_device = binding.sglang_base_gpu_id + binding.sglang_rank * binding.sglang_gpu_id_step
    if binding.cuda_device != expected_cuda_device:
        raise RuntimeError(
            f"xpool config expects SGLang rank {binding.sglang_rank} to run on CUDA device {expected_cuda_device}, "
            f"got {binding.cuda_device}"
        )


def model_runner_server_args(model_runner: ModelRunner) -> ServerArgs:
    """Return resolved SGLang server arguments from a model runner.

    Args:
        model_runner: SGLang model runner expected to expose ``server_args``.

    Returns:
        Resolved ``ServerArgs`` object for compatibility validation.

    Raises:
        RuntimeError: If SGLang did not attach ``server_args`` to the runner.
    """

    server_args = getattr(model_runner, "server_args", None)
    if server_args is None:
        raise RuntimeError("xpool SGLang plugin requires ModelRunner.server_args for compatibility validation")
    return cast(ServerArgs, server_args)


def derive_transport_attributes(
    model_runner: ModelRunner,
    binding: XpoolModelBinding,
    server_args: ServerArgs,
) -> InstanceTransportAttributes:
    """Derive daemon registration transport attributes for one SGLang rank.

    Args:
        model_runner: Loaded runner providing dtype, model shims, hidden size,
            and resolved eager request concurrency.
        binding: Validated xpool attention TP/DP rank binding.
        server_args: Resolved SGLang prefill and CUDA graph limits.

    Returns:
        Transport geometry whose token capacity covers eager decode, prefill,
        full CUDA graph, and piecewise CUDA graph execution.

    Raises:
        RuntimeError: If dtype, hidden size, eager request concurrency, or all
            positive token-capacity limits cannot be resolved.
    """

    dtype = getattr(model_runner.model_config, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError("xpool cannot derive SGLang transport element size from ModelRunner.model_config.dtype")

    model = getattr(model_runner, "model", None)
    hidden_size: int | None = None
    if model is not None:
        hidden_sizes = {shim.hidden_size for shim in iter_ffn_shims(model)}
        if len(hidden_sizes) == 1:
            hidden_size = hidden_sizes.pop()
        elif hidden_sizes:
            raise RuntimeError(f"xpool FFN shims disagree on hidden size: {sorted(hidden_sizes)}")
    if hidden_size is None:
        candidate_hidden_size = getattr(getattr(model_runner.model_config, "hf_config", None), "hidden_size", None)
        if isinstance(candidate_hidden_size, int) and candidate_hidden_size > 0:
            hidden_size = candidate_hidden_size
    if hidden_size is None:
        raise RuntimeError("xpool cannot derive SGLang transport hidden size from installed FFN shims or hf_config")

    max_running_requests = model_runner.max_running_requests
    if max_running_requests is None or max_running_requests <= 0:
        raise RuntimeError("xpool cannot derive a positive SGLang eager decode request capacity")
    max_token_candidates = [max_running_requests]
    for name in ("max_prefill_tokens", "cuda_graph_max_bs", "piecewise_cuda_graph_max_tokens"):
        value = getattr(server_args, name, None)
        if isinstance(value, int) and value > 0:
            max_token_candidates.append(value)
    for name in ("cuda_graph_bs", "piecewise_cuda_graph_tokens"):
        value = getattr(server_args, name, None)
        if isinstance(value, (list, tuple)):
            max_token_candidates.extend(item for item in value if isinstance(item, int) and item > 0)
    if not max_token_candidates:
        raise RuntimeError("xpool cannot derive a positive SGLang transport max token capacity")

    return InstanceTransportAttributes(
        element_size=torch.empty((), dtype=dtype).element_size(),
        hidden_size=hidden_size,
        max_tokens=max(max_token_candidates),
        atn_tp_rank=binding.atn_tp_rank,
        atn_tp_size=binding.atn_tp_size,
        atn_dp_rank=binding.atn_dp_rank,
        atn_dp_size=binding.atn_dp_size,
    )
