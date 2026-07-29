"""SGLang plugin entry point for xpool."""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import threading
from collections.abc import Callable, Sequence
from functools import partial
from types import FrameType
from typing import Concatenate

import psutil
import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import SignalHandler
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.cudacore_pyspy_dump_utils import collect_scheduler_processes

from xpool import bootstrap, devkit
from xpool.abi import TensorDType
from xpool.config import LoopbackSite, get_global_config, init_global_config
from xpool.fabric import FfnLayerSpec, FfnWorkload
from xpool.integrations.sglang.adapter import (
    SglangModelAdapter,
    XpoolModelBinding,
    XpoolModelRuntime,
    model_runner_architectures,
)
from xpool.integrations.sglang.registry import MODELS_PACKAGE, discover_sglang_model_adapters
from xpool.integrations.sglang.server_args import validate_sglang_server_args
from xpool.integrations.sglang.shim import iter_ffn_shims
from xpool.runtime import RuntimeRole
from xpool.runtime.instance import Instance
from xpool.runtime.transport import InstanceTransportAttributes
from xpool.utils.sighandler import sighandle

MODEL_RUNNER_LOAD_MODEL = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
MODEL_RUNNER_INITIALIZE = "sglang.srt.model_executor.model_runner.ModelRunner.initialize"
MODEL_RUNNER_INIT_MEMORY_POOL = "sglang.srt.model_executor.model_runner.ModelRunner.init_memory_pool"
SIGNAL_HANDLER_SIGTERM = "sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler"
SCHEDULER_RUN_EVENT_LOOP = "sglang.srt.managers.scheduler.Scheduler.run_event_loop"
KILL_PROCESS_TREE = "sglang.cli.serve.kill_process_tree"
SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS = 20.0
XPOOL_REQUIRED_HOOK_TARGETS: set[str] = set()
orderly_shutdown_requested = threading.Event()
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
        HookRegistry.register(SIGNAL_HANDLER_SIGTERM, after_sigterm_handler, HookType.AFTER)
        HookRegistry.register(SCHEDULER_RUN_EVENT_LOOP, around_scheduler_run_event_loop, HookType.AROUND)
        HookRegistry.register(KILL_PROCESS_TREE, around_kill_process_tree, HookType.AROUND)
        required_targets.update(
            (
                MODEL_RUNNER_LOAD_MODEL,
                MODEL_RUNNER_INIT_MEMORY_POOL,
                MODEL_RUNNER_INITIALIZE,
                SIGNAL_HANDLER_SIGTERM,
                SCHEDULER_RUN_EVENT_LOOP,
                KILL_PROCESS_TREE,
            )
        )
        XPOOL_REQUIRED_HOOK_TARGETS.update(required_targets)
        install_apply_hooks_guard()
    except Exception as exc:
        raise SystemExit(f"xpool SGLang plugin failed to install: {exc}") from exc


class SchedulerShutdown(BaseException):
    """Interrupt one scheduler event loop for orderly Instance departure."""


class SchedulerShutdownFailure(BaseException):
    """Terminate one scheduler without routing cleanup failure through SIGQUIT."""


def after_sigterm_handler(
    result: object,
    handler: SignalHandler,
    signum: int | None = None,
    frame: FrameType | None = None,
) -> None:
    """Mark the parent process's normal request-draining shutdown path."""

    orderly_shutdown_requested.set()


def around_scheduler_run_event_loop(
    original_fn: Callable[[Scheduler], None],
    scheduler: Scheduler,
) -> None:
    """Stop scheduler submission and detach xpool resources on SIGTERM."""

    shutdown_started = False

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        raise SchedulerShutdown

    with sighandle(signal.SIGTERM, request_shutdown):
        try:
            original_fn(scheduler)
        except SchedulerShutdown:
            model_runner = scheduler.tp_worker.model_runner
            try:
                torch.cuda.synchronize(model_runner.device)
                XpoolModelRuntime.require(model_runner).detach(model_runner)
            except Exception as error:
                logger.exception("xpool scheduler failed orderly Instance departure")
                raise SchedulerShutdownFailure from error


def around_kill_process_tree(
    original_fn: Callable[[int | None, bool, int | None, float | None], None],
    parent_pid: int | None,
    include_parent: bool = True,
    skip_pid: int | None = None,
    wait_timeout: float | None = None,
) -> None:
    """Attempt scheduler departure before always running SGLang cleanup.

    Orderly drain applies only to the marked serve parent. Drain failures are
    diagnostic because SGLang's original process-tree cleanup remains the
    authoritative fallback and is always invoked with its original arguments.
    """

    try:
        if orderly_shutdown_requested.is_set() and parent_pid == os.getpid():
            try:
                schedulers = collect_scheduler_processes()
                for scheduler in schedulers:
                    try:
                        scheduler.send_signal(signal.SIGTERM)
                    except psutil.NoSuchProcess:
                        pass
                exited, alive = psutil.wait_procs(schedulers, timeout=SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS)
                failed = tuple(process for process in exited if getattr(process, "returncode", None) not in {None, 0})
                if failed:
                    logger.error(
                        "xpool scheduler orderly departure failed for processes: %s",
                        ", ".join(f"{process.pid}({process.returncode})" for process in failed),
                    )
                if alive:
                    logger.error(
                        "xpool scheduler orderly departure timed out for processes: %s",
                        ", ".join(str(process.pid) for process in alive),
                    )
            except Exception:
                logger.exception("xpool scheduler orderly departure failed before SGLang cleanup")
    finally:
        original_fn(parent_pid, include_parent, skip_pid, wait_timeout)


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
    binding = XpoolModelBinding.resolve(
        model_runner,
        server_args,
        supports_dp_attention=adapter.supports_dp_attention,
    )
    binding.validate_server_args(server_args)
    bootstrap.init(int(binding.cuda_device), RuntimeRole.INSTANCE)
    devkit.install()
    adapter.validate_before_load(model_runner)

    runtime = XpoolModelRuntime.attach(model_runner, binding)
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
        runtime unless instance loopback is enabled.
    """

    runtime = XpoolModelRuntime.require(model_runner)
    binding = runtime.binding
    config = get_global_config()
    if config.debug.loopback.site is LoopbackSite.INSTANCE:
        return result
    try:
        server_args = model_runner.server_args
        workload = derive_workload(model_runner, binding, server_args)
        transport = derive_transport_attributes(binding, workload)
        runtime.instance = Instance.start(
            instance_id=binding.instance_id,
            rank=binding.worker_rank,
            transport=transport,
            workload=workload,
        )
        if config.debug.loopback.site in {None, LoopbackSite.FFNAGENT}:
            runtime.instance.wait_for_fabric_executable()
        runtime.instance.attach_arena_from_daemon()
        runtime.instance.start_failure_monitor()
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

    config = get_global_config()
    if config.debug.loopback.site in {LoopbackSite.INSTANCE, LoopbackSite.ATNAGENT}:
        return result
    runtime = XpoolModelRuntime.require(model_runner)
    if runtime.instance is None:
        raise RuntimeError("xpool ModelRunner.initialize hook requires a started Instance runtime")
    runtime.instance.publish_initialized()
    runtime.instance.wait_for_ready()
    return result


def derive_workload(
    model_runner: ModelRunner,
    binding: XpoolModelBinding,
    server_args: ServerArgs,
) -> FfnWorkload:
    """Derive the complete FFN executor workload from resolved SGLang state.

    Args:
        model_runner: Loaded runner with memory-pool concurrency and installed
            FFN shims.
        binding: Validated xpool instance and parallel identity.
        server_args: Resolved SGLang graph and eager workload settings.

    Returns:
        Strict rank-independent workload registered with the daemon.

    Raises:
        RuntimeError: If dtype, layers, bucket ceilings, or model config bytes
            cannot be resolved without guessing.
        OSError: If the model's config file cannot be read.
    """

    dtype = getattr(model_runner.model_config, "dtype", None)
    dtype_by_torch = {
        torch.bfloat16: TensorDType.BF16,
        torch.float16: TensorDType.FP16,
        torch.float32: TensorDType.FP32,
    }
    tensor_dtype = dtype_by_torch.get(dtype)
    if tensor_dtype is None:
        raise RuntimeError(f"xpool FFN workload does not support SGLang dtype {dtype!r}")

    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("xpool cannot derive an FFN workload before SGLang installs the model")
    shims = tuple(sorted(iter_ffn_shims(model), key=lambda shim: shim.layer_id))
    if not shims:
        raise RuntimeError("xpool cannot derive an FFN workload without installed FFN shims")
    hidden_sizes = {shim.hidden_size for shim in shims}
    if len(hidden_sizes) != 1:
        raise RuntimeError(f"xpool FFN shims disagree on hidden size: {sorted(hidden_sizes)}")
    layers = tuple(FfnLayerSpec(layer_id=shim.layer_id, kind=shim.layer_kind) for shim in shims)

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
    return FfnWorkload(
        model_config_digest=hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
        dtype=tensor_dtype,
        hidden_size=hidden_sizes.pop(),
        layers=layers,
        max_decode_rows=max_decode_rows,
        max_prefill_rows=max_prefill_rows,
    )


def resolved_graph_capacity(
    *,
    eager_capacity: int,
    enabled: bool,
    buckets: object,
    maximum: object,
    label: str,
) -> int:
    """Resolve one workload capacity from eager and enabled graph geometry.

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


def derive_transport_attributes(
    binding: XpoolModelBinding,
    workload: FfnWorkload,
) -> InstanceTransportAttributes:
    """Derive daemon registration transport attributes for one SGLang rank.

    Args:
        binding: Validated xpool attention TP/DP rank binding.
        workload: Rank-independent hidden geometry and row coverage.

    Returns:
        Transport geometry covering every eager and captured workload shape.
    """

    return InstanceTransportAttributes(
        hidden_size=workload.hidden_size,
        max_tokens=max(workload.max_decode_rows, workload.max_prefill_rows),
        atn_tp_rank=binding.atn_tp_rank,
        atn_tp_size=binding.atn_tp_size,
        atn_dp_rank=binding.atn_dp_rank,
        atn_dp_size=binding.atn_dp_size,
    )
