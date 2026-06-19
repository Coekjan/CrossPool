"""SGLang plugin entry point for xpool."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Concatenate, ParamSpec, TypeVar, cast

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs

from xpool.cext import ensure_xpool_ops_loaded
from xpool.config import init_global_config
from xpool.integrations.sglang.adapter import (
    SglangModelAdapter,
    XpoolModelBinding,
    bind_model_instance,
    clear_model_binding,
    inject_shim_identity,
    model_runner_architectures,
    resolve_model_binding,
)
from xpool.integrations.sglang.registry import sglang_model_adapters
from xpool.integrations.sglang.server_args import validate_sglang_server_args

P = ParamSpec("P")
ReturnT = TypeVar("ReturnT")

MODEL_RUNNER_LOAD_MODEL = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
XPOOL_REQUIRED_HOOK_TARGETS: set[str] = set()


def install() -> None:
    """Install xpool hooks in an SGLang process.

    Side Effects:
        Imports and instantiates all repo-owned SGLang model adapters, registers
        their hooks in SGLang's global ``HookRegistry``, and wraps
        ``ModelRunner.load_model`` for plugin lifecycle validation.

    Raises:
        SystemExit: If native ABI preflight, adapter discovery, or hook
            registration fails. SGLang catches ordinary plugin exceptions, so
            xpool uses ``SystemExit`` for fatal fail-closed startup behavior.
    """

    try:
        ensure_xpool_ops_loaded()
        init_global_config()
        adapters = sglang_model_adapters()
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
        required_targets.add(MODEL_RUNNER_LOAD_MODEL)
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

    def guarded_apply_hooks(_registry: type[object]) -> object:
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


def around_model_runner_load_model(
    adapters: Sequence[SglangModelAdapter],
    original_fn: Callable[Concatenate[ModelRunner, P], ReturnT],
    model_runner: ModelRunner,
    *args: P.args,
    **kwargs: P.kwargs,
) -> ReturnT:
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
    for adapter in matching_adapters:
        adapter.validate_before_load(model_runner)

    bind_model_instance(model_runner, binding)
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


def validate_sglang_parallel_args(server_args: ServerArgs, binding: XpoolModelBinding) -> None:
    """Validate SGLang TP/DP launch dimensions against xpool config.

    Args:
        server_args: Resolved SGLang server arguments from the active model runner.
        binding: xpool model binding derived from ``XPOOL_CONFIG``.

    Raises:
        RuntimeError: If SGLang ``tp_size`` or ``dp_size`` does not match the
            dimensions xpool derives for the bound instance.
    """

    if server_args.tp_size != binding.sglang_tp_size:
        raise RuntimeError(
            f"xpool config expects SGLang tp_size={binding.sglang_tp_size} for {binding.instance_id}, "
            f"got {server_args.tp_size}"
        )
    if server_args.dp_size != binding.sglang_dp_size:
        raise RuntimeError(
            f"xpool config expects SGLang dp_size={binding.sglang_dp_size} for {binding.instance_id}, "
            f"got {server_args.dp_size}"
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
