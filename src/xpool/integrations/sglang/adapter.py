"""Common contracts for SGLang model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.integrations.sglang.shim import FfnShimModule, iter_ffn_shims

# A hook handler is either an SGLang around/before/after wrapper callable or a class
# used as a REPLACE target. ``object`` (not ``Any``) is a deliberate, ANN401-safe escape
# hatch: SGLang hook handlers are variadic and their argument types are enforced by the
# SGLang HookRegistry contract, not by xpool's type checker.
type SglangHookHandler = Callable[..., object] | type


@dataclass(frozen=True, slots=True)
class SglangHook:
    """One SGLang hook owned by a model adapter.

    ``kind`` reuses SGLang's own ``HookType`` rather than a xpool-defined enum: the
    four hook kinds (BEFORE/AFTER/AROUND/REPLACE) are an SGLang contract, and xpool
    only forwards them to ``HookRegistry.register``.

    Attributes:
        target: Fully qualified SGLang hook target path.
        handler: Hook callable or replacement class registered with SGLang.
        kind: SGLang hook kind controlling how ``handler`` is applied.
    """

    target: str
    handler: SglangHookHandler
    kind: HookType


@dataclass(frozen=True, slots=True)
class XpoolModelBinding:
    """xpool runtime identity for one SGLang model runner.

    ``instance_id`` is the human-readable model id from ``XPOOL_CONFIG``; the integer
    ``instance_index``/``model_index`` are the identities fed to the xpool FFN shim ABI
    so the device agent can route FFN results back to the right instance/model.

    Attributes:
        instance_id: Human-readable model/instance id from xpool config.
        model_path: Resolved absolute model path matched against SGLang.
        instance_index: Integer SGLang instance identity for native ABI calls.
        model_index: Integer model identity for native ABI calls.
        sglang_tp_size: Expected SGLang tensor-parallel size for this instance.
        sglang_dp_size: Expected SGLang data-parallel size for this instance.
    """

    instance_id: str
    model_path: Path
    instance_index: int
    model_index: int
    sglang_tp_size: int
    sglang_dp_size: int


class SglangModelAdapter(ABC):
    """Base class for model-specific SGLang adapters.

    Attributes:
        name: Stable adapter name used in diagnostics and duplicate detection.
    """

    name: str

    @abstractmethod
    def hooks(self) -> Sequence[SglangHook]:
        """Return SGLang hooks requested by this adapter.

        Returns:
            Hook declarations registered when the xpool SGLang plugin installs.
        """

    @abstractmethod
    def matches(self, model_runner: ModelRunner) -> bool:
        """Return whether this adapter owns a loaded or loading model runner.

        Args:
            model_runner: SGLang model runner before or after model construction.

        Returns:
            ``True`` when this adapter should validate and bind the runner.
        """

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Reject unsupported SGLang state before model construction.

        Args:
            model_runner: SGLang model runner before model construction.

        Raises:
            RuntimeError: If adapter-specific preconditions are not met.
        """

    def bind_runtime(self, model_runner: ModelRunner) -> None:
        """Bind xpool runtime metadata to the SGLang model runner before load.

        Args:
            model_runner: SGLang model runner before model construction.

        Side Effects:
            Subclasses may attach adapter-specific metadata required during
            SGLang model construction.

        The plugin resolves the :class:`XpoolModelBinding` (and thus the integer
        instance/model identity) before calling this; subclasses may override to add
        model-specific pre-load wiring but should not re-resolve the binding.
        """

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Validate adapter postconditions after SGLang loads the model.

        Args:
            model_runner: SGLang model runner after model construction.

        Raises:
            RuntimeError: If expected FFN shim replacements or metadata are missing.
        """


def model_runner_architectures(model_runner: ModelRunner) -> frozenset[str]:
    """Read Hugging Face architecture names from a SGLang model runner.

    Args:
        model_runner: SGLang model runner exposing ``model_config.hf_config``.

    Returns:
        Architecture strings from the Hugging Face config, or an empty set when
        the config does not declare them.
    """

    config = model_runner.model_config.hf_config
    architectures = getattr(config, "architectures", None)
    if not architectures:
        return frozenset()
    return frozenset(str(architecture) for architecture in architectures)


def assert_ffn_shim_coverage(
    model: nn.Module,
    *,
    adapter_name: str,
    expected_layer_count: int,
    allowed_shim_types: tuple[type[FfnShimModule], ...],
) -> tuple[FfnShimModule, ...]:
    """Assert every decoder FFN layer is an xpool shim of an allowed type.

    Args:
        model: Loaded SGLang model module to inspect.
        adapter_name: Adapter name used in error messages.
        expected_layer_count: Expected decoder-layer count from model config.
        allowed_shim_types: Concrete shim classes this adapter may install.

    Returns:
        Tuple of discovered FFN shim modules.

    Raises:
        RuntimeError: If no shims are found, the shim count is wrong, or a shim
            has a type outside the adapter's allowed set.

    Catches a partial class-REPLACE (SGLang ``apply_hooks`` swallows per-target apply
    errors) that would leave a mix of native and shimmed FFN layers; native layers whose
    weights were already dropped by ``filter_ffn_weights`` would then serve garbage.
    """

    shims = tuple(iter_ffn_shims(model))
    if not shims:
        raise RuntimeError(f"xpool {adapter_name} adapter produced no FFN shim modules")
    if len(shims) != expected_layer_count:
        raise RuntimeError(
            f"xpool {adapter_name} adapter produced {len(shims)} FFN shim modules but the model "
            f"has {expected_layer_count} decoder layers; a class-REPLACE likely failed to apply"
        )
    non_shim = [type(shim).__name__ for shim in shims if not isinstance(shim, allowed_shim_types)]
    if non_shim:
        raise RuntimeError(
            f"xpool {adapter_name} adapter left non-xpool FFN modules ({', '.join(non_shim)}); "
            "expected every decoder FFN layer to be replaced by an xpool shim"
        )
    return shims


def inject_shim_identity(model_runner: ModelRunner, binding: XpoolModelBinding) -> None:
    """Stamp the model metadata and integer identity onto every FFN shim after load.

    Args:
        model_runner: Loaded SGLang model runner whose module tree may contain shims.
        binding: xpool identity resolved for this SGLang instance.

    Side Effects:
        Mutates each discovered FFN shim by setting its diagnostic architecture
        and integer instance/model identity.

    Shims are constructed by SGLang before the binding is known, so the identity is
    injected post-load from the actual model runner config; each shim then forwards
    it to the native FFN op so the device agent can route results back to the right
    instance/model.
    """

    model = getattr(model_runner, "model", None)
    if model is None:
        return
    architectures = getattr(model_runner.model_config.hf_config, "architectures", None)
    model_architecture = ",".join(str(architecture) for architecture in architectures) if architectures else "unknown"
    for shim in iter_ffn_shims(model):
        shim.bind_identity(
            binding.instance_index,
            binding.model_index,
            model_architecture=model_architecture,
        )


def resolve_model_binding(model_runner: ModelRunner) -> XpoolModelBinding:
    """Resolve the xpool binding for a SGLang model runner without mutating it.

    Args:
        model_runner: SGLang model runner whose model path must appear in xpool config.

    Returns:
        Runtime identity binding for the matched configured model.

    Raises:
        ConfigError: If the process-global xpool config is not initialized or
            parallel-policy derivation fails.
        OSError: If the matched model ``config.json`` cannot be opened.
        TopologyError: If the matched model metadata is incompatible with the
            configured attention/FFN device topology.
        RuntimeError: If xpool config has no model entry for the SGLang model path.

    Side Effects:
        Reads the process-global xpool config and resolves only the matched
        model's metadata through SGLang so the plugin path performs topology
        validation without coupling this instance to unrelated configured
        models.

    The xpool plugin is fail-closed: once installed in an SGLang process, the loaded
    model must be declared in ``XPOOL_CONFIG`` so every FFN shim receives a stable
    instance/model identity. A matching path is required to be unique; ambiguity is an
    error.
    """

    from xpool.config import get_global_config
    from xpool.integrations.sglang.topology import derive_parallel_policy, load_model_spec

    model_path = Path(model_runner.model_config.model_path).expanduser().resolve()
    config = get_global_config()
    index = config.model_index_by_path.get(model_path)
    if index is None:
        raise RuntimeError(f"xpool config has no model entry for SGLang model path {model_path}")

    model = config.models[index]
    instance = config.serving_instances[index]
    spec = load_model_spec(config.model_path_of(model.id), model_id=model.id)
    policy = derive_parallel_policy(
        spec,
        attention_device_count=len(config.devices.attention_cuda_devices),
        ffn_tp_size=len(config.devices.ffn_cuda_devices),
    )
    return XpoolModelBinding(
        instance_id=instance.id,
        model_path=model_path,
        instance_index=instance.instance_index,
        model_index=instance.model_index,
        sglang_tp_size=policy.sglang_tp_size,
        sglang_dp_size=policy.sglang_dp_size,
    )


def bind_model_instance(model_runner: ModelRunner, binding: XpoolModelBinding | None = None) -> XpoolModelBinding:
    """Attach a resolved xpool binding to a SGLang model runner.

    Args:
        model_runner: SGLang model runner to mutate.
        binding: Optional binding already resolved by :func:`resolve_model_binding`.
            When omitted, the binding is resolved before attachment.

    Returns:
        Attached runtime identity binding.

    Raises:
        ConfigError: If the process-global xpool config is not initialized or
            runtime resolution fails.
        OSError: If the matched model ``config.json`` cannot be opened.
        RuntimeError: If the SGLang model path is not declared in xpool config.

    Side Effects:
        Sets ``model_runner.xpool_model_binding``.
    """

    resolved = resolve_model_binding(model_runner) if binding is None else binding
    setattr(model_runner, "xpool_model_binding", resolved)
    return resolved


def clear_model_binding(model_runner: ModelRunner, binding: XpoolModelBinding) -> None:
    """Remove a previously attached xpool binding after a rejected load.

    Args:
        model_runner: SGLang model runner that may carry ``xpool_model_binding``.
        binding: Binding instance that should be removed only if it is still current.

    Side Effects:
        Sets ``model_runner.xpool_model_binding`` to ``None`` when it still
        refers to ``binding``.
    """

    if getattr(model_runner, "xpool_model_binding", None) == binding:
        setattr(model_runner, "xpool_model_binding", None)
