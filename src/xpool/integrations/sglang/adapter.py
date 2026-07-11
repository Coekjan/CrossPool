"""Common contracts for SGLang model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType
from torch import nn

from xpool.config import get_global_config
from xpool.integrations.sglang.shim import FfnShimModule, iter_ffn_shims
from xpool.integrations.sglang.topology import derive_parallel_policy, load_model_spec

# A hook handler is either an SGLang around/before/after wrapper callable or a class
# used as a REPLACE target. ``object`` (not ``Any``) is a deliberate, ANN401-safe escape
# hatch: SGLang hook handlers are variadic and their argument types are enforced by the
# SGLang HookRegistry contract, not by xpool's type checker.
type SglangHookHandler = Callable[..., object] | type


@dataclass(frozen=True, slots=True)
class SglangCudaPlacement:
    """SGLang CUDA placement arguments derived from xpool ATN devices.

    Attributes:
        base_gpu_id: SGLang ``base_gpu_id`` corresponding to the first ATN CUDA device.
        gpu_id_step: SGLang ``gpu_id_step`` between adjacent ATN CUDA devices.
    """

    base_gpu_id: int
    gpu_id_step: int


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
    ``instance_index`` is fed to the xpool FFN shim ABI so the devagent can route
    FFN results back to the right instance.

    Attributes:
        instance_id: Human-readable model/instance id from xpool config.
        model_path: Resolved absolute model path matched against SGLang.
        instance_index: Integer SGLang instance identity for native ABI calls.
        sglang_rank: SGLang tensor-parallel rank for this model runner.
        cuda_device: Physical CUDA device used by this SGLang rank.
        sglang_tp_size: Expected SGLang tensor-parallel size for this instance.
        sglang_dp_size: Expected SGLang data-parallel size for this instance.
        sglang_base_gpu_id: Required SGLang base_gpu_id.
        sglang_gpu_id_step: Required SGLang gpu_id_step.
        enable_dp_atn: Whether SGLang DP attention should be enabled.
        atn_tp_rank: Attention tensor-parallel rank for this SGLang rank.
        atn_tp_size: Attention tensor-parallel size for this SGLang rank.
        atn_dp_rank: Attention data-parallel rank for this SGLang rank.
        atn_dp_size: Attention data-parallel size for this SGLang rank.
    """

    instance_id: str
    model_path: Path
    instance_index: int
    sglang_rank: int
    cuda_device: int
    sglang_tp_size: int
    sglang_dp_size: int
    sglang_base_gpu_id: int
    sglang_gpu_id_step: int
    enable_dp_atn: bool
    atn_tp_rank: int
    atn_tp_size: int
    atn_dp_rank: int
    atn_dp_size: int


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
    it to the native FFN op so the devagent can route results back to the right
    instance.
    """

    model = getattr(model_runner, "model", None)
    if model is None:
        return
    architectures = getattr(model_runner.model_config.hf_config, "architectures", None)
    model_architecture = ",".join(str(architecture) for architecture in architectures) if architectures else "unknown"
    for shim in iter_ffn_shims(model):
        shim.bind_identity(
            binding.instance_index,
            binding.sglang_rank,
            model_architecture=model_architecture,
            atn_tp_rank=binding.atn_tp_rank,
            atn_tp_size=binding.atn_tp_size,
            atn_dp_rank=binding.atn_dp_rank,
            atn_dp_size=binding.atn_dp_size,
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
    instance identity. A matching path is required to be unique; ambiguity is an
    error.
    """

    model_path = Path(model_runner.model_config.model_path).expanduser().resolve()
    config = get_global_config()
    matched_model_instance = next(
        (
            (model, instance)
            for model, instance in zip(config.models, config.instances, strict=True)
            if config.model_path_of(model.id).resolve() == model_path
        ),
        None,
    )
    if matched_model_instance is None:
        raise RuntimeError(f"xpool config has no model entry for SGLang model path {model_path}")

    model, instance = matched_model_instance
    spec = load_model_spec(config.model_path_of(model.id), model_id=model.id)
    policy = derive_parallel_policy(
        spec,
        atn_device_count=len(config.devices.atn_cuda_devices),
        ffn_tp_size=len(config.devices.ffn_cuda_devices),
    )
    sglang_cuda_placement = derive_sglang_cuda_placement(config.devices.atn_cuda_devices)
    sglang_rank = model_runner.tp_rank
    cuda_device = model_runner.gpu_id
    attn_cp_size = getattr(model_runner, "attn_cp_size", 1)
    if not isinstance(attn_cp_size, int) or isinstance(attn_cp_size, bool) or attn_cp_size <= 0:
        raise RuntimeError("xpool SGLang plugin requires positive integer ModelRunner.attn_cp_size")
    atn_tp_rank, atn_tp_size, atn_dp_rank, atn_dp_size = compute_dp_attention_world_info(
        policy.enable_dp_atn,
        sglang_rank,
        policy.sglang_tp_size,
        policy.sglang_dp_size,
        attn_cp_size,
    )
    return XpoolModelBinding(
        instance_id=instance.id,
        model_path=model_path,
        instance_index=instance.instance_index,
        sglang_rank=sglang_rank,
        cuda_device=cuda_device,
        sglang_tp_size=policy.sglang_tp_size,
        sglang_dp_size=policy.sglang_dp_size,
        sglang_base_gpu_id=sglang_cuda_placement.base_gpu_id,
        sglang_gpu_id_step=sglang_cuda_placement.gpu_id_step,
        enable_dp_atn=policy.enable_dp_atn,
        atn_tp_rank=atn_tp_rank,
        atn_tp_size=atn_tp_size,
        atn_dp_rank=atn_dp_rank,
        atn_dp_size=atn_dp_size,
    )


def derive_sglang_cuda_placement(atn_cuda_devices: Sequence[int]) -> SglangCudaPlacement:
    """Return SGLang base-gpu-id and gpu-id-step from xpool ATN devices."""

    if not atn_cuda_devices:
        raise RuntimeError("xpool SGLang integration requires at least one attention CUDA device")
    gpu_id_step = atn_cuda_devices[1] - atn_cuda_devices[0] if len(atn_cuda_devices) > 1 else 1
    if any(left >= right for left, right in pairwise(atn_cuda_devices)):
        raise RuntimeError("xpool SGLang integration requires strictly increasing attention CUDA devices")
    if any((right - left) != gpu_id_step for left, right in pairwise(atn_cuda_devices)):
        raise RuntimeError(
            "xpool SGLang integration requires attention CUDA devices to match base_gpu_id + rank * gpu_id_step"
        )
    return SglangCudaPlacement(base_gpu_id=atn_cuda_devices[0], gpu_id_step=gpu_id_step)


def attach_model_binding(model_runner: ModelRunner, binding: XpoolModelBinding | None = None) -> XpoolModelBinding:
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
