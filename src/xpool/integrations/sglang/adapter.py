"""Common contracts for SGLang model adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import ServerArgs
from torch import nn

from xpool.config import get_global_config
from xpool.integrations.sglang.shim import FfnShimModule, iter_ffn_shims
from xpool.integrations.sglang.topology import SglangAttentionTopology, SglangModelMetadata
from xpool.native.ffn import LayerKind
from xpool.runtime.instance import InstanceRankRuntime

# A hook handler is either an SGLang around/before/after wrapper callable or a class
# used as a REPLACE target. ``object`` (not ``Any``) is a deliberate, ANN401-safe escape
# hatch: SGLang hook handlers are variadic and their argument types are enforced by the
# SGLang HookRegistry contract, not by xpool's type checker.
type SglangHookHandler = Callable[..., object] | type

DECODER_FFN_WEIGHT_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp(?:\.|$)")


def filter_decoder_ffn_weights[W](weights: Iterable[tuple[str, W]]) -> Iterator[tuple[str, W]]:
    """Yield checkpoint tensors outside canonical decoder FFN subtrees."""

    for name, tensor in weights:
        if DECODER_FFN_WEIGHT_PATTERN.match(name) is None:
            yield name, tensor


@dataclass(frozen=True, slots=True)
class SglangCudaPlacement:
    """SGLang CUDA placement arguments derived from xpool ATN devices.

    Attributes:
        base_gpu_id: SGLang ``base_gpu_id`` corresponding to the first ATN CUDA device.
        gpu_id_step: SGLang ``gpu_id_step`` between adjacent ATN CUDA devices.
    """

    base_gpu_id: int
    gpu_id_step: int

    @classmethod
    def derive(cls, atn_cuda_devices: Sequence[int]) -> SglangCudaPlacement:
        """Derive SGLang CUDA placement from ordered attention devices.

        Args:
            atn_cuda_devices: Strictly increasing arithmetic device sequence.

        Returns:
            SGLang base GPU id and rank step.

        Raises:
            RuntimeError: If devices are empty, unordered, or nonuniform.
        """

        if not atn_cuda_devices:
            raise RuntimeError("xpool SGLang integration requires at least one attention CUDA device")
        gpu_id_step = atn_cuda_devices[1] - atn_cuda_devices[0] if len(atn_cuda_devices) > 1 else 1
        if any(left >= right for left, right in pairwise(atn_cuda_devices)):
            raise RuntimeError("xpool SGLang integration requires strictly increasing attention CUDA devices")
        if any((right - left) != gpu_id_step for left, right in pairwise(atn_cuda_devices)):
            raise RuntimeError(
                "xpool SGLang integration requires attention CUDA devices to match base_gpu_id + rank * gpu_id_step"
            )
        return cls(base_gpu_id=atn_cuda_devices[0], gpu_id_step=gpu_id_step)


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
class SglangInstanceRankBinding:
    """xpool runtime identity for one SGLang model runner.

    ``instance_id`` is the human-readable model id from ``XPOOL_CONFIG``; the
    integer ``instance_index`` becomes the static transport-arena identity used
    to route FFN results back to the right instance.

    Attributes:
        instance_id: Human-readable model/instance id from xpool config.
        model_path: Resolved absolute model path matched against SGLang.
        instance_index: Config-order identity published during transport setup.
        worker_rank: SGLang model-worker rank for this model runner.
        cuda_device: Physical CUDA device used by this SGLang rank.
        worker_world_size: Expected SGLang model-worker world size.
        sglang_base_gpu_id: Required SGLang base_gpu_id.
        sglang_gpu_id_step: Required SGLang gpu_id_step.
        atn_tp_rank: Attention tensor-parallel rank for this SGLang rank.
        atn_tp_size: Attention tensor-parallel size for this SGLang rank.
        atn_dp_rank: Attention data-parallel rank for this SGLang rank.
        atn_dp_size: Attention data-parallel size for this SGLang rank.
    """

    instance_id: str
    model_path: Path
    instance_index: int
    worker_rank: int
    cuda_device: int
    worker_world_size: int
    sglang_base_gpu_id: int
    sglang_gpu_id_step: int
    atn_tp_rank: int
    atn_tp_size: int
    atn_dp_rank: int
    atn_dp_size: int

    @classmethod
    def resolve(
        cls,
        model_runner: ModelRunner,
        server_args: ServerArgs,
        *,
        supports_dp_attention: bool,
    ) -> SglangInstanceRankBinding:
        """Resolve an xpool binding for one SGLang model runner.

        Args:
            model_runner: Runner whose resolved model path must appear in xpool config.
            server_args: Resolved SGLang launch arguments for this runner.
            supports_dp_attention: Whether the selected model adapter supports DPA.

        Returns:
            Validated runtime identity binding.

        Raises:
            RuntimeError: If the model is absent or SGLang rank metadata is invalid.
            ConfigError: If model topology cannot be derived safely.
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
        spec = SglangModelMetadata.load(config.model_path_of(model.id), model_id=model.id)
        policy = SglangAttentionTopology.from_server_args(
            spec,
            server_args,
            atnagent_count=config.atn_world_size,
            supports_dp_attention=supports_dp_attention,
        )
        placement = SglangCudaPlacement.derive(config.atn.devices)
        worker_rank = model_runner.ps.tp_rank
        cuda_device = model_runner.gpu_id
        if (
            not isinstance(worker_rank, int)
            or isinstance(worker_rank, bool)
            or not 0 <= worker_rank < policy.worker_world_size
        ):
            raise RuntimeError(
                f"xpool SGLang plugin requires ModelRunner.ps.tp_rank in [0, {policy.worker_world_size}), "
                f"got {worker_rank!r}"
            )
        attn_cp_size = model_runner.ps.attn_cp_size
        if not isinstance(attn_cp_size, int) or isinstance(attn_cp_size, bool) or attn_cp_size <= 0:
            raise RuntimeError("xpool SGLang plugin requires positive integer ModelRunner.ps.attn_cp_size")
        if attn_cp_size != 1:
            raise RuntimeError(f"xpool SGLang plugin requires ModelRunner.ps.attn_cp_size=1, got {attn_cp_size}")
        atn_tp_rank, atn_tp_size, atn_dp_rank, atn_dp_size = compute_dp_attention_world_info(
            policy.atn_dp_size > 1,
            worker_rank,
            policy.worker_world_size,
            policy.atn_dp_size,
            attn_cp_size,
        )
        if (atn_tp_size, atn_dp_size) != (policy.atn_tp_size, policy.atn_dp_size):
            raise RuntimeError("xpool SGLang rank geometry disagrees with the validated parallel policy")
        if worker_rank != atn_dp_rank * atn_tp_size + atn_tp_rank:
            raise RuntimeError("xpool SGLang rank does not use TP-fastest attention coordinates")
        return cls(
            instance_id=instance.id,
            model_path=model_path,
            instance_index=instance.instance_index,
            worker_rank=worker_rank,
            cuda_device=cuda_device,
            worker_world_size=policy.worker_world_size,
            sglang_base_gpu_id=placement.base_gpu_id,
            sglang_gpu_id_step=placement.gpu_id_step,
            atn_tp_rank=atn_tp_rank,
            atn_tp_size=atn_tp_size,
            atn_dp_rank=atn_dp_rank,
            atn_dp_size=atn_dp_size,
        )

    def bind_shim_runtime(self, model_runner: ModelRunner) -> None:
        """Bind request-time runtime metadata to every loaded FFN shim.

        Args:
            model_runner: Loaded runner whose module tree may contain shims.

        Side Effects:
            Mutates each discovered shim's request-time runtime fields.
        """

        model = getattr(model_runner, "model", None)
        if model is None:
            return
        architectures = getattr(model_runner.model_config.hf_config, "architectures", None)
        model_architecture = (
            ",".join(str(architecture) for architecture in architectures) if architectures else "unknown"
        )
        for layer_ordinal, shim in enumerate(sorted(iter_ffn_shims(model), key=lambda item: item.layer_id)):
            shim.bind_runtime(
                layer_ordinal=layer_ordinal,
                model_architecture=model_architecture,
                atn_dp_size=self.atn_dp_size,
            )

    def validate_result_group(self, group: GroupCoordinator) -> None:
        """Require SGLang's complete tensor group to match all AtnAgents.

        Args:
            group: Initialized SGLang tensor-model-parallel coordinator.

        Raises:
            RuntimeError: If global membership, order, size, or this rank's
                local coordinate differs from the bound TP-fastest world.
        """

        expected_ranks = tuple(range(self.worker_world_size))
        if tuple(group.ranks) != expected_ranks:
            raise RuntimeError(f"xpool requires SGLang result-group ranks {expected_ranks}, got {tuple(group.ranks)}")
        if group.world_size != self.worker_world_size:
            raise RuntimeError(
                f"xpool requires SGLang result-group size {self.worker_world_size}, got {group.world_size}"
            )
        if group.rank_in_group != self.worker_rank or group.rank != self.worker_rank:
            raise RuntimeError(
                f"xpool requires SGLang result-group rank {self.worker_rank}, got "
                f"global={group.rank}, local={group.rank_in_group}"
            )

    def validate_server_args(self, server_args: ServerArgs) -> None:
        """Validate resolved SGLang placement against this binding.

        Args:
            server_args: Resolved SGLang launch arguments.

        Raises:
            RuntimeError: If launch dimensions or CUDA placement differ.
        """

        for label, actual, expected in (
            ("tp_size", server_args.tp_size, self.worker_world_size),
            ("dp_size", server_args.dp_size, self.atn_dp_size),
            ("base_gpu_id", server_args.base_gpu_id, self.sglang_base_gpu_id),
            ("gpu_id_step", server_args.gpu_id_step, self.sglang_gpu_id_step),
            ("enable_dp_attention", server_args.enable_dp_attention, self.atn_dp_size > 1),
        ):
            if actual != expected:
                raise RuntimeError(
                    f"xpool config expects SGLang {label}={expected} for {self.instance_id}, got {actual}"
                )
        expected_cuda_device = self.sglang_base_gpu_id + self.worker_rank * self.sglang_gpu_id_step
        if self.cuda_device != expected_cuda_device:
            raise RuntimeError(
                f"xpool config expects SGLang rank {self.worker_rank} to run on CUDA device "
                f"{expected_cuda_device}, got {self.cuda_device}"
            )


@dataclass(slots=True)
class SglangInstanceRankRuntime:
    """Runner-owned composition of static binding and live InstanceRankRuntime resources.

    Attributes:
        binding: Immutable xpool identity and topology resolved before load.
        instance_rank: Live daemon/native runtime installed after memory-pool setup,
            or ``None`` before transport startup.
    """

    binding: SglangInstanceRankBinding
    instance_rank: InstanceRankRuntime | None = None

    @classmethod
    def attach(cls, model_runner: ModelRunner, binding: SglangInstanceRankBinding) -> SglangInstanceRankRuntime:
        """Attach exactly one xpool runtime owner to a model runner."""

        if getattr(model_runner, "xpool_runtime", None) is not None:
            raise RuntimeError("xpool model runner already has an attached runtime")
        runtime = cls(binding=binding)
        setattr(model_runner, "xpool_runtime", runtime)
        return runtime

    @classmethod
    def require(cls, model_runner: ModelRunner) -> SglangInstanceRankRuntime:
        """Return the model runner's attached xpool runtime."""

        runtime = getattr(model_runner, "xpool_runtime", None)
        if not isinstance(runtime, cls):
            raise RuntimeError("xpool model runner has no attached runtime")
        return runtime

    def detach(self, model_runner: ModelRunner) -> None:
        """Release live resources and clear this exact runner attachment."""

        if self.instance_rank is not None:
            self.instance_rank.close()
            self.instance_rank = None
        if getattr(model_runner, "xpool_runtime", None) is self:
            setattr(model_runner, "xpool_runtime", None)


class SglangShimAdapter(ABC):
    """Base class for model-specific SGLang adapters.

    Attributes:
        name: Stable adapter name used in diagnostics and duplicate detection.
    """

    name: str
    supports_dp_attention = False

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

        The plugin resolves the :class:`SglangInstanceRankBinding` (and thus the integer
        instance/model identity) before calling this. The default implementation
        validates SGLang's complete tensor group; subclasses that override this
        method must call ``super().bind_runtime(model_runner)``.
        """

        SglangInstanceRankRuntime.require(model_runner).binding.validate_result_group(get_tp_group())

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Validate adapter postconditions after SGLang loads the model.

        Args:
            model_runner: SGLang model runner after model construction.

        Raises:
            RuntimeError: If expected FFN shim replacements or metadata are missing.
        """

    def require_ffn_shims(
        self,
        model: nn.Module,
        *,
        expected_layer_kinds: Sequence[LayerKind],
        allowed_shim_types: tuple[type[FfnShimModule], ...],
    ) -> tuple[FfnShimModule, ...]:
        """Require complete FFN shim coverage after model loading.

        Args:
            model: Loaded SGLang model module to inspect.
            expected_layer_kinds: Expected FFN kind for each decoder-layer ordinal.
            allowed_shim_types: Concrete shim classes this adapter may install.

        Returns:
            Discovered FFN shim modules.

        Raises:
            RuntimeError: If coverage is empty, incomplete, or has an invalid type.
        """

        shims = tuple(sorted(iter_ffn_shims(model), key=lambda shim: shim.layer_id))
        if not shims:
            raise RuntimeError(f"xpool {self.name} adapter produced no FFN shim modules")
        expected_layer_ids = tuple(range(len(expected_layer_kinds)))
        actual_layer_ids = tuple(shim.layer_id for shim in shims)
        if actual_layer_ids != expected_layer_ids:
            raise RuntimeError(
                f"xpool {self.name} adapter requires FFN shim layer ids {expected_layer_ids}, got {actual_layer_ids}"
            )
        non_shim = [type(shim).__name__ for shim in shims if not isinstance(shim, allowed_shim_types)]
        if non_shim:
            raise RuntimeError(
                f"xpool {self.name} adapter left non-xpool FFN modules ({', '.join(non_shim)}); "
                "expected every decoder FFN layer to be replaced by an xpool shim"
            )
        mismatched_kinds = [
            (shim.layer_id, shim.layer_kind, expected_layer_kinds[shim.layer_id])
            for shim in shims
            if shim.layer_kind is not expected_layer_kinds[shim.layer_id]
        ]
        if mismatched_kinds:
            details = ", ".join(
                f"layer {layer_id}: got {actual.value}, expected {expected.value}"
                for layer_id, actual, expected in mismatched_kinds
            )
            raise RuntimeError(f"xpool {self.name} adapter found mismatched FFN layer kinds ({details})")
        return shims

    def require_full_mlp_boundaries(
        self,
        model: nn.Module,
        shims: Sequence[FfnShimModule],
        *,
        allow_reduce_scatter: bool | None = None,
    ) -> None:
        """Prove each replaced decoder MLP receives a full hidden-state buffer.

        Args:
            model: Loaded SGLang model containing decoder layers and FFN shims.
            shims: Complete shim set already validated by ``require_ffn_shims``.
            allow_reduce_scatter: Optional expected value of the owning layer
                communicator's reduce-scatter capability.

        Raises:
            RuntimeError: If a shim has no owning decoder layer, the pinned
                scatter metadata is missing, the MLP mode is not ``FULL``, or
                reduce-scatter capability differs from adapter policy.
        """

        owners = {
            id(mlp): (name, module)
            for name, module in model.named_modules()
            if isinstance((mlp := getattr(module, "mlp", None)), FfnShimModule)
        }
        for shim in shims:
            owner = owners.get(id(shim))
            if owner is None:
                raise RuntimeError(f"xpool {self.name} adapter cannot locate the decoder layer for FFN {shim.layer_id}")
            module_name, layer = owner
            scatter_modes = getattr(layer, "layer_scatter_modes", None)
            if not isinstance(scatter_modes, LayerScatterModes):
                raise RuntimeError(f"xpool {self.name} adapter found no SGLang scatter metadata on {module_name}")
            if scatter_modes.mlp_mode is not ScatterMode.FULL:
                raise RuntimeError(
                    f"xpool {self.name} adapter requires ScatterMode.FULL at {module_name}.mlp, "
                    f"got {scatter_modes.mlp_mode!r}"
                )
            if allow_reduce_scatter is not None:
                communicator = getattr(layer, "layer_communicator", None)
                actual = getattr(communicator, "allow_reduce_scatter", None)
                if actual is not allow_reduce_scatter:
                    raise RuntimeError(
                        f"xpool {self.name} adapter requires allow_reduce_scatter={allow_reduce_scatter} "
                        f"at {module_name}, got {actual!r}"
                    )


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
