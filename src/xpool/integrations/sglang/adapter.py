"""Common contracts for SGLang model adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_device, get_parallel
from torch import nn

from xpool.config import get_global_config
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.kv.capacity import CapacityReconciler
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool
from xpool.integrations.sglang.shim import FfnShimModule, iter_ffn_shims
from xpool.integrations.sglang.topology import SglangAttentionTopology, SglangModelMetadata
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.instance import InstanceRankRuntime

DECODER_FFN_WEIGHT_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp(?:\.|$)")


def filter_decoder_ffn_weights[W](weights: Iterable[tuple[str, W]]) -> Iterator[tuple[str, W]]:
    """Yield checkpoint tensors outside canonical decoder FFN subtrees."""

    for name, tensor in weights:
        if DECODER_FFN_WEIGHT_PATTERN.match(name) is None:
            yield name, tensor


@dataclass(frozen=True, slots=True)
class SglangInstanceRankBinding:
    """CrossPool runtime identity for one SGLang model runner.

    ``model_id`` identifies the configured Model and its Instance; the
    integer ``instance_index`` becomes the static transport-arena identity used
    to route FFN results back to the right instance.

    Attributes:
        model_id: Model ID from CrossPool config.
        model_path: Resolved absolute model path matched against SGLang.
        instance_index: Config-order identity published during transport setup.
        worker_rank: Model-worker rank and ordinal in the ordered attention view.
        device: Deployment-visible device index, retained for identity and logs.
        worker_world_size: Expected SGLang model-worker world size.
        atn_tp_rank: Attention tensor-parallel rank for this SGLang rank.
        atn_tp_size: Attention tensor-parallel size for this SGLang rank.
        atn_dp_rank: Attention data-parallel rank for this SGLang rank.
        atn_dp_size: Attention data-parallel size for this SGLang rank.
    """

    model_id: ModelId
    model_path: Path
    instance_index: int
    worker_rank: int
    device: int
    worker_world_size: int
    atn_tp_rank: int
    atn_tp_size: int
    atn_dp_rank: int
    atn_dp_size: int

    @classmethod
    def resolve(
        cls,
        model_runner: ModelRunner,
        *,
        supports_dp_attention: bool,
    ) -> SglangInstanceRankBinding:
        """Resolve a CrossPool binding for one SGLang model runner.

        Args:
            model_runner: Runner whose resolved model path must appear in CrossPool config.
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
        policy = SglangAttentionTopology.from_runtime(
            spec,
            model=model,
            atnagent_count=config.atn_world_size,
            supports_dp_attention=supports_dp_attention,
        )
        worker_rank = model_runner.ps.tp_rank
        if (
            not isinstance(worker_rank, int)
            or isinstance(worker_rank, bool)
            or not 0 <= worker_rank < policy.worker_world_size
        ):
            raise RuntimeError(
                f"xpool SGLang plugin requires ModelRunner.ps.tp_rank in [0, {policy.worker_world_size}), "
                f"got {worker_rank!r}"
            )
        if (
            not isinstance(model_runner.gpu_id, int)
            or isinstance(model_runner.gpu_id, bool)
            or model_runner.gpu_id != worker_rank
        ):
            raise RuntimeError(
                f"xpool requires SGLang rank {worker_rank} to use attention-local device {worker_rank}, "
                f"got {model_runner.gpu_id!r}"
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
            model_id=instance.model_id,
            model_path=model_path,
            instance_index=instance.instance_index,
            worker_rank=worker_rank,
            device=config.atn.devices[worker_rank],
            worker_world_size=policy.worker_world_size,
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

        model = model_runner.model
        if model is None:
            return
        architectures = model_runner.model_config.hf_config.architectures
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

    def validate_server_args(self) -> None:
        """Validate resolved SGLang placement against this binding.

        Raises:
            RuntimeError: If launch dimensions or CUDA placement differ.
        """

        for label, actual, expected in (
            ("tp_size", get_parallel().tp_size, self.worker_world_size),
            ("dp_size", get_parallel().dp_size, self.atn_dp_size),
            ("base_gpu_id", get_device().base_gpu_id, 0),
            ("gpu_id_step", get_device().gpu_id_step, 1),
            ("enable_dp_attention", get_parallel().enable_dp_attention, self.atn_dp_size > 1),
        ):
            if actual != expected:
                raise RuntimeError(f"xpool config expects SGLang {label}={expected} for {self.model_id}, got {actual}")


@dataclass(slots=True)
class SglangInstanceRankRuntime:
    """Runner-owned composition of static binding and live InstanceRankRuntime resources.

    Attributes:
        binding: Immutable CrossPool identity and topology resolved before load.
        instance_rank: Live daemon/native runtime installed after memory-pool setup,
            or ``None`` before transport startup.
    """

    binding: SglangInstanceRankBinding
    instance_rank: InstanceRankRuntime | None = None
    kv_capacity: CapacityReconciler | None = None

    @classmethod
    def attach(cls, model_runner: ModelRunner, binding: SglangInstanceRankBinding) -> SglangInstanceRankRuntime:
        """Attach exactly one CrossPool runtime owner to a model runner."""

        if getattr(model_runner, "xpool_runtime", None) is not None:
            raise RuntimeError("xpool model runner already has an attached runtime")
        runtime = cls(binding=binding)
        setattr(model_runner, "xpool_runtime", runtime)
        return runtime

    @classmethod
    def require(cls, model_runner: ModelRunner) -> SglangInstanceRankRuntime:
        """Return the model runner's attached CrossPool runtime."""

        runtime = getattr(model_runner, "xpool_runtime", None)
        if not isinstance(runtime, cls):
            raise RuntimeError("xpool model runner has no attached runtime")
        return runtime

    def detach(self, model_runner: ModelRunner) -> None:
        """Drain local device work before releasing this runner's resources.

        Failed synchronization retains the attachment and its live resources.
        """

        torch.cuda.synchronize(self.binding.worker_rank)
        if self.kv_capacity is not None:
            self.kv_capacity.close()
            self.kv_capacity = None
        pool = getattr(model_runner, "token_to_kv_pool", None)
        if isinstance(pool, ElasticMHATokenToKVPool | ElasticMLATokenToKVPool):
            pool.close()
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
            Hook declarations registered when the CrossPool SGLang plugin installs.
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
        """Bind CrossPool runtime metadata to the SGLang model runner before load.

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
    architectures = config.architectures
    if not architectures:
        return frozenset()
    return frozenset(str(architecture) for architecture in architectures)
