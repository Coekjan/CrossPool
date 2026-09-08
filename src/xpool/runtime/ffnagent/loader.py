"""Permanent FFN checkpoint loading, packing, and CUDA materialization."""

from __future__ import annotations

import concurrent.futures
import math
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import DenseFfnLayerPlan, FabricPlan, FabricRole, MoeFfnLayerPlan
from xpool.runtime.ffnagent import architecture, checkpoint, weights


@dataclass(frozen=True, slots=True)
class LocalLayerWeightRequest:
    """One engine-neutral local TP layer requested from a checkpoint.

    Attributes:
        model_path: Checkpoint directory containing the layer's exact keys.
        hidden_size: Model-wide hidden width.
        payload_dtype: Effective BF16 or FP16 execution dtype.
        router_weight_dtype: Retained Router precision, absent for Dense layers.
        layer: Intrinsic Dense or MoE layer specification.
        tp_rank: Rank within this layer's ordered Execution Group.
        tp_size: Fixed model-wide FFN tensor-parallel width.
    """

    model_path: Path
    hidden_size: int
    payload_dtype: torch.dtype
    router_weight_dtype: torch.dtype | None
    layer: ffn.FfnLayerSpec
    tp_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        """Validate request-local scalar geometry before CUDA allocation."""

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("local FFN weights require BF16 or FP16 payloads")
        if isinstance(self.layer, ffn.MoeFfnSpec) and self.router_weight_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError("MoE loading requires an explicit supported Router weight dtype")
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("TP rank must belong to a positive TP world")


@dataclass(frozen=True, slots=True)
class TensorReadDescriptor:
    """Prevalidated source projection and stable CUDA destination."""

    shard_path: Path
    weight_key: str
    source_shape: tuple[int, ...]
    narrow_dimension: int | None
    narrow_start: int
    narrow_length: int
    staged_shape: tuple[int, ...]
    destination: torch.Tensor

    @property
    def staged_bytes(self) -> int:
        """Return exact local-TP staging bytes."""

        return math.prod(self.staged_shape) * self.destination.element_size()


@dataclass(frozen=True, slots=True)
class StagedTensor:
    """One completed reader value retaining its acquired pinned slot."""

    descriptor: TensorReadDescriptor
    slot: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReaderFailure:
    """First ordinary reader failure published to the CUDA owner."""

    error: Exception


@dataclass(frozen=True, slots=True)
class ReaderFinished:
    """One shard task's completion marker."""


type ReaderMessage = StagedTensor | ReaderFailure | ReaderFinished


def materialize_layer_weights(
    *,
    fabric_plan: FabricPlan,
    model_specs: tuple[ffn.FfnModelSpec, ...],
    ffnagent_index: int,
) -> tuple[tuple[weights.FfnLayerWeights | None, ...], ...]:
    """Materialize the complete Plan-indexed local FFN weight projection.

    Args:
        fabric_plan: Daemon-authored generation-static placement.
        model_specs: Model-source specifications in Model Plan order.
        ffnagent_index: Local index in the Fabric FfnAgent suffix.

    Returns:
        Model-then-layer ordered weights, with ``None`` for nonlocal layers.

    Raises:
        ValueError: If Spec identity, Plan geometry, or local membership is
            inconsistent.
        RuntimeError: If checkpoint materialization fails.
    """

    ffnagent_count = sum(placement.role is FabricRole.FFNAGENT for placement in fabric_plan.pe_placements)
    if not 0 <= ffnagent_index < ffnagent_count:
        raise ValueError("FfnAgent index is outside the Fabric Plan")
    if len(model_specs) != len(fabric_plan.model_plans) or len(model_specs) != len(fabric_plan.instance_plans):
        raise ValueError("Model Specs, Model Plans, and Instance Plans are not co-indexed")

    requests = []
    positions = []
    materialized: list[list[weights.FfnLayerWeights | None]] = []
    config = get_global_config()
    for model_index, (spec, model_plan, instance_plan) in enumerate(
        zip(model_specs, fabric_plan.model_plans, fabric_plan.instance_plans, strict=True)
    ):
        if spec.digest() != model_plan.model_spec_digest:
            raise ValueError(f"Model Spec {model_index} does not match its Model Plan digest")
        if len(spec.layers) != len(model_plan.layers):
            raise ValueError(f"Model Spec {model_index} does not match its Model Plan semantics")

        model_weights: list[weights.FfnLayerWeights | None] = [None] * len(spec.layers)
        materialized.append(model_weights)
        model_path = config.model_path_of(spec.model_id)
        model_adapter = architecture.adapter_for(spec)
        router_weight_dtype = (
            model_adapter.router_weight_dtype(payload_dtype=instance_plan.ffn_profile.payload_dtype)
            if issubclass(model_adapter, architecture.MoeFfnModelAdapter)
            else None
        )
        for layer_ordinal, (layer, layer_plan) in enumerate(zip(spec.layers, model_plan.layers, strict=True)):
            if layer.kind is not layer_plan.kind:
                raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} disagrees with its Layer Plan")
            if ffnagent_index not in layer_plan.ffnagent_indices:
                continue
            tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
            requests.append(
                LocalLayerWeightRequest(
                    model_path=model_path,
                    hidden_size=spec.hidden_size,
                    payload_dtype=instance_plan.ffn_profile.payload_dtype,
                    router_weight_dtype=router_weight_dtype if isinstance(layer, ffn.MoeFfnSpec) else None,
                    layer=layer,
                    tp_rank=tp_rank,
                    tp_size=model_plan.tp_size,
                )
            )
            positions.append((model_index, layer_ordinal))

            intrinsic_intermediate_size = (
                layer.intermediate_size if isinstance(layer, ffn.DenseFfnSpec) else layer.expert_intermediate_size
            )
            if intrinsic_intermediate_size % model_plan.tp_size != 0:
                raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} is not divisible by its TP size")
            if layer_plan.local_intermediate_size != intrinsic_intermediate_size // model_plan.tp_size:
                raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} has inconsistent TP geometry")
            if isinstance(layer, ffn.MoeFfnSpec):
                if not isinstance(layer_plan, MoeFfnLayerPlan):
                    raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} requires a MoE Layer Plan")
                if layer_plan.effective_topk != layer.routed_topk + layer.shared_expert_count:
                    raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} has inconsistent MoE semantics")
            elif not isinstance(layer_plan, DenseFfnLayerPlan):
                raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} requires a Dense Layer Plan")

    local_weights = materialize_local_layer_weights(requests=tuple(requests))
    for (model_index, layer_ordinal), layer_weights in zip(positions, local_weights, strict=True):
        materialized[model_index][layer_ordinal] = layer_weights
    return tuple(tuple(model_weights) for model_weights in materialized)


def materialize_local_layer_weights(
    *,
    requests: tuple[LocalLayerWeightRequest, ...],
) -> tuple[weights.FfnLayerWeights, ...]:
    """Materialize ordered local layer shards with one bounded reader pool.

    Args:
        requests: Local layer requests in caller-owned stable result order.

    Returns:
        Canonical CUDA weight owners in exactly the request order.

    Raises:
        RuntimeError: If CUDA ownership, checkpoint metadata, tensor content,
            pinned staging, projection, allocation, or copy fails.

    Side Effects:
        Allocates final tensors on the current CUDA device and a temporary
        bounded pinned Host pool. Reader threads never issue CUDA operations.
    """

    if not requests:
        return ()
    try:
        cuda_device = torch.cuda.current_device()
        model_paths = tuple(dict.fromkeys(request.model_path for request in requests))
        key_views = {model_path: checkpoint.read_checkpoint_key_view(model_path) for model_path in model_paths}
        layer_weights: list[weights.FfnLayerWeights] = []
        descriptors: list[TensorReadDescriptor] = []
        for request in requests:
            materialized, layer_descriptors = prepare_layer_weight_request(
                request,
                key_view=key_views[request.model_path],
                cuda_device=cuda_device,
            )
            layer_weights.append(materialized)
            descriptors.extend(layer_descriptors)

        parallelism = get_global_config().ffn.loader.parallelism
        slot_bytes = max(descriptor.staged_bytes for descriptor in descriptors)
        slots: queue.Queue[torch.Tensor] = queue.Queue()
        for _ in range(parallelism):
            slots.put(torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True))

        descriptors_by_shard: dict[Path, list[TensorReadDescriptor]] = defaultdict(list)
        for descriptor in descriptors:
            descriptors_by_shard[descriptor.shard_path].append(descriptor)
        copy_staged_tensors(
            descriptors_by_shard=descriptors_by_shard,
            slots=slots,
            parallelism=parallelism,
        )
        return tuple(layer_weights)
    except Exception as error:
        raise RuntimeError(f"failed to materialize local FFN weights: {error}") from error


def prepare_layer_weight_request(
    request: LocalLayerWeightRequest,
    *,
    key_view: dict[str, Path],
    cuda_device: int,
) -> tuple[weights.FfnLayerWeights, tuple[TensorReadDescriptor, ...]]:
    """Allocate one canonical destination and resolve all exact-key reads."""

    layer = request.layer
    intermediate_size = (
        layer.intermediate_size if isinstance(layer, ffn.DenseFfnSpec) else layer.expert_intermediate_size
    )
    if intermediate_size % request.tp_size != 0:
        raise ValueError(
            f"layer {layer.layer_id} intermediate size {intermediate_size} is not divisible by TP {request.tp_size}"
        )
    local_intermediate_size = intermediate_size // request.tp_size
    local_start = request.tp_rank * local_intermediate_size
    device = torch.device("cuda", cuda_device)

    if isinstance(layer, ffn.DenseFfnSpec):
        gate_up_weight = torch.empty(
            (2 * local_intermediate_size, request.hidden_size),
            dtype=request.payload_dtype,
            device=device,
        )
        down_weight = torch.empty(
            (request.hidden_size, local_intermediate_size),
            dtype=request.payload_dtype,
            device=device,
        )
        materialized = weights.DenseFfnWeights(gate_up_weight=gate_up_weight, down_weight=down_weight)
        checkpoints = layer.checkpoint
        descriptors = (
            projected_descriptor(
                key_view,
                weight_key=checkpoints.gate_weight_key,
                source_shape=(intermediate_size, request.hidden_size),
                narrow_dimension=0,
                narrow_start=local_start,
                narrow_length=local_intermediate_size,
                destination=gate_up_weight[:local_intermediate_size],
            ),
            projected_descriptor(
                key_view,
                weight_key=checkpoints.up_weight_key,
                source_shape=(intermediate_size, request.hidden_size),
                narrow_dimension=0,
                narrow_start=local_start,
                narrow_length=local_intermediate_size,
                destination=gate_up_weight[local_intermediate_size:],
            ),
            projected_descriptor(
                key_view,
                weight_key=checkpoints.down_weight_key,
                source_shape=(request.hidden_size, intermediate_size),
                narrow_dimension=1,
                narrow_start=local_start,
                narrow_length=local_intermediate_size,
                destination=down_weight,
            ),
        )
        return materialized, descriptors

    routed_expert_count = layer.routed_expert_count
    total_expert_count = routed_expert_count + layer.shared_expert_count
    expert_gate_up_weight = torch.empty(
        (total_expert_count, 2 * local_intermediate_size, request.hidden_size),
        dtype=request.payload_dtype,
        device=device,
    )
    expert_down_weight = torch.empty(
        (total_expert_count, request.hidden_size, local_intermediate_size),
        dtype=request.payload_dtype,
        device=device,
    )
    descriptors = []
    for expert_id, checkpoints in enumerate(layer.checkpoint.routed_experts):
        descriptors.extend(
            expert_descriptors(
                key_view,
                checkpoints=checkpoints,
                expert_id=expert_id,
                source_intermediate_size=intermediate_size,
                source_start=local_start,
                local_intermediate_size=local_intermediate_size,
                hidden_size=request.hidden_size,
                expert_gate_up_weight=expert_gate_up_weight,
                expert_down_weight=expert_down_weight,
            )
        )
    if layer.checkpoint.shared_expert is not None:
        for shared_ordinal in range(layer.shared_expert_count):
            descriptors.extend(
                expert_descriptors(
                    key_view,
                    checkpoints=layer.checkpoint.shared_expert,
                    expert_id=routed_expert_count + shared_ordinal,
                    source_intermediate_size=intermediate_size * layer.shared_expert_count,
                    source_start=shared_ordinal * intermediate_size + local_start,
                    local_intermediate_size=local_intermediate_size,
                    hidden_size=request.hidden_size,
                    expert_gate_up_weight=expert_gate_up_weight,
                    expert_down_weight=expert_down_weight,
                )
            )

    router = None
    if request.tp_rank == 0:
        router_weight = torch.empty(
            (routed_expert_count, request.hidden_size),
            dtype=request.router_weight_dtype,
            device=device,
        )
        correction_bias = None
        descriptors.append(
            projected_descriptor(
                key_view,
                weight_key=layer.checkpoint.router_weight_key,
                source_shape=(routed_expert_count, request.hidden_size),
                narrow_dimension=None,
                narrow_start=0,
                narrow_length=0,
                destination=router_weight,
            )
        )
        if layer.checkpoint.router_correction_bias_key is not None:
            correction_bias = torch.empty((routed_expert_count,), dtype=torch.float32, device=device)
            descriptors.append(
                projected_descriptor(
                    key_view,
                    weight_key=layer.checkpoint.router_correction_bias_key,
                    source_shape=(routed_expert_count,),
                    narrow_dimension=None,
                    narrow_start=0,
                    narrow_length=0,
                    destination=correction_bias,
                )
            )
        router = weights.MoeRouterWeights(weight=router_weight, correction_bias=correction_bias)
    materialized = weights.MoeFfnWeights(
        expert_gate_up_weight=expert_gate_up_weight,
        expert_down_weight=expert_down_weight,
        router=router,
    )
    return materialized, tuple(descriptors)


def expert_descriptors(
    key_view: dict[str, Path],
    *,
    checkpoints: ffn.GatedFfnCheckpointKeys,
    expert_id: int,
    source_intermediate_size: int,
    source_start: int,
    local_intermediate_size: int,
    hidden_size: int,
    expert_gate_up_weight: torch.Tensor,
    expert_down_weight: torch.Tensor,
) -> tuple[TensorReadDescriptor, ...]:
    """Resolve one routed or logical shared Expert's three local projections."""

    return (
        projected_descriptor(
            key_view,
            weight_key=checkpoints.gate_weight_key,
            source_shape=(source_intermediate_size, hidden_size),
            narrow_dimension=0,
            narrow_start=source_start,
            narrow_length=local_intermediate_size,
            destination=expert_gate_up_weight[expert_id, :local_intermediate_size],
        ),
        projected_descriptor(
            key_view,
            weight_key=checkpoints.up_weight_key,
            source_shape=(source_intermediate_size, hidden_size),
            narrow_dimension=0,
            narrow_start=source_start,
            narrow_length=local_intermediate_size,
            destination=expert_gate_up_weight[expert_id, local_intermediate_size:],
        ),
        projected_descriptor(
            key_view,
            weight_key=checkpoints.down_weight_key,
            source_shape=(hidden_size, source_intermediate_size),
            narrow_dimension=1,
            narrow_start=source_start,
            narrow_length=local_intermediate_size,
            destination=expert_down_weight[expert_id],
        ),
    )


def projected_descriptor(
    key_view: dict[str, Path],
    *,
    weight_key: str,
    source_shape: tuple[int, ...],
    narrow_dimension: int | None,
    narrow_start: int,
    narrow_length: int,
    destination: torch.Tensor,
) -> TensorReadDescriptor:
    """Resolve one selected key to a stable preallocated destination."""

    shard_path = key_view.get(weight_key)
    if shard_path is None:
        raise ValueError(f"checkpoint key {weight_key!r} is absent from its authoritative key view")
    return TensorReadDescriptor(
        shard_path=shard_path,
        weight_key=weight_key,
        source_shape=source_shape,
        narrow_dimension=narrow_dimension,
        narrow_start=narrow_start,
        narrow_length=narrow_length,
        staged_shape=tuple(destination.shape),
        destination=destination,
    )


def copy_staged_tensors(
    *,
    descriptors_by_shard: dict[Path, list[TensorReadDescriptor]],
    slots: queue.Queue[torch.Tensor],
    parallelism: int,
) -> None:
    """Run shard readers and synchronously install completed values."""

    messages: queue.Queue[ReaderMessage] = queue.Queue()
    cancelled = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = tuple(
            executor.submit(
                stage_shard,
                shard_path=shard_path,
                descriptors=tuple(descriptors),
                slots=slots,
                messages=messages,
                cancelled=cancelled,
            )
            for shard_path, descriptors in descriptors_by_shard.items()
        )
        finished_readers = 0
        first_error: Exception | None = None
        while finished_readers < len(futures):
            message = messages.get()
            if isinstance(message, ReaderFinished):
                finished_readers += 1
                continue
            if isinstance(message, ReaderFailure):
                if first_error is None:
                    first_error = message.error
                    cancelled.set()
                continue
            try:
                if first_error is None:
                    staged_view = slot_view(message.slot, message.descriptor)
                    message.descriptor.destination.copy_(staged_view, non_blocking=False)
            except Exception as error:
                if first_error is None:
                    first_error = error
                    cancelled.set()
            finally:
                slots.put(message.slot)
        for future in futures:
            future.result()
    if first_error is not None:
        raise first_error


def stage_shard(
    *,
    shard_path: Path,
    descriptors: tuple[TensorReadDescriptor, ...],
    slots: queue.Queue[torch.Tensor],
    messages: queue.Queue[ReaderMessage],
    cancelled: threading.Event,
) -> None:
    """Read and locally project selected keys from one shard without CUDA."""

    try:
        with safe_open(str(shard_path), framework="pt", device="cpu") as checkpoint:
            for descriptor in descriptors:
                if cancelled.is_set():
                    break
                slot = acquire_slot(slots, cancelled)
                if slot is None:
                    break
                try:
                    source = checkpoint.get_tensor(descriptor.weight_key)
                    if not source.is_floating_point() or tuple(source.shape) != descriptor.source_shape:
                        raise ValueError(
                            f"checkpoint tensor {descriptor.weight_key!r} expected "
                            f"floating-point {descriptor.source_shape}, found {source.dtype} {tuple(source.shape)}"
                        )
                    projected = source
                    if descriptor.narrow_dimension is not None:
                        projected = source.narrow(
                            descriptor.narrow_dimension,
                            descriptor.narrow_start,
                            descriptor.narrow_length,
                        )
                    if tuple(projected.shape) != descriptor.staged_shape:
                        raise ValueError(f"checkpoint tensor {descriptor.weight_key!r} local TP shape is invalid")
                    slot_view(slot, descriptor).copy_(projected)
                except Exception:
                    slots.put(slot)
                    raise
                messages.put(StagedTensor(descriptor=descriptor, slot=slot))
    except Exception as error:
        messages.put(ReaderFailure(error=error))
        cancelled.set()
    finally:
        messages.put(ReaderFinished())


def acquire_slot(
    slots: queue.Queue[torch.Tensor],
    cancelled: threading.Event,
) -> torch.Tensor | None:
    """Acquire one bounded slot while remaining responsive to cancellation."""

    while not cancelled.is_set():
        try:
            return slots.get(timeout=0.1)
        except queue.Empty:
            continue
    return None


def slot_view(slot: torch.Tensor, descriptor: TensorReadDescriptor) -> torch.Tensor:
    """Return the typed prefix of one byte-addressable pinned slot."""

    return slot[: descriptor.staged_bytes].view(descriptor.destination.dtype).view(descriptor.staged_shape)
