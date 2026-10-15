"""Permanent FFN checkpoint loading, packing, and CUDA materialization."""

from __future__ import annotations

import concurrent.futures
import math
import queue
import threading
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open

from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import FabricPlan, FabricRole, MoeFfnLayerPlan
from xpool.quantization import MXFP4_DECODE_BLOCK_LIMIT, Mxfp4TensorSource, decode_mxfp4_blocks
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
class Mxfp4TensorReadDescriptor:
    """Preflighted encoded matrix view and its final floating destination."""

    source: Mxfp4TensorSource
    blocks_shard_path: Path
    scales_shard_path: Path
    narrow_dimension: int | None
    narrow_start: int
    narrow_length: int
    staged_shape: tuple[int, ...]
    destination: torch.Tensor
    tp_rank: int

    @property
    def staged_bytes(self) -> int:
        """Return exact decoded local-TP staging bytes."""

        return math.prod(self.staged_shape) * self.destination.element_size()


type WeightReadDescriptor = TensorReadDescriptor | Mxfp4TensorReadDescriptor


@dataclass(frozen=True, slots=True)
class ProjectionRead:
    """Allocation-independent semantic projection and logical TP selection."""

    source: str | Mxfp4TensorSource
    source_shape: tuple[int, int]
    narrow_dimension: int
    narrow_start: int
    narrow_length: int


@dataclass(frozen=True, slots=True)
class StagedTensor:
    """One completed reader value retaining its acquired pinned slot."""

    descriptor: WeightReadDescriptor
    slot: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReaderFailure:
    """First ordinary reader failure published to the CUDA owner."""

    error: Exception


@dataclass(frozen=True, slots=True)
class ReaderFinished:
    """One source-group task's completion marker."""


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
    if len(model_specs) != len(fabric_plan.model_plans):
        raise ValueError("Model Specs and Model Plans are not co-indexed")

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
                moe_plan = cast(MoeFfnLayerPlan, layer_plan)
                if moe_plan.effective_topk != layer.routed_topk + layer.shared_expert_count:
                    raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} has inconsistent MoE semantics")

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
        Allocates final tensors on the current device and a temporary
        bounded pinned Host pool. Reader threads never issue CUDA operations.
    """

    if not requests:
        return ()
    try:
        model_paths = tuple(dict.fromkeys(request.model_path for request in requests))
        key_views = {model_path: checkpoint.read_checkpoint_key_view(model_path) for model_path in model_paths}
        # Phase A: All encoded metadata and local geometry precede any final
        # CUDA owner, including floating owners earlier in the request batch.
        preflight_local_sources(requests, key_views=key_views)
        # Phase B: Preserve the existing exact CUDA allocations and packing.
        device = torch.cuda.current_device()
        layer_weights: list[weights.FfnLayerWeights] = []
        descriptors: list[WeightReadDescriptor] = []
        for request in requests:
            materialized, layer_descriptors = prepare_layer_weight_request(
                request,
                key_view=key_views[request.model_path],
                device=device,
            )
            layer_weights.append(materialized)
            descriptors.extend(layer_descriptors)

        parallelism = get_global_config().ffn.loader.parallelism
        slot_bytes = max(descriptor.staged_bytes for descriptor in descriptors)
        slots: queue.Queue[torch.Tensor] = queue.Queue()
        for _ in range(parallelism):
            slots.put(torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True))

        # Phase C: Each worker owns all files needed by its paired reads.
        descriptor_groups: dict[tuple[Path, ...], list[WeightReadDescriptor]] = defaultdict(list)
        for descriptor in descriptors:
            paths = (
                (descriptor.shard_path,)
                if isinstance(descriptor, TensorReadDescriptor)
                else (descriptor.blocks_shard_path, descriptor.scales_shard_path)
            )
            descriptor_groups[tuple(sorted(set(paths)))].append(descriptor)
        copy_staged_tensors(
            descriptor_groups=descriptor_groups,
            slots=slots,
            parallelism=parallelism,
        )
        return tuple(layer_weights)
    except Exception as error:
        notes = "; ".join(getattr(error, "__notes__", ()))
        detail = f"; {notes}" if notes else ""
        raise RuntimeError(f"failed to materialize local FFN weights: {error}{detail}") from error


def gated_projection_reads(
    checkpoints: ffn.GatedFfnCheckpoint,
    *,
    source_intermediate_size: int,
    source_start: int,
    local_intermediate_size: int,
    hidden_size: int,
) -> tuple[ProjectionRead, ...]:
    """Describe the shared Dense/Expert projection geometry without allocating."""

    if isinstance(checkpoints, ffn.GatedFfnCheckpointKeys):
        sources: tuple[str | Mxfp4TensorSource, ...] = (
            checkpoints.gate_weight_key,
            checkpoints.up_weight_key,
            checkpoints.down_weight_key,
        )
    else:
        sources = (checkpoints.gate, checkpoints.up, checkpoints.down)
    return tuple(
        ProjectionRead(source, shape, dimension, source_start, local_intermediate_size)
        for source, shape, dimension in zip(
            sources,
            (
                (source_intermediate_size, hidden_size),
                (source_intermediate_size, hidden_size),
                (hidden_size, source_intermediate_size),
            ),
            (0, 0, 1),
            strict=True,
        )
    )


def request_projection_reads(request: LocalLayerWeightRequest) -> tuple[ProjectionRead, ...]:
    """Derive every requested logical projection, including wide shared Experts."""

    layer = request.layer
    width = layer.intermediate_size if isinstance(layer, ffn.DenseFfnSpec) else layer.expert_intermediate_size
    if width % request.tp_size:
        raise ValueError(f"layer {layer.layer_id} intermediate size {width} is not divisible by TP {request.tp_size}")
    local_width = width // request.tp_size
    local_start = request.tp_rank * local_width
    if isinstance(layer, ffn.DenseFfnSpec):
        triplets = ((layer.checkpoint, width, local_start),)
    else:
        triplets = tuple((expert, width, local_start) for expert in layer.checkpoint.routed_experts)
        if layer.checkpoint.shared_expert is not None:
            triplets += tuple(
                (layer.checkpoint.shared_expert, width * layer.shared_expert_count, ordinal * width + local_start)
                for ordinal in range(layer.shared_expert_count)
            )
    return tuple(
        read
        for triplet, source_width, start in triplets
        for read in gated_projection_reads(
            triplet,
            source_intermediate_size=source_width,
            source_start=start,
            local_intermediate_size=local_width,
            hidden_size=request.hidden_size,
        )
    )


def preflight_local_sources(
    requests: tuple[LocalLayerWeightRequest, ...],
    *,
    key_views: dict[Path, dict[str, Path]],
) -> None:
    """Check all encoded headers, aliases and TP geometry on the Host only.

    Reads safetensors metadata without retrieving tensor data. No final weight
    allocation or CUDA operation occurs in this phase.
    """

    layers_by_model: dict[Path, dict[int, ffn.FfnLayerSpec]] = defaultdict(dict)
    encoded_models: set[Path] = set()
    checked_headers: set[tuple[Path, str, tuple[int, ...]]] = set()
    with ExitStack() as stack:
        handles: dict[Path, safe_open] = {}
        for request in requests:
            previous = layers_by_model[request.model_path].setdefault(request.layer.layer_id, request.layer)
            if previous != request.layer:
                raise ValueError("local requests disagree on the same model layer")
            for read in request_projection_reads(request):
                source = read.source
                if isinstance(source, str):
                    continue
                encoded_models.add(request.model_path)
                paths = []
                try:
                    if source.logical_shape != read.source_shape:
                        raise ValueError(f"MXFP4 logical shape {source.logical_shape} expected {read.source_shape}")
                    if not 0 <= read.narrow_start < read.source_shape[read.narrow_dimension]:
                        raise ValueError("MXFP4 local TP slice starts outside its projection")
                    if read.narrow_start + read.narrow_length > read.source_shape[read.narrow_dimension]:
                        raise ValueError("MXFP4 local TP slice ends outside its projection")
                    for key, expected in (
                        (source.blocks_key, source.blocks_shape),
                        (source.scales_key, source.scales_shape),
                    ):
                        path = key_views[request.model_path].get(key)
                        if path is None:
                            raise ValueError(f"checkpoint key {key!r} is absent from its authoritative key view")
                        paths.append(path)
                        identity = (path, key, expected)
                        if identity in checked_headers:
                            continue
                        if path not in handles:
                            handles[path] = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                        header = handles[path].get_slice(key)
                        if header.get_dtype() != "U8" or tuple(header.get_shape()) != expected:
                            raise ValueError(
                                f"MXFP4 tensor {key!r} expected U8 {expected}, "
                                f"found {header.get_dtype()} {tuple(header.get_shape())}"
                            )
                        checked_headers.add(identity)
                except Exception as error:
                    error.add_note(
                        f"MXFP4 pair {(source.blocks_key, source.scales_key)!r}, files={paths}, "
                        f"Expert={source.expert_index}, rows={source.row_part}, "
                        f"TP rank={request.tp_rank}, dimension={read.narrow_dimension}, "
                        f"slice=[{read.narrow_start},{read.narrow_start + read.narrow_length})"
                    )
                    raise
        for model_path in encoded_models:
            ffn.validate_checkpoint_references(tuple(layers_by_model[model_path].values()))


def prepare_layer_weight_request(
    request: LocalLayerWeightRequest,
    *,
    key_view: dict[str, Path],
    device: int,
) -> tuple[weights.FfnLayerWeights, tuple[WeightReadDescriptor, ...]]:
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
    torch_device = torch.device("cuda", device)

    if isinstance(layer, ffn.DenseFfnSpec):
        gate_up_weight = torch.empty(
            (2 * local_intermediate_size, request.hidden_size),
            dtype=request.payload_dtype,
            device=torch_device,
        )
        down_weight = torch.empty(
            (request.hidden_size, local_intermediate_size),
            dtype=request.payload_dtype,
            device=torch_device,
        )
        materialized = weights.DenseFfnWeights(gate_up_weight=gate_up_weight, down_weight=down_weight)
        descriptors = gated_descriptors(
            key_view,
            checkpoints=layer.checkpoint,
            source_intermediate_size=intermediate_size,
            source_start=local_start,
            local_intermediate_size=local_intermediate_size,
            hidden_size=request.hidden_size,
            destinations=(
                gate_up_weight[:local_intermediate_size],
                gate_up_weight[local_intermediate_size:],
                down_weight,
            ),
            tp_rank=request.tp_rank,
        )
        return materialized, descriptors

    routed_expert_count = layer.routed_expert_count
    total_expert_count = routed_expert_count + layer.shared_expert_count
    expert_gate_up_weight = torch.empty(
        (total_expert_count, 2 * local_intermediate_size, request.hidden_size),
        dtype=request.payload_dtype,
        device=torch_device,
    )
    expert_down_weight = torch.empty(
        (total_expert_count, request.hidden_size, local_intermediate_size),
        dtype=request.payload_dtype,
        device=torch_device,
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
                tp_rank=request.tp_rank,
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
                    tp_rank=request.tp_rank,
                )
            )

    router = None
    if request.tp_rank == 0:
        router_weight = torch.empty(
            (routed_expert_count, request.hidden_size),
            dtype=request.router_weight_dtype,
            device=torch_device,
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
            correction_bias = torch.empty((routed_expert_count,), dtype=torch.float32, device=torch_device)
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
    checkpoints: ffn.GatedFfnCheckpoint,
    expert_id: int,
    source_intermediate_size: int,
    source_start: int,
    local_intermediate_size: int,
    hidden_size: int,
    expert_gate_up_weight: torch.Tensor,
    expert_down_weight: torch.Tensor,
    tp_rank: int,
) -> tuple[WeightReadDescriptor, ...]:
    """Resolve one routed or logical shared Expert's three local projections."""

    return gated_descriptors(
        key_view,
        checkpoints=checkpoints,
        source_intermediate_size=source_intermediate_size,
        source_start=source_start,
        local_intermediate_size=local_intermediate_size,
        hidden_size=hidden_size,
        destinations=(
            expert_gate_up_weight[expert_id, :local_intermediate_size],
            expert_gate_up_weight[expert_id, local_intermediate_size:],
            expert_down_weight[expert_id],
        ),
        tp_rank=tp_rank,
    )


def gated_descriptors(
    key_view: dict[str, Path],
    *,
    checkpoints: ffn.GatedFfnCheckpoint,
    source_intermediate_size: int,
    source_start: int,
    local_intermediate_size: int,
    hidden_size: int,
    destinations: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tp_rank: int,
) -> tuple[WeightReadDescriptor, ...]:
    """Bind the shared projection geometry to existing final destination views."""

    descriptors: list[WeightReadDescriptor] = []
    reads = gated_projection_reads(
        checkpoints,
        source_intermediate_size=source_intermediate_size,
        source_start=source_start,
        local_intermediate_size=local_intermediate_size,
        hidden_size=hidden_size,
    )
    for read, destination in zip(reads, destinations, strict=True):
        if isinstance(read.source, str):
            descriptors.append(
                projected_descriptor(
                    key_view,
                    weight_key=read.source,
                    source_shape=read.source_shape,
                    narrow_dimension=read.narrow_dimension,
                    narrow_start=read.narrow_start,
                    narrow_length=read.narrow_length,
                    destination=destination,
                )
            )
        else:
            descriptors.append(
                Mxfp4TensorReadDescriptor(
                    source=read.source,
                    blocks_shard_path=key_view[read.source.blocks_key],
                    scales_shard_path=key_view[read.source.scales_key],
                    narrow_dimension=read.narrow_dimension,
                    narrow_start=read.narrow_start,
                    narrow_length=read.narrow_length,
                    staged_shape=tuple(destination.shape),
                    destination=destination,
                    tp_rank=tp_rank,
                )
            )
    return tuple(descriptors)


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
    descriptor_groups: dict[tuple[Path, ...], list[WeightReadDescriptor]],
    slots: queue.Queue[torch.Tensor],
    parallelism: int,
) -> None:
    """Run paired-source readers and synchronously install completed values."""

    messages: queue.Queue[ReaderMessage] = queue.Queue()
    cancelled = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = tuple(
            executor.submit(
                stage_source_group,
                shard_paths=shard_paths,
                descriptors=tuple(descriptors),
                slots=slots,
                messages=messages,
                cancelled=cancelled,
            )
            for shard_paths, descriptors in descriptor_groups.items()
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
                    if isinstance(message.descriptor, Mxfp4TensorReadDescriptor):
                        descriptor = message.descriptor
                        error.add_note(
                            "MXFP4 destination copy, "
                            f"pair={(descriptor.source.blocks_key, descriptor.source.scales_key)!r}, "
                            f"files={(descriptor.blocks_shard_path, descriptor.scales_shard_path)}, "
                            f"Expert={descriptor.source.expert_index}, rows={descriptor.source.row_part}, "
                            f"TP rank={descriptor.tp_rank}, dimension={descriptor.narrow_dimension}, "
                            f"slice=[{descriptor.narrow_start},{descriptor.narrow_start + descriptor.narrow_length})"
                        )
                    first_error = error
                    cancelled.set()
            finally:
                slots.put(message.slot)
        for future in futures:
            future.result()
    if first_error is not None:
        raise first_error


def stage_source_group(
    *,
    shard_paths: tuple[Path, ...],
    descriptors: tuple[WeightReadDescriptor, ...],
    slots: queue.Queue[torch.Tensor],
    messages: queue.Queue[ReaderMessage],
    cancelled: threading.Event,
) -> None:
    """Read paired sources without CUDA, publishing one terminal group marker."""

    try:
        with ExitStack() as stack:
            handles = {
                path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in shard_paths
            }
            for descriptor in descriptors:
                if cancelled.is_set():
                    break
                slot = acquire_slot(slots, cancelled)
                if slot is None:
                    break
                try:
                    if isinstance(descriptor, Mxfp4TensorReadDescriptor):
                        stage_mxfp4_tensor(
                            descriptor, handles=handles, out=slot_view(slot, descriptor), cancelled=cancelled
                        )
                    else:
                        source = handles[descriptor.shard_path].get_tensor(descriptor.weight_key)
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
                except Exception as error:
                    if isinstance(descriptor, Mxfp4TensorReadDescriptor):
                        error.add_note(
                            f"MXFP4 pair {(descriptor.source.blocks_key, descriptor.source.scales_key)!r}, "
                            f"Expert={descriptor.source.expert_index}, rows={descriptor.source.row_part}, "
                            f"TP rank={descriptor.tp_rank}, dimension={descriptor.narrow_dimension}, "
                            f"slice=[{descriptor.narrow_start},{descriptor.narrow_start + descriptor.narrow_length})"
                        )
                    slots.put(slot)
                    raise
                if cancelled.is_set():
                    slots.put(slot)
                    break
                messages.put(StagedTensor(descriptor=descriptor, slot=slot))
    except Exception as error:
        error.add_note(f"checkpoint source group files={shard_paths}")
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


def slot_view(slot: torch.Tensor, descriptor: WeightReadDescriptor) -> torch.Tensor:
    """Return the typed prefix of one byte-addressable pinned slot."""

    return slot[: descriptor.staged_bytes].view(descriptor.destination.dtype).view(descriptor.staged_shape)


def stage_mxfp4_tensor(
    descriptor: Mxfp4TensorReadDescriptor,
    *,
    handles: dict[Path, safe_open],
    out: torch.Tensor,
    cancelled: threading.Event,
) -> None:
    """Fill one slot using bounded contiguous physical row/group rectangles.

    Complementary gate/up views use separate bounded reads: coalescing them
    would couple two slot lifetimes. Each descriptor owns exactly one slot.
    Physical rectangles, including unselected parity rows, fit the block limit.
    """

    source = descriptor.source
    rows, columns = source.logical_shape
    row_start, column_start = 0, 0
    if descriptor.narrow_dimension == 0:
        row_start, rows = descriptor.narrow_start, descriptor.narrow_length
    elif descriptor.narrow_dimension == 1:
        column_start, columns = descriptor.narrow_start, descriptor.narrow_length
    group_start = column_start // 32
    group_end = (column_start + columns + 31) // 32
    row_stride = 1 if source.row_part == "all" else 2
    group_tile_limit = MXFP4_DECODE_BLOCK_LIMIT // row_stride
    for group0 in range(group_start, group_end, group_tile_limit):
        group1 = min(group0 + group_tile_limit, group_end)
        row_tile_limit = MXFP4_DECODE_BLOCK_LIMIT // ((group1 - group0) * row_stride)
        for row0 in range(row_start, row_start + rows, row_tile_limit):
            if cancelled.is_set():
                return
            row1 = min(row0 + row_tile_limit, row_start + rows)
            stage_mxfp4_rectangle(
                descriptor,
                handles=handles,
                out=out,
                row0=row0,
                row1=row1,
                group0=group0,
                group1=group1,
                row_start=row_start,
                column_start=column_start,
                columns=columns,
            )


def stage_mxfp4_rectangle(
    descriptor: Mxfp4TensorReadDescriptor,
    *,
    handles: dict[Path, safe_open],
    out: torch.Tensor,
    row0: int,
    row1: int,
    group0: int,
    group1: int,
    row_start: int,
    column_start: int,
    columns: int,
) -> None:
    """Read, decode and retire one tile before allocating the next rectangle."""

    source = descriptor.source
    stride = 1 if source.row_part == "all" else 2
    prefix = () if source.expert_index is None else (source.expert_index,)
    selection = (*prefix, slice(row0 * stride, row1 * stride), slice(group0, group1))
    packed = handles[descriptor.blocks_shard_path].get_slice(source.blocks_key)[(*selection, slice(None))]
    scales = handles[descriptor.scales_shard_path].get_slice(source.scales_key)[selection]
    if stride == 2:
        parity = int(source.row_part == "odd")
        packed = packed[parity::2]
        scales = scales[parity::2]
    packed = packed.contiguous()
    scales = scales.contiguous()
    group_count = (row1 - row0) * (group1 - group0)
    decoded = torch.empty((group_count, 32), dtype=out.dtype, device="cpu")
    decode_mxfp4_blocks(packed.reshape(group_count, 16), scales.reshape(group_count), out=decoded)
    values = decoded.view(row1 - row0, (group1 - group0) * 32)
    first_column = max(column_start, group0 * 32)
    last_column = min(column_start + columns, group1 * 32)
    out[row0 - row_start : row1 - row_start, first_column - column_start : last_column - column_start].copy_(
        values[:, first_column - group0 * 32 : last_column - group0 * 32]
    )
