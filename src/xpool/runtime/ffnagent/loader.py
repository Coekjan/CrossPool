"""Permanent FFN checkpoint loading, packing, and CUDA materialization."""

from __future__ import annotations

import concurrent.futures
import math
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

# cuda-bindings exposes this binary module without Python type stubs.
from cuda.bindings import runtime  # ty: ignore[unresolved-import]
from safetensors import safe_open

from xpool import ffn
from xpool.config import get_global_config
from xpool.fabric import FabricPlan, FabricRole, MoeFfnLayerPlan
from xpool.quantization import TensorSlice
from xpool.runtime.ffnagent import checkpoint, preparation, weights
from xpool.runtime.ffnagent.operators import moe
from xpool.utils.cuda import check_runtime


@dataclass(frozen=True, slots=True)
class LocalLayerWeightRequest:
    """One checkpoint directory and its prepared local TP resource description."""

    model_path: Path
    layer_layout: preparation.FfnLayerLayout


@dataclass(frozen=True, slots=True)
class TensorReadDescriptor:
    """Prevalidated source view, TP slice and stable destination."""

    shard_path: Path
    weight_key: str
    source_shape: tuple[int, ...]
    view_shape: tuple[int, ...]
    selection: TensorSlice | None
    staged_shape: tuple[int, ...]
    destination: torch.Tensor

    @classmethod
    def from_key_view(
        cls,
        key_view: dict[str, Path],
        *,
        weight_key: str,
        source_shape: tuple[int, ...],
        view_shape: tuple[int, ...] | None = None,
        selection: TensorSlice | None,
        destination: torch.Tensor,
    ) -> TensorReadDescriptor:
        """Resolve one selected key to a stable preallocated destination."""

        shard_path = key_view.get(weight_key)
        if shard_path is None:
            raise ValueError(f"checkpoint key {weight_key!r} is absent from its authoritative key view")
        return cls(
            shard_path=shard_path,
            weight_key=weight_key,
            source_shape=source_shape,
            view_shape=source_shape if view_shape is None else view_shape,
            selection=selection,
            staged_shape=tuple(destination.shape),
            destination=destination,
        )

    def slot_view(self, slot: torch.Tensor) -> torch.Tensor:
        """Interpret the typed prefix of one byte-addressable pinned slot."""

        return slot[: self.staged_bytes].view(self.destination.dtype).view(self.staged_shape)

    @property
    def staged_bytes(self) -> int:
        """Return exact local-TP staging bytes."""

        return math.prod(self.staged_shape) * self.destination.element_size()

    def copy_from(self, slot: torch.Tensor) -> None:
        """Finish a pinned Host transfer on the destination's current stream.

        Destinations are the loader's one-to-three-dimensional weight views.
        Non-contiguous rows use direct pitched copies rather than Torch device
        temporaries. Each span completes before advancing or returning the slot.
        """

        source = self.slot_view(slot)
        destination = self.destination
        if destination.is_contiguous():
            destination.copy_(source, non_blocking=False)
            return
        if destination.stride(-1) != 1:
            # A strided bias is an element-wide matrix column.
            source = source.unsqueeze(-1)
            destination = destination.unsqueeze(-1)
        if destination.ndim == 2:
            source = source.unsqueeze(0)
            destination = destination.unsqueeze(0)
        itemsize = destination.element_size()
        width_bytes = destination.shape[2] * itemsize
        row_count = destination.shape[1]
        stream = torch.cuda.current_stream(destination.device).cuda_stream
        for expert in range(destination.shape[0]):
            (status,) = runtime.cudaMemcpy2DAsync(
                destination[expert].data_ptr(),
                destination.stride(1) * itemsize,
                source[expert].data_ptr(),
                source.stride(1) * itemsize,
                width_bytes,
                row_count,
                runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
            check_runtime(status, "cudaMemcpy2DAsync(ffn weight)")
            (status,) = runtime.cudaStreamSynchronize(stream)
            check_runtime(status, "cudaStreamSynchronize(ffn weight)")


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
        for layer_ordinal, (layer, layer_plan) in enumerate(zip(spec.layers, model_plan.layers, strict=True)):
            if layer.kind is not layer_plan.kind:
                raise ValueError(f"Model Spec {model_index} layer {layer_ordinal} disagrees with its Layer Plan")
            if ffnagent_index not in layer_plan.ffnagent_indices:
                continue
            tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
            layer_layout = preparation.prepare_ffn_layer(
                spec,
                layer_ordinal=layer_ordinal,
                payload_dtype=instance_plan.ffn_profile.payload_dtype,
                tp_size=model_plan.tp_size,
                tp_rank=tp_rank,
            )
            requests.append(
                LocalLayerWeightRequest(
                    model_path=model_path,
                    layer_layout=layer_layout,
                )
            )
            positions.append((model_index, layer_ordinal))

            if layer_plan.local_intermediate_size != layer_layout.local_intermediate_size:
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
        device = torch.cuda.current_device()
        model_paths = tuple(dict.fromkeys(request.model_path for request in requests))
        key_views = {model_path: checkpoint.read_checkpoint_key_view(model_path) for model_path in model_paths}
        layer_weights: list[weights.FfnLayerWeights] = []
        descriptors: list[TensorReadDescriptor] = []
        for request in requests:
            materialized, layer_descriptors = prepare_layer_weight_request(
                request,
                key_view=key_views[request.model_path],
                device=device,
            )
            layer_weights.append(materialized)
            descriptors.extend(layer_descriptors)

        parallelism = get_global_config().ffn.loader.parallelism
        if not descriptors:
            return tuple(layer_weights)
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
    device: int,
) -> tuple[weights.FfnLayerWeights, tuple[TensorReadDescriptor, ...]]:
    """Allocate prepared neutral resources and bind only their actual source slices."""

    layout = request.layer_layout
    torch_device = torch.device("cuda", device)
    if isinstance(layout, preparation.DenseFfnLayerLayout):
        materialized = weights.DenseFfnWeights.allocate(gate_up=layout.gate_up, down=layout.down, device=torch_device)
        descriptors = gated_descriptors(
            key_view,
            projections=layout.layer.projections,
            layout=layout,
            source_intermediate_size=layout.layer.intermediate_size,
            source_start=layout.source_start,
            expert_id=None,
            gate_up=materialized.gate_up,
            down=materialized.down,
            read_down_bias=layout.tp_rank == 0,
        )
        return materialized, descriptors

    layer = layout.layer
    materialized = weights.MoeFfnWeights.allocate(
        gate_up=layout.experts.gate_up,
        down=layout.experts.down,
        router=layout.router,
        correction_bias=layout.router_correction_bias,
        device=torch_device,
    )
    descriptors: list[TensorReadDescriptor] = []
    source = layer.routed_experts
    if isinstance(source, ffn.StackedExpertSource):
        if layout.source_length:
            target_order = moe.expert_operation_type(layout.operation).required_gate_up_order
            segments = (
                (
                    TensorSlice(1, 2 * layout.source_start, 2 * layout.source_length),
                    TensorSlice(1, 0, 2 * layout.source_length),
                ),
            )
            if (
                source.gate_up_order is not ffn.GateUpOrder.INTERLEAVED
                or target_order is not ffn.GateUpOrder.INTERLEAVED
            ):
                source_rows = source.gate_up_order.row_slices(
                    axis=1,
                    width=layer.expert_intermediate_size,
                    start=layout.source_start,
                    length=layout.source_length,
                )
                target_rows = target_order.row_slices(
                    axis=1,
                    width=layout.local_intermediate_size,
                    start=0,
                    length=layout.source_length,
                )
                segments = tuple(zip(source_rows, target_rows, strict=True))
            for source_selection, target_selection in segments:
                target_indices = target_selection.indices()
                descriptors.extend(
                    projection_descriptors(
                        key_view,
                        projection=source.gate_up,
                        logical_shape=(
                            layer.routed_expert_count,
                            2 * layer.expert_intermediate_size,
                            layout.hidden_size,
                        ),
                        selection=source_selection,
                        dtype=layout.payload_dtype,
                        values=materialized.gate_up.values[: layer.routed_expert_count][target_indices],
                        scales=None
                        if materialized.gate_up.scales is None
                        else materialized.gate_up.scales[: layer.routed_expert_count][target_indices],
                        bias=None
                        if materialized.gate_up.bias is None
                        else materialized.gate_up.bias[: layer.routed_expert_count][target_indices],
                        read_bias=True,
                    )
                )
        descriptors.extend(
            projection_descriptors(
                key_view,
                projection=source.down,
                logical_shape=(layer.routed_expert_count, layout.hidden_size, layer.expert_intermediate_size),
                selection=TensorSlice(2, layout.source_start, layout.source_length),
                dtype=layout.payload_dtype,
                values=materialized.down.values[: layer.routed_expert_count],
                scales=None
                if materialized.down.scales is None
                else materialized.down.scales[: layer.routed_expert_count],
                bias=None if materialized.down.bias is None else materialized.down.bias[: layer.routed_expert_count],
                read_bias=layout.tp_rank == 0,
            )
        )
    else:
        for expert_id, projections in enumerate(source.experts):
            descriptors.extend(
                gated_descriptors(
                    key_view,
                    projections=projections,
                    layout=layout,
                    source_intermediate_size=layer.expert_intermediate_size,
                    source_start=layout.source_start,
                    expert_id=expert_id,
                    gate_up=materialized.gate_up,
                    down=materialized.down,
                    read_down_bias=layout.tp_rank == 0,
                )
            )
    if layer.shared_expert is not None:
        for ordinal in range(layer.shared_expert_count):
            descriptors.extend(
                gated_descriptors(
                    key_view,
                    projections=layer.shared_expert,
                    layout=layout,
                    source_intermediate_size=layer.expert_intermediate_size * layer.shared_expert_count,
                    source_start=ordinal * layer.expert_intermediate_size + layout.source_start,
                    expert_id=layer.routed_expert_count + ordinal,
                    gate_up=materialized.gate_up,
                    down=materialized.down,
                    read_down_bias=layout.tp_rank == 0 and ordinal == 0,
                )
            )
    router = materialized.router
    if router is not None:
        descriptors.extend(
            projection_descriptors(
                key_view,
                projection=layer.router,
                logical_shape=(layer.routed_expert_count, layout.hidden_size),
                selection=None,
                dtype=layer.router_projection_precision.weight_dtype(payload_dtype=layout.payload_dtype),
                values=router.projection.values,
                scales=router.projection.scales,
                bias=router.projection.bias,
                read_bias=True,
            )
        )
        if layer.router_correction_bias_key is not None:
            if router.correction_bias is None:
                raise AssertionError("prepared Router lacks its correction resource")
            descriptors.append(
                TensorReadDescriptor.from_key_view(
                    key_view,
                    weight_key=layer.router_correction_bias_key,
                    source_shape=(layer.routed_expert_count,),
                    selection=None,
                    destination=router.correction_bias,
                )
            )
    return materialized, tuple(descriptors)


def gated_descriptors(
    key_view: dict[str, Path],
    *,
    projections: ffn.GatedProjectionSpec,
    layout: preparation.FfnLayerLayout,
    source_intermediate_size: int,
    source_start: int,
    expert_id: int | None,
    gate_up: weights.ProjectionWeights,
    down: weights.ProjectionWeights,
    read_down_bias: bool,
) -> tuple[TensorReadDescriptor, ...]:
    """Bind separate gate/up sources into the prepared fused destination order."""

    gate_values = gate_up.values if expert_id is None else gate_up.values[expert_id]
    gate_scales = gate_up.scales if expert_id is None or gate_up.scales is None else gate_up.scales[expert_id]
    gate_bias = gate_up.bias if expert_id is None or gate_up.bias is None else gate_up.bias[expert_id]
    order = (
        ffn.GateUpOrder.CONCATENATED
        if isinstance(layout, preparation.DenseFfnLayerLayout)
        else moe.expert_operation_type(layout.operation).required_gate_up_order
    )
    target_rows = order.row_slices(
        axis=0,
        width=layout.local_intermediate_size,
        start=0,
        length=layout.source_length,
    )
    descriptors = []
    for source, target_selection in zip((projections.gate, projections.up), target_rows, strict=True):
        rows = target_selection.indices()
        descriptors.extend(
            projection_descriptors(
                key_view,
                projection=source,
                logical_shape=(source_intermediate_size, layout.hidden_size),
                selection=TensorSlice(0, source_start, layout.source_length),
                dtype=layout.payload_dtype,
                values=gate_values[rows],
                scales=None if gate_scales is None else gate_scales[rows],
                bias=None if gate_bias is None else gate_bias[rows],
                read_bias=True,
            )
        )
    descriptors.extend(
        projection_descriptors(
            key_view,
            projection=projections.down,
            logical_shape=(layout.hidden_size, source_intermediate_size),
            selection=TensorSlice(1, source_start, layout.source_length),
            dtype=layout.payload_dtype,
            values=down.values if expert_id is None else down.values[expert_id],
            scales=down.scales if expert_id is None or down.scales is None else down.scales[expert_id],
            bias=down.bias if expert_id is None or down.bias is None else down.bias[expert_id],
            read_bias=read_down_bias,
        )
    )
    return tuple(descriptors)


def projection_descriptors(
    key_view: dict[str, Path],
    *,
    projection: ffn.ProjectionSpec,
    logical_shape: tuple[int, ...],
    selection: TensorSlice | None,
    dtype: torch.dtype,
    values: torch.Tensor,
    scales: torch.Tensor | None,
    bias: torch.Tensor | None,
    read_bias: bool,
) -> tuple[TensorReadDescriptor, ...]:
    """Translate a scheme's source geometry into bounded, encoding-preserving reads."""

    encoded = projection.quantization.weight_layout(logical_shape=logical_shape, dtype=dtype)
    slices = (
        projection.quantization.weight_slices(logical_shape=logical_shape, selection=selection)
        if selection is not None
        else None
    )
    if (encoded.scales is None) != (projection.scales_key is None) or (encoded.scales is None) != (scales is None):
        raise ValueError("projection source and destination scales disagree with its encoding")
    source_shape = encoded.values.shape
    if projection.values_layout is ffn.ValuesSourceLayout.SCALE_BLOCKED:
        if encoded.scales is None or source_shape[-1] % encoded.scales.shape[-1]:
            raise ValueError("blocked source values require resolvable scale groups")
        source_shape = (*source_shape[:-1], encoded.scales.shape[-1], source_shape[-1] // encoded.scales.shape[-1])
    descriptors = []
    if selection is None or selection.length:
        physical_slice = None if slices is None else slices.values
        descriptors.append(
            TensorReadDescriptor.from_key_view(
                key_view,
                weight_key=projection.values_key,
                source_shape=source_shape,
                view_shape=encoded.values.shape,
                selection=physical_slice,
                destination=values
                if physical_slice is None
                else values.narrow(physical_slice.axis, 0, physical_slice.length),
            )
        )
        if encoded.scales is not None and projection.scales_key is not None and scales is not None:
            scale_slice = None if slices is None else slices.scales
            descriptors.append(
                TensorReadDescriptor.from_key_view(
                    key_view,
                    weight_key=projection.scales_key,
                    source_shape=encoded.scales.shape,
                    selection=scale_slice,
                    destination=scales
                    if scale_slice is None
                    else scales.narrow(scale_slice.axis, 0, scale_slice.length),
                )
            )
    if projection.bias_key is not None and read_bias:
        if bias is None:
            raise ValueError("projection source bias lacks a prepared destination")
        bias_slice = selection if selection is not None and selection.axis < len(logical_shape) - 1 else None
        if bias_slice is None or bias_slice.length:
            descriptors.append(
                TensorReadDescriptor.from_key_view(
                    key_view,
                    weight_key=projection.bias_key,
                    source_shape=logical_shape[:-1],
                    selection=bias_slice,
                    destination=bias if bias_slice is None else bias.narrow(bias_slice.axis, 0, bias_slice.length),
                )
            )
    return tuple(descriptors)


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
                    message.descriptor.copy_from(message.slot)
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
                    compatible_dtype = source.dtype == descriptor.destination.dtype or (
                        source.is_floating_point() and descriptor.destination.is_floating_point()
                    )
                    if not compatible_dtype or tuple(source.shape) != descriptor.source_shape:
                        raise ValueError(
                            f"checkpoint tensor {descriptor.weight_key!r} has incompatible dtype or shape: expected "
                            f"{descriptor.destination.dtype} {descriptor.source_shape}, "
                            f"found {source.dtype} {tuple(source.shape)}"
                        )
                    projected = source.view(descriptor.view_shape)
                    if descriptor.selection is not None:
                        projected = projected[descriptor.selection.indices()]
                    if tuple(projected.shape) != descriptor.staged_shape:
                        raise ValueError(f"checkpoint tensor {descriptor.weight_key!r} local TP shape is invalid")
                    descriptor.slot_view(slot).copy_(projected)
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
