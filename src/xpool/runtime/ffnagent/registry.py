"""Generation-scoped FFN Graph Capture and retained execution ownership."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from time import monotonic
from typing import cast

import torch

import xpool.native
from xpool import ffn
from xpool.fabric import FabricPlan, FabricRole
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, preparation, weights
from xpool.runtime.ffnagent.device_memory import capture_pool_reservation_bytes
from xpool.runtime.ffnagent.operators import dense, moe, routing

logger = logging.getLogger(__name__)


class LaneCaptureResources:
    """Own one exclusive capture domain through confirmed native retirement."""

    def __init__(self) -> None:
        self.pool: torch.cuda.MemPool | None = None
        self.stream_owner: xpool.native.cuda.OwnedCudaStream | None = None
        self.stream: torch.cuda.ExternalStream | None = None
        self.capture_graphs: list[torch.cuda.CUDAGraph] = []
        self.capture_tensors: list[torch.Tensor] = []
        self.sealed = False

    def initialize(self) -> None:
        """Acquire owners incrementally after the Registry has retained this Lane."""

        if self.stream_owner is not None or self.sealed:
            raise RuntimeError("Lane capture resources cannot be reinitialized")
        self.stream_owner = xpool.native.cuda.OwnedCudaStream.create()
        self.stream = torch.cuda.ExternalStream(self.stream_owner.address)
        self.pool = torch.cuda.MemPool(use_on_oom=False)

    def capture_graph(self, launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
        """Retain a Graph before capture; capture_pair establishes the live domain."""

        stream = cast(torch.cuda.ExternalStream, self.stream)
        pool = cast(torch.cuda.MemPool, self.pool)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.use_mem_pool(pool), torch.cuda.stream(stream):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        self.capture_graphs.append(graph)
        # Torch annotates pool handles as a class while MemPool.id returns its native tuple.
        with torch.cuda.graph(graph, stream=stream, pool=pool.id):  # ty: ignore[invalid-argument-type]
            launch()
        stream.synchronize()
        return graph

    def capture_pair(
        self,
        *,
        primary_launch: Callable[[], None],
        control_launch: Callable[[], None],
        workspace: torch.Tensor,
    ) -> tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph, xpool.native.ffnagent.CaptureWorkspaceView | None]:
        """Capture P/C using one retained root and verify its private backing.

        Each launch runs for warmup and capture. Failures keep the acquired
        Graphs and backing for confirmed native retirement.
        """

        if self.pool is None or self.stream is None or self.sealed:
            raise RuntimeError("capture requires an initialized unsealed Lane")
        captures = []
        used = []
        address, size = workspace.data_ptr(), workspace.numel()
        for launch in (primary_launch, control_launch):
            graph = self.capture_graph(launch)
            captures.append(graph)
            used.append(size > 0 and xpool.native.cuda.graph.references_range(graph.raw_cuda_graph(), address, size))
        if used[0] != used[1]:
            raise ValueError("Primary and Control workspace use disagrees")
        view = None
        if used[0]:
            if not any(
                address >= segment["address"] and address + size <= segment["address"] + segment["total_size"]
                for segment in self.pool.snapshot()
            ):
                raise ValueError("capture workspace has no retained Lane backing owner")
            view = xpool.native.ffnagent.CaptureWorkspaceView(address, size)
        return captures[0], captures[1], view

    def release_capture_resources(self) -> None:
        """Seal after native consumption; preserve backing and stream for service."""

        if self.sealed:
            return
        if self.stream_owner is not None:
            self.stream_owner.synchronize()
        while self.capture_graphs:
            self.capture_graphs[-1].reset()
            self.capture_graphs.pop()
        self.capture_tensors.clear()
        self.sealed = True

    def report_reservation(self, estimated_bytes: int) -> None:
        """Report physical prediction error independently of strict logical coverage."""

        if self.pool is None:
            raise RuntimeError("Lane capture pool has not been acquired")
        actual_bytes = sum(segment["total_size"] for segment in self.pool.snapshot())
        logger.info(
            "capture pool measured estimated_bytes=%s actual_bytes=%s error_bytes=%s",
            estimated_bytes,
            actual_bytes,
            actual_bytes - estimated_bytes,
        )

    def close(self) -> None:
        """Release backing after native graph retirement has been confirmed."""

        self.release_capture_resources()
        self.pool = None
        self.stream = None
        if self.stream_owner is not None:
            self.stream_owner.destroy()
            self.stream_owner = None


def validate_layer_weights_against_layout(
    *,
    layer_weights: weights.FfnLayerWeights,
    layer_layout: preparation.FfnLayerLayout,
    device: int,
) -> None:
    """Require retained resources to match preparation and the current device."""

    if isinstance(layer_layout, preparation.DenseFfnLayerLayout):
        if not isinstance(layer_weights, weights.DenseFfnWeights):
            raise ValueError("Dense preparation requires Dense resources")
        layer_weights.i_proj.validate(layer_layout.i_proj)
        layer_weights.o_proj.validate(layer_layout.o_proj)
        projections = (layer_weights.i_proj, layer_weights.o_proj)
        tensors: tuple[torch.Tensor, ...] = ()
    else:
        if not isinstance(layer_weights, weights.MoeFfnWeights):
            raise ValueError("MoE preparation requires MoE resources")
        layer_weights.i_proj.validate(layer_layout.i_proj)
        layer_weights.o_proj.validate(layer_layout.o_proj)
        projections = (layer_weights.i_proj, layer_weights.o_proj)
        tensors = ()
        router = layer_weights.router
        if (router is None) != (layer_layout.router is None):
            raise ValueError("MoE Router ownership disagrees with its prepared TP role")
        if router is not None and layer_layout.router is not None:
            router.projection.validate(layer_layout.router)
            projections += (router.projection,)
            correction = layer_layout.router_correction_bias
            if (router.correction_bias is None) != (correction is None):
                raise ValueError("MoE correction resource disagrees with preparation")
            if router.correction_bias is not None and correction is not None:
                if (
                    tuple(router.correction_bias.shape) != correction.shape
                    or router.correction_bias.dtype != correction.dtype
                ):
                    raise ValueError("MoE correction shape or dtype disagrees with preparation")
                tensors += (router.correction_bias,)
    tensors += tuple(
        tensor
        for projection in projections
        for tensor in (projection.values, projection.scales, projection.bias)
        if tensor is not None
    )
    if any(tensor.device.index != device for tensor in tensors):
        raise ValueError("local FFN resources must reside on the current device")
    if len({tensor.data_ptr() for tensor in tensors}) != len(tensors):
        raise ValueError("local FFN resources must own independent storages")


def capture_dense_signature(
    plan: dense.DenseExecutionPlan,
    primary_weights: weights.DenseFfnWeights,
    control_probe: weights.DenseFfnWeights,
    *,
    owner: LaneCaptureResources,
) -> xpool.native.ffnagent.DenseCaptureDescriptor:
    """Capture primary/control Dense bodies from one retained preparation."""

    signature = plan.signature
    placeholder_layout = signature.capture_placeholder_layout()
    hidden_states = torch.zeros(
        placeholder_layout.hidden_states.shape,
        dtype=placeholder_layout.hidden_states.dtype,
        device="cuda",
    )
    owner.capture_tensors.append(hidden_states)
    partial = torch.empty(placeholder_layout.partial.shape, dtype=placeholder_layout.partial.dtype, device="cuda")
    owner.capture_tensors.append(partial)
    if owner.pool is None:
        raise AssertionError("Dense capture requires its retained Lane pool")
    stream = cast(torch.cuda.ExternalStream, owner.stream)
    with torch.cuda.use_mem_pool(owner.pool), torch.cuda.stream(stream):
        workspace = torch.empty(plan.workspace.nbytes, dtype=torch.uint8, device="cuda")
    owner.capture_tensors.append(workspace)

    operation = dense.Dense(plan, workspace)

    def launch(layer_weights: weights.DenseFfnWeights) -> None:
        operation.run_into(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            output=partial,
        )

    primary_graph, control_graph, capture_workspace = owner.capture_pair(
        primary_launch=lambda: launch(primary_weights),
        control_launch=lambda: launch(control_probe),
        workspace=workspace,
    )
    primary_resources = primary_weights.binding_view()
    control_resources = control_probe.binding_view()
    descriptor = xpool.native.ffnagent.DenseCaptureDescriptor(
        payload_dtype=signature.payload_dtype,
        payload_row_capacity=signature.payload_row_capacity,
        hidden_size=signature.hidden_size,
        local_intermediate_size=signature.local_intermediate_size,
        primary_graph_address=primary_graph.raw_cuda_graph(),
        control_graph_address=control_graph.raw_cuda_graph(),
        capture_input=hidden_states,
        capture_partial=partial,
        capture_workspace=capture_workspace,
        primary_weight_bindings=primary_resources,
        control_weight_bindings=control_resources,
    )
    return descriptor


def capture_moe_signature(
    plan: preparation.MoeExecutionPlan,
    primary_weights: weights.MoeFfnWeights,
    control_probe: weights.MoeFfnWeights,
    *,
    owner: LaneCaptureResources,
) -> xpool.native.ffnagent.MoeCaptureDescriptor:
    """Capture primary/control MoE bodies from one retained preparation."""

    signature = plan.experts.signature
    placeholder_layout = signature.capture_placeholder_layout()
    hidden_states = torch.zeros(
        placeholder_layout.hidden_states.shape,
        dtype=placeholder_layout.hidden_states.dtype,
        device="cuda",
    )
    owner.capture_tensors.append(hidden_states)
    partial = torch.empty(placeholder_layout.partial.shape, dtype=placeholder_layout.partial.dtype, device="cuda")
    owner.capture_tensors.append(partial)
    payload_rows = (
        torch.full(
            placeholder_layout.payload_rows.shape,
            signature.payload_row_capacity,
            dtype=placeholder_layout.payload_rows.dtype,
            device="cuda",
        )
        if placeholder_layout.payload_rows is not None
        else None
    )
    if payload_rows is not None:
        owner.capture_tensors.append(payload_rows)
    routing_layout = cast(TensorLayout, placeholder_layout.routing)
    routing_storage = torch.empty(
        routing_layout.shape,
        dtype=routing_layout.dtype,
        device="cuda",
    )
    owner.capture_tensors.append(routing_storage)
    route_elements = math.prod(signature.routing_shape)
    topk_ids = routing_storage[: route_elements * 4].view(torch.int32).view(signature.routing_shape)
    topk_weights = routing_storage[route_elements * 4 :].view(torch.float32).view(signature.routing_shape)
    topk_ids.copy_(
        torch.arange(signature.effective_topk, dtype=torch.int32, device="cuda")
        .remainder(signature.expert_count)
        .expand_as(topk_ids)
    )
    topk_weights.fill_(1.0 / signature.effective_topk)
    if owner.pool is None:
        raise AssertionError("MoE capture requires its retained Lane pool")
    stream = cast(torch.cuda.ExternalStream, owner.stream)
    with torch.cuda.use_mem_pool(owner.pool), torch.cuda.stream(stream):
        workspace = torch.empty(plan.workspace.nbytes, dtype=torch.uint8, device="cuda")
    owner.capture_tensors.append(workspace)
    views = plan.workspace.views(workspace)
    operation = moe.Experts(plan.experts, views[0])
    router = routing.Router(plan.router, views[1]) if plan.router is not None else None

    def launch(layer_weights: weights.MoeFfnWeights) -> None:
        router_signature = signature.router
        if router is not None and router_signature is not None:
            router_weights = layer_weights.router
            if payload_rows is None or router_weights is None:
                raise AssertionError("Router-owner capture lacks Router resources")
            router.run_into(hidden_states=hidden_states, router_weights=router_weights)
            routing.finalize_moe_routing(
                routed_ids=router.ids,
                routed_weights=router.weights,
                final_ids=topk_ids,
                final_weights=topk_weights,
                payload_rows=payload_rows,
                routed_expert_count=router_signature.routed_expert_count,
                shared_expert_count=signature.expert_count - router_signature.routed_expert_count,
                routed_scaling_factor=signature.routed_scaling_factor,
            )
        operation.run_into(
            hidden_states=hidden_states,
            i_proj=layer_weights.i_proj,
            o_proj=layer_weights.o_proj,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=partial,
        )

    primary_graph, control_graph, capture_workspace = owner.capture_pair(
        primary_launch=lambda: launch(primary_weights),
        control_launch=lambda: launch(control_probe),
        workspace=workspace,
    )
    primary_resources = primary_weights.binding_view()
    control_resources = control_probe.binding_view()
    descriptor = xpool.native.ffnagent.MoeCaptureDescriptor(
        payload_dtype=signature.payload_dtype,
        payload_row_capacity=signature.payload_row_capacity,
        hidden_size=signature.hidden_size,
        local_intermediate_size=signature.local_intermediate_size,
        expert_count=signature.expert_count,
        effective_topk=signature.effective_topk,
        routed_expert_count=(None if signature.router is None else signature.router.routed_expert_count),
        primary_graph_address=primary_graph.raw_cuda_graph(),
        control_graph_address=control_graph.raw_cuda_graph(),
        capture_input=hidden_states,
        capture_partial=partial,
        capture_workspace=capture_workspace,
        capture_routing_metadata=routing_storage,
        capture_payload_rows=payload_rows,
        primary_weight_bindings=primary_resources,
        control_weight_bindings=control_resources,
    )
    return descriptor


class FfnExecutionRegistry:
    """Retain service weights and capture backing while native execution borrows their addresses."""

    def __init__(
        self,
        *,
        layer_weights: tuple[tuple[weights.FfnLayerWeights | None, ...], ...],
    ) -> None:
        """Retain service weights through confirmed native retirement."""

        self.layer_weights = layer_weights
        self.capture_lanes: list[LaneCaptureResources] = []
        self.capture_weight_pairs: dict[
            execution.ExecutionSignature, tuple[weights.FfnLayerWeights, weights.FfnLayerWeights]
        ] = {}
        self.materialization_started = False

    def release_capture_resources(self) -> None:
        """Release construction owners after native consumption or confirmed retirement.

        Successful installation or confirmed retirement permits construction
        views to retire. Each sealed Lane keeps its backing and exclusive stream.
        A synchronization/reset error retains remaining owners for safe shutdown.
        """

        for lane in self.capture_lanes:
            lane.release_capture_resources()
        self.capture_weight_pairs.clear()

    def close(self) -> None:
        """Release retained backing only after native retirement has been confirmed."""

        while self.capture_lanes:
            self.capture_lanes[-1].close()
            self.capture_lanes.pop()
        self.capture_weight_pairs.clear()
        self.layer_weights = ()

    def materialize(
        self,
        *,
        fabric_plan: FabricPlan,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        ffnagent_index: int,
    ) -> None:
        """Complete local construction and retained cleanup before readiness.

        Consume one attempt. Every failure retains this Registry's remaining
        owners until native retirement; success preserves borrowed service storage.
        """

        if self.materialization_started:
            raise RuntimeError("FFN execution materialization cannot be retried")
        self.materialization_started = True
        device = torch.cuda.current_device()
        self.construct_execution(
            fabric_plan=fabric_plan,
            model_specs=model_specs,
            ffnagent_index=ffnagent_index,
        )
        # Construction locals have exited; trim only releasable default-domain cache.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    def construct_execution(
        self,
        *,
        fabric_plan: FabricPlan,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        ffnagent_index: int,
    ) -> None:
        """Own the construction-local frame for materialize's consumed attempt."""

        layer_weights = self.layer_weights

        ffnagent_count = sum(placement.role is FabricRole.FFNAGENT for placement in fabric_plan.pe_placements)
        if not 0 <= ffnagent_index < ffnagent_count:
            raise ValueError("FfnAgent index is outside the Fabric Plan")
        if len(model_specs) != len(fabric_plan.model_plans) or len(layer_weights) != len(model_specs):
            raise ValueError("Model Specs, layer weights, and Model Plans are not co-indexed")

        # Validate each local layer against the Plan and collapse
        # equivalent (shape, Capacity, operator) work into shared signatures.
        signatures: list[execution.ExecutionSignature] = []
        representatives: list[weights.FfnLayerWeights] = []
        signature_indices: dict[execution.ExecutionSignature, int] = {}
        native_layers: list[xpool.native.ffnagent.LayerExecutionDescriptor] = []
        for instance_index, (model_plan, instance_plan, model_spec, model_weights) in enumerate(
            zip(fabric_plan.model_plans, fabric_plan.instance_plans, model_specs, layer_weights, strict=True)
        ):
            if len(model_weights) != len(model_plan.layers):
                raise ValueError("Model Spec, layer weights, and Model Plan layers are not co-indexed")
            profile = instance_plan.ffn_profile
            for layer_ordinal, (layer_plan, layer_weights_value) in enumerate(
                zip(model_plan.layers, model_weights, strict=True)
            ):
                local = ffnagent_index in layer_plan.ffnagent_indices
                if local != (layer_weights_value is not None):
                    raise ValueError("local Layer Execution Group and retained weights disagree")
                if not local or layer_weights_value is None:
                    continue
                tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
                layer_layout = preparation.prepare_ffn_layer(
                    model_spec,
                    layer_ordinal=layer_ordinal,
                    payload_dtype=profile.payload_dtype,
                    tp_size=len(layer_plan.ffnagent_indices),
                    tp_rank=tp_rank,
                )
                validate_layer_weights_against_layout(
                    layer_weights=layer_weights_value,
                    layer_layout=layer_layout,
                    device=torch.cuda.current_device(),
                )
                layer_signature_indices = []
                for signature in layer_layout.execution_signatures(profile=profile):
                    signature_index = signature_indices.get(signature)
                    if signature_index is None:
                        signature_index = len(signatures)
                        signature_indices[signature] = signature_index
                        signatures.append(signature)
                        representatives.append(layer_weights_value)
                    layer_signature_indices.append(signature_index)
                native_layers.append(
                    xpool.native.ffnagent.LayerExecutionDescriptor(
                        instance_index=instance_index,
                        layer_ordinal=layer_ordinal,
                        execution_signature_indices=tuple(layer_signature_indices),
                        layer_resource_targets=layer_weights_value.binding_view(),
                    )
                )

        # Lane streams and pools are private; representative weights and
        # Control Capture Probes are shared read-only.
        compute_plans = preparation.prepare_execution_plans(tuple(signatures))
        capture_started_at = monotonic()
        lane_descriptors = []
        estimated_pool_bytes = capture_pool_reservation_bytes(compute_plans)
        capture_weight_pair_keys = tuple(execution.capture_weight_pair_key(signature) for signature in signatures)
        for _ in range(fabric_plan.executor_lane_count):
            lane = LaneCaptureResources()
            self.capture_lanes.append(lane)
            captures: list[
                xpool.native.ffnagent.DenseCaptureDescriptor | xpool.native.ffnagent.MoeCaptureDescriptor
            ] = []
            if signatures:
                lane.initialize()
            for signature, compute_plan, representative, key in zip(
                signatures,
                compute_plans,
                representatives,
                capture_weight_pair_keys,
                strict=True,
            ):
                capture_weight_pair = self.capture_weight_pairs.get(key)
                if capture_weight_pair is None:
                    device = torch.device("cuda", torch.cuda.current_device())
                    probe = (
                        weights.DenseFfnWeights.allocate(
                            i_proj=signature.i_proj,
                            o_proj=signature.o_proj,
                            device=device,
                        )
                        if isinstance(signature, execution.DenseFfnExecutionSignature)
                        else weights.MoeFfnWeights.allocate(
                            i_proj=signature.i_proj,
                            o_proj=signature.o_proj,
                            router=None if signature.router is None else signature.router.projection,
                            correction_bias=None if signature.router is None else signature.router.correction_bias,
                            device=device,
                        )
                    )
                    capture_weight_pair = (representative, probe)
                    self.capture_weight_pairs[key] = capture_weight_pair
                # Admission and the signature-keyed cache preserve the weight kind.
                if isinstance(compute_plan, dense.DenseExecutionPlan):
                    dense_weights, dense_probe = cast(
                        tuple[weights.DenseFfnWeights, weights.DenseFfnWeights], capture_weight_pair
                    )
                    captures.append(
                        capture_dense_signature(
                            compute_plan,
                            dense_weights,
                            dense_probe,
                            owner=lane,
                        )
                    )
                else:
                    moe_weights, moe_probe = cast(
                        tuple[weights.MoeFfnWeights, weights.MoeFfnWeights], capture_weight_pair
                    )
                    captures.append(
                        capture_moe_signature(
                            compute_plan,
                            moe_weights,
                            moe_probe,
                            owner=lane,
                        )
                    )
            if signatures:
                lane.report_reservation(estimated_pool_bytes)
            lane_descriptors.append(xpool.native.ffnagent.LaneCaptureDescriptor(signatures=tuple(captures)))
        logger.info(
            "graph templates captured device=%s signature_count=%s dense_count=%s moe_count=%s "
            "row_capacities=%s elapsed=%.3fs",
            torch.cuda.current_device(),
            len(signatures),
            sum(isinstance(signature, execution.DenseFfnExecutionSignature) for signature in signatures),
            sum(isinstance(signature, execution.MoeFfnExecutionSignature) for signature in signatures),
            sorted({signature.payload_row_capacity for signature in signatures}),
            monotonic() - capture_started_at,
        )

        # Native installation consumes captures; Registry retains the weight
        # owners while Lane Graphs borrow their bindings.
        descriptor = xpool.native.ffnagent.ExecutionInstallDescriptor(
            lanes=tuple(lane_descriptors),
            layers=tuple(native_layers),
        )
        install_started_at = monotonic()
        xpool.native.ffnagent.install_execution(descriptor)
        self.release_capture_resources()
        logger.info(
            "execution installed device=%s layer_count=%s lane_count=%s elapsed=%.3fs",
            torch.cuda.current_device(),
            len(native_layers),
            fabric_plan.executor_lane_count,
            monotonic() - install_started_at,
        )
