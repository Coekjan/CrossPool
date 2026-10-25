"""Generation-scoped FFN Graph Capture and retained execution ownership."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import cast

import torch

import xpool.native
from xpool import ffn
from xpool.fabric import FabricPlan, FabricRole, place_ffn_layer
from xpool.runtime.ffnagent import execution, preparation, weights
from xpool.runtime.ffnagent.operators import dense, linear, moe, routing

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CapturedExecutionSignature:
    """Graph Capture owners retained through atomic native installation."""

    descriptor: xpool.native.ffnagent.DenseCaptureDescriptor | xpool.native.ffnagent.MoeCaptureDescriptor
    primary_graph: torch.cuda.CUDAGraph
    control_graph: torch.cuda.CUDAGraph
    primary_weights: weights.FfnLayerWeights
    control_probe: weights.FfnLayerWeights
    tensors: tuple[torch.Tensor, ...]


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
        layer_weights.gate_up.validate(layer_layout.gate_up)
        layer_weights.down.validate(layer_layout.down)
        projections = (layer_weights.gate_up, layer_weights.down)
        tensors: tuple[torch.Tensor, ...] = ()
    else:
        if not isinstance(layer_weights, weights.MoeFfnWeights):
            raise ValueError("MoE preparation requires MoE resources")
        layer_weights.gate_up.validate(layer_layout.experts.gate_up)
        layer_weights.down.validate(layer_layout.experts.down)
        projections = (layer_weights.gate_up, layer_weights.down)
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


def capture_graph(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Warm and capture one allocation-free ordinary CUDA Graph body."""

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    return graph


def capture_dense_signature(
    signature: execution.DenseFfnExecutionSignature,
    primary_weights: weights.DenseFfnWeights,
    control_probe: weights.DenseFfnWeights,
) -> CapturedExecutionSignature:
    """Capture primary/control Dense bodies sharing lane placeholders."""

    hidden_states = torch.zeros(
        (signature.payload_row_capacity, signature.hidden_size),
        dtype=signature.payload_dtype,
        device="cuda",
    )
    partial = torch.empty_like(hidden_states)
    workspace = torch.empty(preparation.compute_workspace_bytes(signature), dtype=torch.uint8, device="cuda")

    def launch(layer_weights: weights.DenseFfnWeights) -> None:
        dense.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=workspace,
            output=partial,
            activation=signature.activation,
        )

    primary_graph = capture_graph(lambda: launch(primary_weights))
    control_graph = capture_graph(lambda: launch(control_probe))
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
        capture_workspace=workspace,
        compute_workspace_bytes=workspace.numel(),
        primary_capture_resources=primary_resources,
        control_capture_resources=control_resources,
    )
    return CapturedExecutionSignature(
        descriptor=descriptor,
        primary_graph=primary_graph,
        control_graph=control_graph,
        primary_weights=primary_weights,
        control_probe=control_probe,
        tensors=(hidden_states, partial, workspace),
    )


def capture_moe_signature(
    signature: execution.MoeFfnExecutionSignature,
    primary_weights: weights.MoeFfnWeights,
    control_probe: weights.MoeFfnWeights,
) -> CapturedExecutionSignature:
    """Capture primary/control MoE bodies sharing lane placeholders."""

    hidden_states = torch.zeros(
        (signature.payload_row_capacity, signature.hidden_size),
        dtype=signature.payload_dtype,
        device="cuda",
    )
    partial = torch.empty_like(hidden_states)
    payload_rows = (
        torch.full((1,), signature.payload_row_capacity, dtype=torch.int64, device="cuda")
        if signature.router is not None
        else None
    )
    routing_storage = torch.empty(
        signature.payload_row_capacity * signature.effective_topk * 8,
        dtype=torch.uint8,
        device="cuda",
    )
    route_elements = signature.payload_row_capacity * signature.effective_topk
    topk_ids = (
        routing_storage[: route_elements * 4]
        .view(torch.int32)
        .view(signature.payload_row_capacity, signature.effective_topk)
    )
    topk_weights = (
        routing_storage[route_elements * 4 :]
        .view(torch.float32)
        .view(signature.payload_row_capacity, signature.effective_topk)
    )
    topk_ids.copy_(
        torch.arange(signature.effective_topk, dtype=torch.int32, device="cuda")
        .remainder(signature.expert_count)
        .expand_as(topk_ids)
    )
    topk_weights.fill_(1.0 / signature.effective_topk)
    workspace_layout = preparation.MoeWorkspaceLayout.from_signature(signature)
    workspace = torch.empty(workspace_layout.nbytes, dtype=torch.uint8, device="cuda")
    operation_type = moe.expert_operation_type(signature.operation)
    operation = operation_type(signature, workspace[workspace_layout.experts])
    router_workspace = None
    routed_ids = None
    routed_weights = None
    projection_operation = linear.LinearOperationKind.TORCH
    router_signature = signature.router
    router_layout = workspace_layout.router
    if router_signature is not None and router_layout is not None:
        routed_topk = signature.effective_topk - (signature.expert_count - router_signature.routed_expert_count)
        routed_shape = (signature.payload_row_capacity, routed_topk)
        router_workspace = workspace[router_layout.projection]
        routed_ids = workspace[router_layout.ids].view(torch.int32).view(routed_shape)
        routed_weights = workspace[router_layout.weights].view(torch.float32).view(routed_shape)
        projection_operation = linear.select_router_projection_operation(
            router_signature,
            payload_dtype=signature.payload_dtype,
            row_capacity=signature.payload_row_capacity,
            device=hidden_states.device,
        )

    def launch(layer_weights: weights.MoeFfnWeights) -> None:
        router_signature = signature.router
        if router_signature is not None:
            router_weights = layer_weights.router
            if (
                payload_rows is None
                or router_weights is None
                or router_workspace is None
                or routed_ids is None
                or routed_weights is None
            ):
                raise AssertionError("Router-owner capture lacks Router resources")
            routing.compute_routed_topk(
                router_signature.algorithm,
                router_signature.projection_precision,
                projection_operation=projection_operation,
                weight_rounding=router_signature.weight_rounding,
                hidden_states=hidden_states,
                router_weights=router_weights,
                workspace=router_workspace,
                routed_ids=routed_ids,
                routed_weights=routed_weights,
                renormalize=router_signature.renormalize,
            )
            routing.finalize_moe_routing(
                routed_ids=routed_ids,
                routed_weights=routed_weights,
                final_ids=topk_ids,
                final_weights=topk_weights,
                payload_rows=payload_rows,
                routed_expert_count=router_signature.routed_expert_count,
                shared_expert_count=signature.expert_count - router_signature.routed_expert_count,
                routed_scaling_factor=signature.routed_scaling_factor,
            )
        operation.run(
            hidden_states=hidden_states,
            gate_up=layer_weights.gate_up,
            down=layer_weights.down,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=partial,
        )

    primary_graph = capture_graph(lambda: launch(primary_weights))
    control_graph = capture_graph(lambda: launch(control_probe))
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
        capture_workspace=workspace,
        compute_workspace_bytes=workspace.numel(),
        capture_routing_metadata=routing_storage,
        capture_payload_rows=payload_rows,
        primary_capture_resources=primary_resources,
        control_capture_resources=control_resources,
    )
    return CapturedExecutionSignature(
        descriptor=descriptor,
        primary_graph=primary_graph,
        control_graph=control_graph,
        primary_weights=primary_weights,
        control_probe=control_probe,
        tensors=(
            (hidden_states, partial, routing_storage, workspace)
            if payload_rows is None
            else (hidden_states, partial, routing_storage, workspace, payload_rows)
        ),
    )


class FfnExecutionRegistry:
    """Retain service weight owners referenced by installed Lane GraphExecs."""

    def __init__(
        self,
        *,
        layer_weights: tuple[tuple[weights.FfnLayerWeights | None, ...], ...],
    ) -> None:
        """Retain the Plan-selected service weight owners."""

        self.layer_weights = layer_weights

    @classmethod
    def materialize(
        cls,
        *,
        fabric_plan: FabricPlan,
        model_specs: tuple[ffn.FfnModelSpec, ...],
        ffnagent_index: int,
        layer_weights: tuple[tuple[weights.FfnLayerWeights | None, ...], ...],
    ) -> FfnExecutionRegistry:
        """Capture, install, and retain the Plan-selected local execution.

        Equal execution signatures share one captured compute body. Capture
        resources remain alive through synchronous native installation; the
        returned Registry retains the service weights borrowed by Lane GraphExecs.
        """

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
            if model_plan.model_spec_digest != model_spec.digest():
                raise ValueError("Model Plan digest disagrees with the co-indexed Model Spec")
            if len(model_weights) != len(model_plan.layers) or len(model_spec.layers) != len(model_plan.layers):
                raise ValueError("Model Spec, layer weights, and Model Plan layers are not co-indexed")
            profile = instance_plan.ffn_profile
            if profile.hidden_size != model_spec.hidden_size or tuple(
                (layer.layer_id, layer.kind) for layer in profile.layers
            ) != tuple((layer.layer_id, layer.kind) for layer in model_spec.layers):
                raise ValueError("Instance Profile disagrees with the co-indexed Model Spec")
            for layer_ordinal, (layer_plan, layer_spec, layer_weights_value) in enumerate(
                zip(model_plan.layers, model_spec.layers, model_weights, strict=True)
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
                if layer_plan != place_ffn_layer(
                    layer_spec,
                    layer_plan.ffnagent_indices,
                    local_intermediate_size=layer_layout.local_intermediate_size,
                ):
                    raise ValueError("Layer Plan geometry disagrees with the Model Spec")
                validate_layer_weights_against_layout(
                    layer_weights=layer_weights_value,
                    layer_layout=layer_layout,
                    device=torch.cuda.current_device(),
                )
                layer_signature_indices = []
                for signature in preparation.required_execution_signatures(
                    layer_layout=layer_layout,
                    profile=profile,
                ):
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

        # One Primary/Control pair discovers weight bindings
        # for every compatible signature while retaining representative owners.
        capture_started_at = monotonic()
        captures = []
        capture_weight_pairs: dict[
            execution.ExecutionSignature,
            tuple[weights.FfnLayerWeights, weights.FfnLayerWeights],
        ] = {}
        for signature, representative in zip(signatures, representatives, strict=True):
            key = execution.capture_weight_pair_key(signature)
            capture_weight_pair = capture_weight_pairs.get(key)
            if capture_weight_pair is None:
                device = torch.device("cuda", torch.cuda.current_device())
                probe = (
                    weights.DenseFfnWeights.allocate(gate_up=signature.gate_up, down=signature.down, device=device)
                    if isinstance(signature, execution.DenseFfnExecutionSignature)
                    else weights.MoeFfnWeights.allocate(
                        gate_up=signature.gate_up,
                        down=signature.down,
                        router=None if signature.router is None else signature.router.projection,
                        correction_bias=None if signature.router is None else signature.router.correction_bias,
                        device=device,
                    )
                )
                capture_weight_pair = (representative, probe)
                capture_weight_pairs[key] = capture_weight_pair
            # Admission and the signature-keyed cache preserve the weight kind.
            if isinstance(signature, execution.DenseFfnExecutionSignature):
                dense_weights, dense_probe = cast(
                    tuple[weights.DenseFfnWeights, weights.DenseFfnWeights], capture_weight_pair
                )
                captures.append(capture_dense_signature(signature, dense_weights, dense_probe))
            else:
                moe_weights, moe_probe = cast(tuple[weights.MoeFfnWeights, weights.MoeFfnWeights], capture_weight_pair)
                captures.append(capture_moe_signature(signature, moe_weights, moe_probe))
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

        # Native installation synchronously consumes captures and Lane Graphs
        # retain the selected service weight storage.
        descriptor = xpool.native.ffnagent.ExecutionInstallDescriptor(
            signatures=tuple(capture.descriptor for capture in captures),
            layers=tuple(native_layers),
        )
        install_started_at = monotonic()
        xpool.native.ffnagent.install_execution(descriptor)
        logger.info(
            "execution installed device=%s layer_count=%s lane_count=%s elapsed=%.3fs",
            torch.cuda.current_device(),
            len(native_layers),
            fabric_plan.executor_lane_count,
            monotonic() - install_started_at,
        )
        return cls(layer_weights=layer_weights)
