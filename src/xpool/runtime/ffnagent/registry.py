"""Generation-scoped FFN Graph Capture and retained execution ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import cast

import torch

import xpool.native
from xpool import ffn
from xpool.fabric import DenseFfnLayerPlan, FabricPlan, FabricRole, MoeFfnLayerPlan
from xpool.runtime.ffnagent import architecture, execution, operators, weights


@dataclass(slots=True)
class CapturedExecutionSignature:
    """Graph Capture owners retained through atomic native installation."""

    projection: (
        xpool.native.ffnagent.DenseExecutionSignatureProjection | xpool.native.ffnagent.MoeExecutionSignatureProjection
    )
    primary_graph: torch.cuda.CUDAGraph
    control_graph: torch.cuda.CUDAGraph
    primary_weights: weights.FfnLayerWeights
    control_probe: weights.FfnLayerWeights
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class MoeGraphCaptureWorkspace:
    """Typed views over one contiguous Graph Capture workspace allocation."""

    allocation: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    cumsum_buffer: torch.Tensor
    activated: torch.Tensor
    gate_up: torch.Tensor
    route_outputs: torch.Tensor
    router_workspace: torch.Tensor | None
    routed_ids: torch.Tensor | None
    routed_weights: torch.Tensor | None


def binding_resource_projection(
    layer_weights: weights.FfnLayerWeights,
) -> xpool.native.ffnagent.DenseBindingResourceProjection | xpool.native.ffnagent.MoeBindingResourceProjection:
    """Project retained weight addresses into the closed native resource union."""

    if isinstance(layer_weights, weights.DenseFfnWeights):
        return xpool.native.ffnagent.DenseBindingResourceProjection(
            gate_up_weight_address=layer_weights.gate_up_weight.data_ptr(),
            down_weight_address=layer_weights.down_weight.data_ptr(),
        )
    router = layer_weights.router
    return xpool.native.ffnagent.MoeBindingResourceProjection(
        expert_gate_up_weight_address=layer_weights.expert_gate_up_weight.data_ptr(),
        expert_down_weight_address=layer_weights.expert_down_weight.data_ptr(),
        router=(
            None
            if router is None
            else xpool.native.ffnagent.MoeRouterBindingResourceProjection(
                weight_address=router.weight.data_ptr(),
                correction_bias_address=(None if router.correction_bias is None else router.correction_bias.data_ptr()),
            )
        ),
    )


def create_control_capture_probe(layer_weights: weights.FfnLayerWeights) -> weights.FfnLayerWeights:
    """Allocate one zero-valued shape-equivalent Control Capture Probe."""

    if isinstance(layer_weights, weights.DenseFfnWeights):
        return weights.DenseFfnWeights(
            gate_up_weight=torch.zeros_like(layer_weights.gate_up_weight),
            down_weight=torch.zeros_like(layer_weights.down_weight),
        )
    router = layer_weights.router
    return weights.MoeFfnWeights(
        expert_gate_up_weight=torch.zeros_like(layer_weights.expert_gate_up_weight),
        expert_down_weight=torch.zeros_like(layer_weights.expert_down_weight),
        router=(
            None
            if router is None
            else weights.MoeRouterWeights(
                weight=torch.zeros_like(router.weight),
                correction_bias=(None if router.correction_bias is None else torch.zeros_like(router.correction_bias)),
            )
        ),
    )


def validate_layer_weights_against_plan(
    *,
    layer_weights: weights.FfnLayerWeights,
    layer_plan: DenseFfnLayerPlan | MoeFfnLayerPlan,
    layer_spec: ffn.FfnLayerSpec,
    hidden_size: int,
    payload_dtype: torch.dtype,
    tp_rank: int,
    cuda_device: int,
) -> None:
    """Require one retained weight owner to match its exact local Plan role."""

    if isinstance(layer_plan, DenseFfnLayerPlan):
        if not isinstance(layer_spec, ffn.DenseFfnSpec) or not isinstance(layer_weights, weights.DenseFfnWeights):
            raise ValueError("Dense Layer Plan requires Dense Model Spec and weights")
        if layer_weights.gate_up_weight.shape != (2 * layer_plan.local_intermediate_size, hidden_size):
            raise ValueError("Dense gate/up weights disagree with the Layer Plan")
        if layer_weights.down_weight.shape != (hidden_size, layer_plan.local_intermediate_size):
            raise ValueError("Dense down weights disagree with the Layer Plan")
        tensors = (layer_weights.gate_up_weight, layer_weights.down_weight)
        if layer_weights.gate_up_weight.dtype is not payload_dtype:
            raise ValueError("Dense weights disagree with the Instance payload dtype")
    else:
        if not isinstance(layer_spec, ffn.MoeFfnSpec) or not isinstance(layer_weights, weights.MoeFfnWeights):
            raise ValueError("MoE Layer Plan requires MoE Model Spec and weights")
        expert_count = layer_spec.routed_expert_count + layer_spec.shared_expert_count
        if layer_plan.effective_topk != layer_spec.routed_topk + layer_spec.shared_expert_count:
            raise ValueError("MoE Layer Plan routing width disagrees with the Model Spec")
        expected_gate_up = (expert_count, 2 * layer_plan.local_intermediate_size, hidden_size)
        expected_down = (expert_count, hidden_size, layer_plan.local_intermediate_size)
        if layer_weights.expert_gate_up_weight.shape != expected_gate_up:
            raise ValueError("MoE gate/up weights disagree with the Layer Plan")
        if layer_weights.expert_down_weight.shape != expected_down:
            raise ValueError("MoE down weights disagree with the Layer Plan")
        router = layer_weights.router
        if (tp_rank == 0) != (router is not None):
            raise ValueError("MoE Router ownership does not match TP rank zero")
        tensors = (layer_weights.expert_gate_up_weight, layer_weights.expert_down_weight)
        if layer_weights.expert_gate_up_weight.dtype is not payload_dtype:
            raise ValueError("MoE weights disagree with the Instance payload dtype")
        if router is not None:
            if router.weight.shape != (layer_spec.routed_expert_count, hidden_size):
                raise ValueError("MoE Router weights disagree with the Layer Plan")
            correction_required = layer_spec.checkpoint.router_correction_bias_key is not None
            if correction_required != (router.correction_bias is not None):
                raise ValueError("MoE correction-bias ownership disagrees with the routing formula")
            tensors += (router.weight,)
            if router.correction_bias is not None:
                tensors += (router.correction_bias,)

    if any(tensor.device.index != cuda_device for tensor in tensors):
        raise ValueError("local FFN weights must reside on the current CUDA device")


def capture_graph(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Warm and capture one allocation-free ordinary CUDA Graph body."""

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    return graph


def allocate_moe_graph_capture_workspace(
    *,
    signature: execution.MoeFfnExecutionSignature,
    w13_config: dict[str, int],
) -> MoeGraphCaptureWorkspace:
    """Allocate exact typed views in one contiguous MoE Graph Capture workspace."""

    route_count = signature.payload_row_capacity * signature.effective_topk
    region_specs, offsets, selected_extent = execution.moe_workspace_layout(
        signature,
        block_size_m=w13_config["BLOCK_SIZE_M"],
    )
    allocation_extent = execution.compute_workspace_bytes(signature)
    if selected_extent > allocation_extent:
        raise AssertionError("selected MoE workspace layout exceeds its fixed allocation")
    allocation = torch.empty(allocation_extent, dtype=torch.uint8, device="cuda")

    views = tuple(
        allocation[offset : offset + prod(shape) * dtype.itemsize].view(dtype).view(shape)
        for offset, (dtype, shape) in zip(offsets, region_specs, strict=True)
    )
    overlap = views[5]
    gate_up_elements = route_count * 2 * signature.local_intermediate_size
    route_output_elements = route_count * signature.hidden_size
    gate_up = (
        overlap[: gate_up_elements * signature.payload_dtype.itemsize]
        .view(signature.payload_dtype)
        .view(route_count, 2 * signature.local_intermediate_size)
    )
    route_outputs = (
        overlap[: route_output_elements * signature.payload_dtype.itemsize]
        .view(signature.payload_dtype)
        .view(
            signature.payload_row_capacity,
            signature.effective_topk,
            signature.hidden_size,
        )
    )
    router_views = views[6:]
    return MoeGraphCaptureWorkspace(
        allocation=allocation,
        sorted_token_ids=views[0],
        expert_ids=views[1],
        num_tokens_post_padded=views[2],
        cumsum_buffer=views[3],
        activated=views[4],
        gate_up=gate_up,
        route_outputs=route_outputs,
        router_workspace=None if not router_views else router_views[0],
        routed_ids=None if not router_views else router_views[1],
        routed_weights=None if not router_views else router_views[2],
    )


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
    workspace = torch.empty(execution.compute_workspace_bytes(signature), dtype=torch.uint8, device="cuda")

    def launch(layer_weights: weights.DenseFfnWeights) -> None:
        operators.compute_dense_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            workspace=workspace,
            output=partial,
            activation=signature.activation,
        )

    primary_graph = capture_graph(lambda: launch(primary_weights))
    control_graph = capture_graph(lambda: launch(control_probe))
    primary_resources = cast(
        xpool.native.ffnagent.DenseBindingResourceProjection,
        binding_resource_projection(primary_weights),
    )
    control_resources = cast(
        xpool.native.ffnagent.DenseBindingResourceProjection,
        binding_resource_projection(control_probe),
    )
    projection = xpool.native.ffnagent.DenseExecutionSignatureProjection(
        payload_dtype=signature.payload_dtype,
        payload_row_capacity=signature.payload_row_capacity,
        hidden_size=signature.hidden_size,
        local_intermediate_size=signature.local_intermediate_size,
        primary_graph_address=primary_graph.raw_cuda_graph(),
        control_graph_address=control_graph.raw_cuda_graph(),
        capture_input_address=hidden_states.data_ptr(),
        capture_partial_address=partial.data_ptr(),
        capture_workspace_address=workspace.data_ptr(),
        compute_workspace_bytes=workspace.numel(),
        primary_capture_resources=primary_resources,
        control_capture_resources=control_resources,
    )
    return CapturedExecutionSignature(
        projection=projection,
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

    w13_config, w2_config = operators.select_moe_kernel_configs(
        layer_weights=primary_weights,
        row_capacity=signature.payload_row_capacity,
        effective_topk=signature.effective_topk,
    )
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
    workspace = allocate_moe_graph_capture_workspace(signature=signature, w13_config=w13_config)

    def launch(layer_weights: weights.MoeFfnWeights) -> None:
        router_signature = signature.router
        if router_signature is not None:
            router_weights = layer_weights.router
            if (
                payload_rows is None
                or router_weights is None
                or workspace.router_workspace is None
                or workspace.routed_ids is None
                or workspace.routed_weights is None
            ):
                raise AssertionError("Router-owner capture lacks Router resources")
            router_signature.compute_routed_topk(
                hidden_states=hidden_states,
                router_weights=router_weights,
                workspace=workspace.router_workspace,
                routed_ids=workspace.routed_ids,
                routed_weights=workspace.routed_weights,
                renormalize=router_signature.renormalize,
            )
            operators.finalize_moe_routing(
                routed_ids=workspace.routed_ids,
                routed_weights=workspace.routed_weights,
                final_ids=topk_ids,
                final_weights=topk_weights,
                payload_rows=payload_rows,
                routed_expert_count=router_signature.routed_expert_count,
                shared_expert_count=signature.expert_count - router_signature.routed_expert_count,
                routed_scaling_factor=signature.routed_scaling_factor,
            )
        operators.compute_moe_partial(
            hidden_states=hidden_states,
            layer_weights=layer_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            sorted_token_ids=workspace.sorted_token_ids,
            expert_ids=workspace.expert_ids,
            num_tokens_post_padded=workspace.num_tokens_post_padded,
            cumsum_buffer=workspace.cumsum_buffer,
            gate_up=workspace.gate_up,
            activated=workspace.activated,
            route_outputs=workspace.route_outputs,
            output=partial,
            w13_config=w13_config,
            w2_config=w2_config,
            activation=signature.activation,
            routed_scaling_factor=signature.routed_scaling_factor,
        )

    primary_graph = capture_graph(lambda: launch(primary_weights))
    control_graph = capture_graph(lambda: launch(control_probe))
    primary_resources = cast(
        xpool.native.ffnagent.MoeBindingResourceProjection,
        binding_resource_projection(primary_weights),
    )
    control_resources = cast(
        xpool.native.ffnagent.MoeBindingResourceProjection,
        binding_resource_projection(control_probe),
    )
    projection = xpool.native.ffnagent.MoeExecutionSignatureProjection(
        payload_dtype=signature.payload_dtype,
        payload_row_capacity=signature.payload_row_capacity,
        hidden_size=signature.hidden_size,
        local_intermediate_size=signature.local_intermediate_size,
        expert_count=signature.expert_count,
        effective_topk=signature.effective_topk,
        routed_expert_count=(None if signature.router is None else signature.router.routed_expert_count),
        primary_graph_address=primary_graph.raw_cuda_graph(),
        control_graph_address=control_graph.raw_cuda_graph(),
        capture_input_address=hidden_states.data_ptr(),
        capture_partial_address=partial.data_ptr(),
        capture_workspace_address=workspace.allocation.data_ptr(),
        compute_workspace_bytes=workspace.allocation.numel(),
        capture_routing_metadata_address=routing_storage.data_ptr(),
        capture_payload_rows_address=None if payload_rows is None else payload_rows.data_ptr(),
        primary_capture_resources=primary_resources,
        control_capture_resources=control_resources,
    )
    return CapturedExecutionSignature(
        projection=projection,
        primary_graph=primary_graph,
        control_graph=control_graph,
        primary_weights=primary_weights,
        control_probe=control_probe,
        tensors=(
            (hidden_states, partial, routing_storage, workspace.allocation)
            if payload_rows is None
            else (hidden_states, partial, routing_storage, workspace.allocation, payload_rows)
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
        native_layers: list[xpool.native.ffnagent.LayerExecutionProjection] = []
        for instance_index, (model_plan, instance_plan, model_spec, model_weights) in enumerate(
            zip(fabric_plan.model_plans, fabric_plan.instance_plans, model_specs, layer_weights, strict=True)
        ):
            if model_plan.model_spec_digest != model_spec.digest():
                raise ValueError("Model Plan digest disagrees with the co-indexed Model Spec")
            if len(model_weights) != len(model_plan.layers) or len(model_spec.layers) != len(model_plan.layers):
                raise ValueError("Model Spec, layer weights, and Model Plan layers are not co-indexed")
            profile = instance_plan.ffn_profile
            if (
                profile.model_config_digest != model_spec.model_config_digest
                or profile.hidden_size != model_spec.hidden_size
                or tuple((layer.layer_id, layer.kind) for layer in profile.layers)
                != tuple((layer.layer_id, layer.kind) for layer in model_spec.layers)
            ):
                raise ValueError("Instance Profile disagrees with the co-indexed Model Spec")
            capacities = execution.derive_payload_row_capacities(
                max(profile.decode_payload_row_capacity, profile.prefill_payload_row_capacity)
            )
            for layer_ordinal, (layer_plan, layer_spec, layer_weights_value) in enumerate(
                zip(model_plan.layers, model_spec.layers, model_weights, strict=True)
            ):
                local = ffnagent_index in layer_plan.ffnagent_indices
                if local != (layer_weights_value is not None):
                    raise ValueError("local Layer Execution Group and retained weights disagree")
                if not local or layer_weights_value is None:
                    continue
                tp_rank = layer_plan.ffnagent_indices.index(ffnagent_index)
                validate_layer_weights_against_plan(
                    layer_weights=layer_weights_value,
                    layer_plan=layer_plan,
                    layer_spec=layer_spec,
                    hidden_size=profile.hidden_size,
                    payload_dtype=profile.payload_dtype,
                    tp_rank=tp_rank,
                    cuda_device=torch.cuda.current_device(),
                )
                layer_signature_indices = []
                for capacity in capacities:
                    if isinstance(layer_plan, DenseFfnLayerPlan):
                        if not isinstance(layer_weights_value, weights.DenseFfnWeights):
                            raise AssertionError("validated Dense Layer Plan changed weight type")
                        signature: execution.ExecutionSignature = execution.DenseFfnExecutionSignature(
                            payload_dtype=profile.payload_dtype,
                            payload_row_capacity=capacity,
                            hidden_size=profile.hidden_size,
                            local_intermediate_size=layer_plan.local_intermediate_size,
                            activation=model_spec.activation,
                        )
                    else:
                        if (
                            not isinstance(layer_plan, MoeFfnLayerPlan)
                            or not isinstance(layer_spec, ffn.MoeFfnSpec)
                            or not isinstance(layer_weights_value, weights.MoeFfnWeights)
                        ):
                            raise ValueError("MoE Layer Plan requires MoE Model Spec and weights")
                        model_adapter = architecture.adapter_for(model_spec)
                        if not issubclass(model_adapter, architecture.MoeFfnModelAdapter):
                            raise ValueError("MoE Model Spec requires a MoE FFN Model Adapter")
                        router = None
                        if tp_rank == 0:
                            router = execution.MoeRouterExecutionSignature(
                                compute_routed_topk=model_adapter.compute_routed_topk,
                                routed_expert_count=layer_spec.routed_expert_count,
                                router_workspace_bytes=model_adapter.router_workspace_bytes(
                                    payload_dtype=profile.payload_dtype,
                                    payload_row_capacity=capacity,
                                    routed_expert_count=layer_spec.routed_expert_count,
                                    routed_topk=layer_spec.routed_topk,
                                ),
                                correction_bias_present=(layer_spec.checkpoint.router_correction_bias_key is not None),
                                renormalize=layer_spec.renormalize,
                            )
                        signature = execution.MoeFfnExecutionSignature(
                            payload_dtype=profile.payload_dtype,
                            payload_row_capacity=capacity,
                            hidden_size=profile.hidden_size,
                            local_intermediate_size=layer_plan.local_intermediate_size,
                            expert_count=layer_spec.routed_expert_count + layer_spec.shared_expert_count,
                            effective_topk=layer_plan.effective_topk,
                            activation=model_spec.activation,
                            routed_scaling_factor=layer_spec.routed_scaling_factor,
                            router=router,
                        )
                    signature_index = signature_indices.get(signature)
                    if signature_index is None:
                        signature_index = len(signatures)
                        signature_indices[signature] = signature_index
                        signatures.append(signature)
                        representatives.append(layer_weights_value)
                    layer_signature_indices.append(signature_index)
                native_layers.append(
                    xpool.native.ffnagent.LayerExecutionProjection(
                        instance_index=instance_index,
                        layer_ordinal=layer_ordinal,
                        execution_signature_indices=tuple(layer_signature_indices),
                        layer_resource_targets=binding_resource_projection(layer_weights_value),
                    )
                )

        # One Primary/Control pair discovers weight bindings
        # for every compatible signature while retaining representative owners.
        captures = []
        capture_weight_pairs: dict[
            execution.ExecutionSignature,
            tuple[weights.FfnLayerWeights, weights.FfnLayerWeights],
        ] = {}
        for signature, representative in zip(signatures, representatives, strict=True):
            key = execution.capture_weight_pair_key(signature)
            capture_weight_pair = capture_weight_pairs.get(key)
            if capture_weight_pair is None:
                capture_weight_pair = (representative, create_control_capture_probe(representative))
                capture_weight_pairs[key] = capture_weight_pair
            primary_weights, control_probe = capture_weight_pair
            if (
                isinstance(signature, execution.DenseFfnExecutionSignature)
                and isinstance(primary_weights, weights.DenseFfnWeights)
                and isinstance(control_probe, weights.DenseFfnWeights)
            ):
                captures.append(capture_dense_signature(signature, primary_weights, control_probe))
            elif (
                isinstance(signature, execution.MoeFfnExecutionSignature)
                and isinstance(primary_weights, weights.MoeFfnWeights)
                and isinstance(control_probe, weights.MoeFfnWeights)
            ):
                captures.append(capture_moe_signature(signature, primary_weights, control_probe))
            else:
                raise AssertionError("Execution Signature and representative weights disagree")

        # Native installation synchronously consumes captures and Lane Graphs
        # retain the selected service weight storage.
        projection = xpool.native.ffnagent.ExecutionProjection(
            signatures=tuple(capture.projection for capture in captures),
            layers=tuple(native_layers),
        )
        xpool.native.ffnagent.install_execution(projection)
        return cls(layer_weights=layer_weights)
