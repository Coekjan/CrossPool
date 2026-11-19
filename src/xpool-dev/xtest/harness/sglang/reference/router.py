"""Isolated prepared-Router reproduction with a stage-local CPU reference."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

from xkit.child import PythonChildProcess
from xpool.ffn import FfnModelSpec, MoeFfnSpec
from xpool.runtime.ffnagent import checkpoint
from xtest.harness.support.ffn import FfnRouterComparison, compare_ffn_router
from xtest.harness.support.wait import remaining_seconds


@dataclass(frozen=True, slots=True)
class RouterReferenceCase:
    """One CPU input and installed route at its observed execution Capacity."""

    case_id: str
    layer_id: int
    hidden_states: torch.Tensor
    execution_capacity: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    def __post_init__(self) -> None:
        hidden, ids, weights = self.hidden_states, self.topk_ids, self.topk_weights
        if (
            not self.case_id
            or self.layer_id < 0
            or hidden.ndim != 2
            or hidden.dtype not in (torch.bfloat16, torch.float16)
            or hidden.shape[0] <= 0
            or self.execution_capacity < hidden.shape[0]
            or ids.ndim != 2
            or ids.shape[0] != hidden.shape[0]
            or weights.shape != ids.shape
            or ids.dtype != torch.int32
            or weights.dtype != torch.float32
            or any(value.device.type != "cpu" for value in (hidden, ids, weights))
            or not torch.isfinite(hidden).all()
            or not torch.isfinite(weights).all()
        ):
            raise ValueError("Router reference requires CPU hidden states, matching routes and a covering Capacity")


@dataclass(frozen=True, slots=True)
class RouterReferenceJob:
    model_spec: FfnModelSpec
    model_path: Path
    tensor_parallel_size: int
    cases: tuple[RouterReferenceCase, ...]
    device: int
    workdir: Path


@dataclass(frozen=True, slots=True)
class RouterReferenceCompleted:
    comparisons: dict[str, FfnRouterComparison]


def run_router_reference(
    *,
    model_spec: FfnModelSpec,
    model_path: Path,
    tensor_parallel_size: int,
    cases: Sequence[RouterReferenceCase],
    device: int,
    workdir: Path,
    timeout_seconds: float,
) -> dict[str, FfnRouterComparison]:
    """Run one Router batch and require child retirement before returning.

    The fresh process contains CrossPool's process-global GEMM constraints;
    the later original-SGLang reference never imports those prepared operators.
    Raw Router tensors and comparisons remain under ``workdir``.
    """

    cases = tuple(cases)
    if timeout_seconds <= 0 or tensor_parallel_size <= 0 or device < 0 or not cases:
        raise ValueError("Router reference requires cases and positive timeout/TP with a nonnegative device")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Router reference case IDs must be unique")
    workdir.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout_seconds
    process = PythonChildProcess(
        "router-reference",
        run_router_reference_child,
        RouterReferenceJob(model_spec, model_path, tensor_parallel_size, cases, device, workdir),
        log_path=workdir / "child.log",
    )
    try:
        process.start()
        completed = process.receive(
            RouterReferenceCompleted, timeout_seconds=remaining_seconds(deadline, "Router comparison")
        )
        process.wait(timeout_seconds=remaining_seconds(deadline, "Router child exit"))
        if set(completed.comparisons) != {case.case_id for case in cases}:
            raise RuntimeError("Router reference did not return every requested case")
        return completed.comparisons
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()


def run_router_reference_child(connection: Connection, job: RouterReferenceJob) -> None:
    torch.cuda.set_device(job.device)
    key_view = checkpoint.read_checkpoint_key_view(job.model_path)
    layer_ordinal_by_id = {layer.layer_id: ordinal for ordinal, layer in enumerate(job.model_spec.layers)}
    comparisons = {}
    for index, case in enumerate(job.cases):
        layer_ordinal = layer_ordinal_by_id[case.layer_id]
        layer = job.model_spec.layers[layer_ordinal]
        if not isinstance(layer, MoeFfnSpec):
            raise ValueError("Router reference requires an MoE layer")
        source_keys = layer.router.tensor_keys()
        if layer.router_correction_bias_key is not None:
            source_keys += (layer.router_correction_bias_key,)
        source = {}
        for key in source_keys:
            with safe_open(str(key_view[key]), framework="pt", device="cpu") as reader:
                source[key] = reader.get_tensor(key)
        destination = job.workdir / f"case-{index:04d}"
        destination.mkdir()
        comparisons[case.case_id] = evaluate_router_case(
            model_spec=job.model_spec,
            layer_ordinal=layer_ordinal,
            tensor_parallel_size=job.tensor_parallel_size,
            case=case,
            source=source,
            destination=destination,
        )
    connection.send(RouterReferenceCompleted(comparisons))


def evaluate_router_case(
    *,
    model_spec: FfnModelSpec,
    layer_ordinal: int,
    tensor_parallel_size: int,
    case: RouterReferenceCase,
    source: dict[str, torch.Tensor],
    destination: Path,
) -> FfnRouterComparison:
    # Prepared GEMM installs process-global library policy only in this child.
    from xpool.quantization import TensorLayout
    from xpool.runtime.ffnagent import execution, preparation, weights
    from xpool.runtime.ffnagent.operators import routing

    layer = model_spec.layers[layer_ordinal]
    assert isinstance(layer, MoeFfnSpec)
    layout = preparation.prepare_ffn_layer(
        model_spec,
        layer_ordinal=layer_ordinal,
        payload_dtype=case.hidden_states.dtype,
        tp_size=tensor_parallel_size,
        tp_rank=0,
    )
    assert isinstance(layout, preparation.MoeFfnLayerLayout) and layout.router is not None
    if layer.router.scales_key is not None:
        raise ValueError("Router reference requires the supported unquantized projection")
    bias_round_dtype = layer.router_projection_precision.bias_round_dtype(payload_dtype=case.hidden_states.dtype)
    projection = weights.ProjectionWeights(
        values=source[layer.router.values_key].to(device="cuda", dtype=layout.router.values.dtype),
        scales=None,
        bias=(
            source[layer.router.bias_key].to(dtype=bias_round_dtype).to(device="cuda", dtype=layout.router.bias.dtype)
            if layer.router.bias_key is not None and layout.router.bias is not None
            else None
        ),
    )
    correction = (
        source[layer.router_correction_bias_key].to(device="cuda", dtype=torch.float32)
        if layer.router_correction_bias_key is not None
        else None
    )
    router_weights = weights.MoeRouterWeights(projection, correction)
    signature = execution.MoeRouterExecutionSignature(
        algorithm=layer.router_algorithm,
        projection_precision=layer.router_projection_precision,
        weight_rounding=layer.topk_weight_rounding,
        projection=layout.router,
        routed_expert_count=layer.routed_expert_count,
        correction_bias=layout.router_correction_bias,
        renormalize=layer.renormalize,
    )
    capacity, rows = case.execution_capacity, case.hidden_states.shape[0]
    plan = routing.RouterExecutionPlan.prepare(
        signature,
        input_layout=TensorLayout((capacity, model_spec.hidden_size), case.hidden_states.dtype, 0),
        routed_topk=layer.routed_topk,
    )
    operation = routing.Router(plan, torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8))
    hidden = torch.zeros((capacity, model_spec.hidden_size), device="cuda", dtype=case.hidden_states.dtype)
    hidden[:rows].copy_(case.hidden_states)
    live_rows = torch.tensor([rows], device="cuda", dtype=torch.int64)
    final_ids = torch.empty((capacity, layer.routed_topk + layer.shared_expert_count), device="cuda", dtype=torch.int32)
    final_weights = torch.empty_like(final_ids, dtype=torch.float32)

    def launch() -> None:
        operation.run_into(hidden_states=hidden, router_weights=router_weights)
        routing.finalize_moe_routing(
            routed_ids=operation.ids,
            routed_weights=operation.weights,
            final_ids=final_ids,
            final_weights=final_weights,
            payload_rows=live_rows,
            routed_expert_count=layer.routed_expert_count,
            shared_expert_count=layer.shared_expert_count,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    for _ in range(3):
        launch()
    eager_ids, eager_weights = final_ids.cpu(), final_weights.cpu()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    ids, carrier_weights = final_ids.cpu(), final_weights.cpu()
    graph_equal = torch.equal(eager_ids, ids) and torch.equal(eager_weights, carrier_weights)
    graph_equal &= bool((ids[rows:] == -1).all() and (carrier_weights[rows:] == 0).all())

    # Sigmoid overwrites its workspace; rerun the same projection into a separate destination.
    projected = torch.empty_like(operation.logits)
    projected_input = hidden.float() if projection.values.dtype == torch.float32 else hidden
    operation.projection.run_into(projected_input, projection, output=projected)
    reference_projection = hidden[:rows].double() @ projection.values.double().T
    if projection.bias is not None:
        reference_projection += projection.bias.double()
    logits, reference_logits = projected[:rows].cpu(), reference_projection.cpu()
    tensors = {
        "hidden_states": case.hidden_states,
        "projection": logits,
        "reference_projection": reference_logits,
        "ids": ids,
        "weights": carrier_weights,
        "installed_ids": case.topk_ids,
        "installed_weights": case.topk_weights,
        "eager_ids": eager_ids,
        "eager_weights": eager_weights,
        "projection_values": projection.values.cpu(),
    }
    if correction is not None:
        tensors["correction_bias"] = correction.cpu()
    if projection.bias is not None:
        tensors["projection_bias"] = projection.bias.cpu()
    safetensors.torch.save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        destination / "router.safetensors",
        metadata={
            "case_id": case.case_id,
            "layer_id": str(case.layer_id),
            "execution_capacity": str(capacity),
        },
    )
    comparison = compare_ffn_router(
        layer=layer,
        execution_capacity=capacity,
        logits=logits,
        reference_logits=reference_logits,
        correction_bias=None if correction is None else correction.cpu(),
        topk_ids=ids[:rows, : layer.routed_topk],
        topk_weights=carrier_weights[:rows, : layer.routed_topk],
        installed_ids_equal=torch.equal(ids[:rows], case.topk_ids),
        installed_weights_equal=torch.equal(carrier_weights[:rows], case.topk_weights),
        eager_graph_equal=graph_equal,
    )
    (destination / "comparison.json").write_text(json.dumps(asdict(comparison), indent=2) + "\n")
    return comparison
