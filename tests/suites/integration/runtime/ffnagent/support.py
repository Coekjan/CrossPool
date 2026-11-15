"""Build execution plans from the floating tensors used by operator tests."""

import torch

from xpool import ffn, quantization
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import dense, routing
from xpool.runtime.ffnagent.operators.activation import Activation


def floating_projection_layout(
    projection: weights.ProjectionWeights, *, order: ffn.GateUpOrder | None = None
) -> weights.ProjectionLayout:
    return weights.ProjectionLayout(
        quantization=quantization.Unquantized(),
        values=TensorLayout(tuple(projection.values.shape), projection.values.dtype, 0),
        scales=None,
        bias=TensorLayout(tuple(projection.bias.shape), projection.bias.dtype, 0)
        if projection.bias is not None
        else None,
        gate_up_order=order,
    )


def dense_plan(
    hidden_states: torch.Tensor, layer_weights: weights.DenseFfnWeights, activation: Activation
) -> dense.DenseExecutionPlan:
    return dense.DenseExecutionPlan.prepare(
        execution.DenseFfnExecutionSignature(
            payload_dtype=hidden_states.dtype,
            payload_row_capacity=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            local_intermediate_size=layer_weights.o_proj.values.shape[1],
            activation=activation,
            i_proj=floating_projection_layout(
                layer_weights.i_proj,
                order=ffn.GateUpOrder.CONCATENATED if activation.gate is not None else None,
            ),
            o_proj=floating_projection_layout(layer_weights.o_proj),
        )
    )


def router_plan(
    hidden_states: torch.Tensor,
    router_weights: weights.MoeRouterWeights,
    *,
    algorithm: ffn.RouterAlgorithm,
    precision: ffn.RouterProjectionPrecision,
    routed_topk: int,
    renormalize: bool,
    weight_rounding: ffn.TopKWeightRounding = ffn.TopKWeightRounding.NONE,
) -> routing.RouterExecutionPlan:
    correction_bias = router_weights.correction_bias
    return routing.RouterExecutionPlan.prepare(
        execution.MoeRouterExecutionSignature(
            algorithm=algorithm,
            projection_precision=precision,
            weight_rounding=weight_rounding,
            projection=floating_projection_layout(router_weights.projection),
            correction_bias=TensorLayout(tuple(correction_bias.shape), correction_bias.dtype, 0)
            if correction_bias is not None
            else None,
            routed_expert_count=router_weights.projection.values.shape[0],
            renormalize=renormalize,
        ),
        input_layout=TensorLayout(tuple(hidden_states.shape), hidden_states.dtype, 0),
        routed_topk=routed_topk,
    )
