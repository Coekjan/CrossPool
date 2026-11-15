"""Prepared Dense execution over caller-owned Lane storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import linear


@dataclass(frozen=True, slots=True)
class DenseExecutionPlan:
    """Two prepared projections and their exact shared intermediate backing."""

    signature: execution.DenseFfnExecutionSignature
    i_proj: linear.GemmPlan
    o_proj: linear.GemmPlan
    workspace: execution.WorkspaceLayout

    @classmethod
    def prepare(cls, signature: execution.DenseFfnExecutionSignature) -> DenseExecutionPlan:
        """Derive storage once, retaining standalone ungated input/output aliasing."""

        activated = TensorLayout(
            (signature.payload_row_capacity, signature.local_intermediate_size), signature.payload_dtype, 0
        )
        i_proj = linear.GemmPlan.prepare(
            input_layout=TensorLayout(signature.payload_shape, signature.payload_dtype, 0),
            weight_layout=signature.i_proj,
            result_layout=activated,
            activation=signature.activation,
        )
        o_proj = linear.GemmPlan.prepare(
            input_layout=activated,
            weight_layout=signature.o_proj,
            result_layout=TensorLayout(signature.payload_shape, signature.payload_dtype, 0),
            activation=None,
        )
        regions = [(i_proj.output_layout.dtype, i_proj.output_layout.shape)]
        if i_proj.fused_activation is None and not signature.activation.supports_inplace:
            regions.append((activated.dtype, activated.shape))
        regions.extend(((torch.uint8, (i_proj.workspace.nbytes,)), (torch.uint8, (o_proj.workspace.nbytes,))))
        workspace = execution.WorkspaceLayout.from_regions(tuple(regions))
        return cls(
            signature=signature,
            i_proj=i_proj,
            o_proj=o_proj,
            workspace=workspace,
        )


class Dense:
    """Capture-time Dense computation borrowing prepared views and layer resources."""

    def __init__(self, plan: DenseExecutionPlan, workspace: torch.Tensor) -> None:
        views = plan.workspace.views(workspace)
        self.plan = plan
        self.projected = views[0]
        self.activated = (
            views[1]
            if plan.i_proj.fused_activation is None and not plan.signature.activation.supports_inplace
            else self.projected
        )
        self.i_proj = linear.Gemm(plan.i_proj, views[-2])
        self.o_proj = linear.Gemm(plan.o_proj, views[-1])

    def run_into(
        self,
        *,
        hidden_states: torch.Tensor,
        layer_weights: weights.DenseFfnWeights,
        output: torch.Tensor,
    ) -> None:
        """Write a complete rank-local Partial using the current layer bindings."""

        self.i_proj.run_into(hidden_states, layer_weights.i_proj, output=self.projected)
        if self.plan.i_proj.fused_activation is None:
            self.plan.signature.activation.apply_into(
                inputs=self.projected,
                output=self.activated,
                order=self.plan.i_proj.weight_layout.gate_up_order,
            )
        self.o_proj.run_into(self.activated, layer_weights.o_proj, output=output)
