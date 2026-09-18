"""Qwen3-MoE FFN architecture compiler."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, operators, weights


class Qwen3MoeAdapter(architecture.MoeFfnModelAdapter):
    """Compile the strict all-MoE Qwen3 FFN profile."""

    architecture_name = "Qwen3MoeForCausalLM"

    @staticmethod
    def router_weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype:
        return payload_dtype

    @staticmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
        hidden_size: int,
        routed_expert_count: int,
        routed_topk: int,
    ) -> int:
        """Return the caller-owned Router-logit extent."""

        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Qwen3-MoE Router requires BF16 or FP16 payloads")
        if min(payload_row_capacity, routed_expert_count, routed_topk) <= 0 or routed_topk > routed_expert_count:
            raise ValueError("Qwen3-MoE Router dimensions are inconsistent")
        return payload_row_capacity * routed_expert_count * payload_dtype.itemsize

    @staticmethod
    def compute_routed_topk(
        *,
        hidden_states: torch.Tensor,
        router_weights: weights.MoeRouterWeights,
        workspace: torch.Tensor,
        routed_ids: torch.Tensor,
        routed_weights: torch.Tensor,
        renormalize: bool,
    ) -> None:
        """Run Qwen3-MoE projection and public caller-output TopK."""

        payload_dtype = hidden_states.dtype
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Qwen3-MoE Router requires BF16 or FP16 payloads")
        weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
        weights.validate_tensor(workspace, name="Router workspace", dtype=torch.uint8, dimensions=1)
        weights.validate_tensor(routed_ids, name="routed Expert ids", dtype=torch.int32, dimensions=2)
        weights.validate_tensor(routed_weights, name="routed Expert weights", dtype=torch.float32, dimensions=2)
        row_capacity, hidden_size = hidden_states.shape
        routed_expert_count, router_hidden_size = router_weights.weight.shape
        if (
            router_hidden_size != hidden_size
            or router_weights.weight.dtype is not payload_dtype
            or routed_ids.shape != routed_weights.shape
        ):
            raise ValueError("Qwen3-MoE Router tensor geometry disagrees")
        routed_topk = routed_ids.shape[1]
        if routed_ids.shape[0] != row_capacity or router_weights.correction_bias is not None:
            raise ValueError("Qwen3-MoE Router resources disagree with the admitted profile")
        expected_bytes = Qwen3MoeAdapter.router_workspace_bytes(
            payload_dtype=payload_dtype,
            payload_row_capacity=row_capacity,
            hidden_size=hidden_size,
            routed_expert_count=routed_expert_count,
            routed_topk=routed_topk,
        )
        if workspace.numel() != expected_bytes:
            raise ValueError(f"Qwen3-MoE Router workspace must contain exactly {expected_bytes} bytes")
        tensors = (router_weights.weight, workspace, routed_ids, routed_weights)
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("Qwen3-MoE Router tensors must share one CUDA device")
        logits = workspace.view(payload_dtype).view(row_capacity, routed_expert_count)
        torch.mm(hidden_states, router_weights.weight.t(), out=logits)
        operators.compute_softmax_topk(
            logits=logits,
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            renormalize=renormalize,
        )

    @classmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: architecture.FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile all main Qwen3-MoE decoder layers and routing semantics."""

        model_config.validate_family_profile(model_type="qwen3_moe", dtype_field="torch_dtype")
        if model_config.get("decoder_sparse_step", int, ge=1) != 1:
            raise ValueError("decoder_sparse_step must equal 1")
        mlp_only_layers = model_config.optional("mlp_only_layers", list)
        if not isinstance(mlp_only_layers, list) or mlp_only_layers:
            raise ValueError("mlp_only_layers must be an empty JSON array")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        expert_intermediate_size = model_config.get("moe_intermediate_size", int, ge=1)
        routed_expert_count = model_config.get("num_experts", int, ge=1)
        routed_topk = model_config.get("num_experts_per_tok", int, ge=1)
        renormalize = model_config.get("norm_topk_prob", bool)
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    layer_id=layer_id,
                    expert_intermediate_size=expert_intermediate_size,
                    shared_expert_count=0,
                    routed_topk=routed_topk,
                    renormalize=renormalize,
                    routed_scaling_factor=1.0,
                    checkpoint=ffn.MoeFfnCheckpointKeys(
                        router_weight_key=f"model.layers.{layer_id}.mlp.gate.weight",
                        router_correction_bias_key=None,
                        routed_experts=tuple(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.experts.{expert_id}")
                            for expert_id in range(routed_expert_count)
                        ),
                        shared_expert=None,
                    ),
                )
                for layer_id in range(layer_count)
            ),
        )
