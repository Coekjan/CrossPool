"""DeepSeek-V2 FFN architecture compiler."""

from __future__ import annotations

import torch

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, operators, weights


class DeepseekV2Adapter(architecture.MoeFfnModelAdapter):
    """Compile the strict mixed Dense/MoE DeepSeek-V2 FFN profile."""

    architecture_name = "DeepseekV2ForCausalLM"

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
        """Return the caller-owned FP32 Router-logit extent."""

        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("DeepSeek Router requires BF16 or FP16 payloads")
        if min(payload_row_capacity, routed_expert_count, routed_topk) <= 0 or routed_topk > routed_expert_count:
            raise ValueError("DeepSeek Router dimensions are inconsistent")
        return payload_row_capacity * routed_expert_count * torch.float32.itemsize

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
        """Run the admitted DeepSeek grouped-Softmax replica."""

        payload_dtype = hidden_states.dtype
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("DeepSeek Router requires BF16 or FP16 payloads")
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
            raise ValueError("DeepSeek Router tensor geometry disagrees")
        routed_topk = routed_ids.shape[1]
        if routed_ids.shape[0] != row_capacity or router_weights.correction_bias is not None:
            raise ValueError("DeepSeek Router resources disagree with the admitted profile")
        expected_bytes = DeepseekV2Adapter.router_workspace_bytes(
            payload_dtype=payload_dtype,
            payload_row_capacity=row_capacity,
            hidden_size=hidden_size,
            routed_expert_count=routed_expert_count,
            routed_topk=routed_topk,
        )
        if workspace.numel() != expected_bytes:
            raise ValueError(f"DeepSeek Router workspace must contain exactly {expected_bytes} bytes")
        tensors = (router_weights.weight, workspace, routed_ids, routed_weights)
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("DeepSeek Router tensors must share one CUDA device")
        logits = workspace.view(torch.float32).view(row_capacity, routed_expert_count)
        torch.mm(hidden_states, router_weights.weight.t(), out=logits, out_dtype=torch.float32)
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
        """Compile DeepSeek layer placement and routing semantics."""

        model_config.validate_family_profile(model_type="deepseek_v2", dtype_field="torch_dtype")
        hidden_size = model_config.get("hidden_size", int, ge=1)
        layer_count = model_config.get("num_hidden_layers", int, ge=1)
        intermediate_size = model_config.get("intermediate_size", int, ge=1)
        expert_intermediate_size = model_config.get("moe_intermediate_size", int, ge=1)
        routed_expert_count = model_config.get("n_routed_experts", int, ge=1)
        shared_expert_count = model_config.get("n_shared_experts", int, ge=0)
        routed_topk = model_config.get("num_experts_per_tok", int, ge=1)
        first_moe_layer = model_config.get("first_k_dense_replace", int, ge=0)
        moe_layer_frequency = model_config.get("moe_layer_freq", int, ge=1)
        expert_group_count = model_config.get("n_group", int, ge=1)
        selected_expert_group_count = model_config.get("topk_group", int, ge=1)
        if model_config.get("scoring_func", str) != "softmax":
            raise ValueError("scoring_func must equal 'softmax'")
        if model_config.get("topk_method", str) != "greedy":
            raise ValueError("topk_method must equal 'greedy'")
        if expert_group_count != 1 or selected_expert_group_count != 1:
            raise ValueError("first-production DeepSeek routing requires n_group == topk_group == 1")
        renormalize = model_config.get("norm_topk_prob", bool)
        routed_scaling_factor = model_config.get("routed_scaling_factor", float, gt=0)

        layers: list[ffn.FfnLayerSpec] = []
        for layer_id in range(layer_count):
            if layer_id < first_moe_layer or layer_id % moe_layer_frequency != 0:
                layers.append(
                    ffn.DenseFfnSpec(
                        kind=LayerKind.DENSE,
                        layer_id=layer_id,
                        intermediate_size=intermediate_size,
                        checkpoint=architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp"),
                    )
                )
                continue
            layers.append(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    layer_id=layer_id,
                    expert_intermediate_size=expert_intermediate_size,
                    shared_expert_count=shared_expert_count,
                    routed_topk=routed_topk,
                    renormalize=renormalize,
                    routed_scaling_factor=routed_scaling_factor,
                    checkpoint=ffn.MoeFfnCheckpointKeys(
                        router_weight_key=f"model.layers.{layer_id}.mlp.gate.weight",
                        router_correction_bias_key=None,
                        routed_experts=tuple(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.experts.{expert_id}")
                            for expert_id in range(routed_expert_count)
                        ),
                        shared_expert=(
                            architecture.gated_checkpoint_keys(f"model.layers.{layer_id}.mlp.shared_experts")
                            if shared_expert_count > 0
                            else None
                        ),
                    ),
                )
            )
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(layers),
        )
