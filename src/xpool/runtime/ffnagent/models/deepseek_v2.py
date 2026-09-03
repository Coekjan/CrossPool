"""DeepSeek-V2 FFN architecture compiler."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, operators, weights


class DeepseekV2Adapter(architecture.MoeFfnModelAdapter):
    """Compile the strict mixed Dense/MoE DeepSeek-V2 FFN profile."""

    architecture_name = "DeepseekV2ForCausalLM"

    @staticmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
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
        model_config: Mapping[str, object],
        model_config_digest: str,
    ) -> ffn.FfnModelSpec:
        """Compile DeepSeek layer placement and routing semantics."""

        architecture.require_family_profile(
            model_config,
            architecture_name="DeepseekV2ForCausalLM",
            model_type="deepseek_v2",
            dtype_field="torch_dtype",
        )
        hidden_size = architecture.require_integer(model_config, "hidden_size", minimum=1)
        layer_count = architecture.require_integer(model_config, "num_hidden_layers", minimum=1)
        intermediate_size = architecture.require_integer(model_config, "intermediate_size", minimum=1)
        expert_intermediate_size = architecture.require_integer(model_config, "moe_intermediate_size", minimum=1)
        routed_expert_count = architecture.require_integer(model_config, "n_routed_experts", minimum=1)
        shared_expert_count = architecture.require_integer(model_config, "n_shared_experts", minimum=0)
        routed_topk = architecture.require_integer(model_config, "num_experts_per_tok", minimum=1)
        first_moe_layer = architecture.require_integer(model_config, "first_k_dense_replace", minimum=0)
        moe_layer_frequency = architecture.require_integer(model_config, "moe_layer_freq", minimum=1)
        expert_group_count = architecture.require_integer(model_config, "n_group", minimum=1)
        selected_expert_group_count = architecture.require_integer(model_config, "topk_group", minimum=1)
        if architecture.require_string(model_config, "scoring_func") != "softmax":
            raise ValueError("scoring_func must equal 'softmax'")
        if architecture.require_string(model_config, "topk_method") != "greedy":
            raise ValueError("topk_method must equal 'greedy'")
        if expert_group_count != 1 or selected_expert_group_count != 1:
            raise ValueError("first-production DeepSeek routing requires n_group == topk_group == 1")
        renormalize = architecture.require_boolean(model_config, "norm_topk_prob")
        routed_scaling_factor = architecture.require_positive_number(model_config, "routed_scaling_factor")

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
            model_config_digest=model_config_digest,
            hidden_size=hidden_size,
            activation=ffn.ActivationKind.SILU,
            layers=tuple(layers),
        )
