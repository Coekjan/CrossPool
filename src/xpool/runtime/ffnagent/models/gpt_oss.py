"""GPT-OSS biased MoE semantics with serialized MXFP4 Experts."""

from __future__ import annotations

import torch

import xpool.native
from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, operators, weights


class GptOssAdapter(architecture.MoeFfnModelAdapter):
    """Compile GPT-OSS config independently of the serving engine runtime."""

    architecture_name = "GptOssForCausalLM"

    @staticmethod
    def router_weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype:
        """Preserve the upstream BF16 biased Router boundary."""

        if payload_dtype is not torch.bfloat16:
            raise ValueError("GPT-OSS requires BF16 hidden-state payloads")
        return torch.bfloat16

    @staticmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
        hidden_size: int,
        routed_expert_count: int,
        routed_topk: int,
    ) -> int:
        """Account for aligned explicit cuBLASLt scratch and BF16 logits."""

        GptOssAdapter.router_weight_dtype(payload_dtype=payload_dtype)
        return (
            xpool.native.ffnagent.BF16_ROUTER_WORKSPACE_BYTES
            + xpool.native.ffnagent.BF16_ROUTER_WORKSPACE_ALIGNMENT
            - 1
            + payload_row_capacity * routed_expert_count * 2
        )

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
        """Add bias before selecting and normalizing Expert probabilities."""

        if router_weights.bias is None or router_weights.correction_bias is not None or not renormalize:
            raise ValueError("GPT-OSS requires additive Router bias and normalized TopK")
        alignment = xpool.native.ffnagent.BF16_ROUTER_WORKSPACE_ALIGNMENT
        begin = (-workspace.data_ptr()) % alignment
        end = begin + xpool.native.ffnagent.BF16_ROUTER_WORKSPACE_BYTES
        logits_bytes = hidden_states.shape[0] * router_weights.weight.shape[0] * 2
        logits = (
            workspace[end : end + logits_bytes]
            .view(torch.bfloat16)
            .view(hidden_states.shape[0], router_weights.weight.shape[0])
        )
        xpool.native.ffnagent.compute_bf16_router_logits(
            hidden_states, router_weights.weight, router_weights.bias, logits, workspace[begin:end]
        )
        operators.compute_softmax_topk(
            logits=logits, routed_ids=routed_ids, routed_weights=routed_weights, renormalize=True
        )

    @classmethod
    def compile(cls, *, model_id: ModelId, model_config: architecture.FfnSourceConfig) -> ffn.FfnModelSpec:
        """Validate serialized MXFP4 source semantics and fused checkpoint keys."""

        if model_config.get("model_type", str) != "gpt_oss":
            raise ValueError("model_type must equal 'gpt_oss'")
        quantization = model_config.get("quantization_config", dict)
        if quantization.get("quant_method") != "mxfp4":
            raise ValueError("GPT-OSS requires a serialized MXFP4 checkpoint")
        dtype = model_config.optional("dtype", str) or model_config.optional("torch_dtype", str) or "bfloat16"

        if dtype != "bfloat16":
            raise ValueError("GPT-OSS source dtype must equal 'bfloat16'")
        h = model_config.get("hidden_size", int, ge=32)
        i = model_config.get("intermediate_size", int, ge=32)
        if h % 32 or i % 32:
            raise ValueError("GPT-OSS dimensions must align to 32-value scale blocks")
        e = model_config.get("num_local_experts", int, ge=1)
        topk = model_config.get("num_experts_per_tok", int, ge=1, le=e)
        alpha = model_config.optional("hidden_act_alpha", float)
        limit = model_config.get("swiglu_limit", float, gt=0)
        layers = []
        for layer_id in range(model_config.get("num_hidden_layers", int, ge=1)):
            prefix = f"model.layers.{layer_id}.mlp"
            layers.append(
                ffn.MoeFfnSpec(
                    kind=LayerKind.MOE,
                    layer_id=layer_id,
                    expert_intermediate_size=i,
                    shared_expert_count=0,
                    routed_topk=topk,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    checkpoint=ffn.Mxfp4MoeCheckpointKeys(
                        expert_count=e,
                        router_weight_key=f"{prefix}.router.weight",
                        router_bias_key=f"{prefix}.router.bias",
                        gate_up_blocks_key=f"{prefix}.experts.gate_up_proj_blocks",
                        gate_up_scales_key=f"{prefix}.experts.gate_up_proj_scales",
                        gate_up_bias_key=f"{prefix}.experts.gate_up_proj_bias",
                        down_blocks_key=f"{prefix}.experts.down_proj_blocks",
                        down_scales_key=f"{prefix}.experts.down_proj_scales",
                        down_bias_key=f"{prefix}.experts.down_proj_bias",
                    ),
                )
            )
        return ffn.FfnModelSpec(
            model_id=model_id,
            architecture_name=cls.architecture_name,
            hidden_size=h,
            activation=ffn.ActivationKind.SWIGLU_OAI,
            activation_alpha=alpha if alpha is not None else 1.702,
            activation_limit=limit,
            layers=tuple(layers),
        )
