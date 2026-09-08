"""GLM-4 MoE Lite FFN architecture compiler."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import torch
import triton
from triton import language

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture, weights


@torch.compile(fullgraph=True)
def copy_router_input(source: torch.Tensor, target: torch.Tensor) -> None:
    """Convert into caller-owned FP32 storage before the Router GEMM.

    Pre-capture warmup compiles this operation. Inductor avoids the opaque
    empty-object parameters in ATen's copy kernel that conflict with the
    declared-address-only Graph parameterization contract.
    """

    target.copy_(source)


@triton.jit
def biased_sigmoid_topk_kernel(
    logits_pointer,
    correction_bias_pointer,
    routed_ids_pointer,
    routed_weights_pointer,
    EXPERT_COUNT: language.constexpr,
    TOPK: language.constexpr,
    EXPERT_BLOCK_SIZE: language.constexpr,
    TOPK_BLOCK_SIZE: language.constexpr,
):
    """Select normalized GLM routes using caller-owned FP32 logits as scratch.

    Equal biased scores prefer higher Expert IDs. The stored-score boundary
    preserves the qualified scoring arithmetic; removing it can change routes
    near a tied cutoff. Expert padding never participates in selection.
    """

    row = language.program_id(0)
    experts = language.arange(0, EXPERT_BLOCK_SIZE)
    valid = experts < EXPERT_COUNT
    logits_offsets = row * EXPERT_COUNT + experts
    correction_bias = language.load(correction_bias_pointer + experts, mask=valid, other=0.0)
    selection = language.sigmoid(language.load(logits_pointer + logits_offsets, mask=valid, other=0.0))
    selection += correction_bias
    language.store(logits_pointer + logits_offsets, selection, mask=valid)
    language.debug_barrier()
    selection = language.load(logits_pointer + logits_offsets, mask=valid, other=-float("inf"))
    scores = selection - correction_bias

    normalizer = language.full((), 0.0, language.float32)
    for slot in range(TOPK):
        maximum = language.max(selection, axis=0)
        expert = language.max(language.where(valid & (selection == maximum), experts, -1), axis=0)
        weight = language.sum(language.where(experts == expert, scores, 0.0), axis=0)
        language.store(routed_ids_pointer + row * TOPK + slot, expert)
        language.store(routed_weights_pointer + row * TOPK + slot, weight)
        normalizer += weight
        selection = language.where(experts == expert, -float("inf"), selection)

    # The selection loop and this vectorized load may assign slots to different
    # threads. Complete their stores before normalizing the caller-owned output.
    language.debug_barrier()
    slots = language.arange(0, TOPK_BLOCK_SIZE)
    output_offsets = row * TOPK + slots
    selected_weights = language.load(routed_weights_pointer + output_offsets, mask=slots < TOPK, other=0.0)
    denominator = language.where(normalizer > 0.0, normalizer, 1.0)
    language.store(routed_weights_pointer + output_offsets, selected_weights / denominator, mask=slots < TOPK)


class Glm4MoeLiteAdapter(architecture.MoeFfnModelAdapter):
    """Compile the strict mixed Dense/MoE GLM-4 Lite FFN profile."""

    architecture_name = "Glm4MoeLiteForCausalLM"

    @staticmethod
    def router_weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype:
        return torch.float32

    @staticmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
        hidden_size: int,
        routed_expert_count: int,
        routed_topk: int,
    ) -> int:
        """Size caller-owned FP32 input conversion and Router-logit storage."""

        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("GLM Router requires BF16 or FP16 payloads")
        if min(payload_row_capacity, routed_expert_count, routed_topk) <= 0 or routed_topk > routed_expert_count:
            raise ValueError("GLM Router dimensions are inconsistent")
        return payload_row_capacity * (hidden_size + routed_expert_count) * torch.float32.itemsize

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
        """Run GLM projection and caller-owned biased-sigmoid TopK."""

        payload_dtype = hidden_states.dtype
        if payload_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("GLM Router requires BF16 or FP16 payloads")
        weights.validate_tensor(hidden_states, name="hidden_states", dtype=payload_dtype, dimensions=2)
        weights.validate_tensor(workspace, name="Router workspace", dtype=torch.uint8, dimensions=1)
        weights.validate_tensor(routed_ids, name="routed Expert ids", dtype=torch.int32, dimensions=2)
        weights.validate_tensor(routed_weights, name="routed Expert weights", dtype=torch.float32, dimensions=2)
        row_capacity, hidden_size = hidden_states.shape
        routed_expert_count, router_hidden_size = router_weights.weight.shape
        routed_topk = routed_ids.shape[1]
        correction_bias = router_weights.correction_bias
        if (
            router_hidden_size != hidden_size
            or router_weights.weight.dtype is not torch.float32
            or routed_ids.shape != routed_weights.shape
            or routed_ids.shape[0] != row_capacity
            or correction_bias is None
            or correction_bias.shape != (routed_expert_count,)
            or not renormalize
        ):
            raise ValueError("GLM Router tensors or semantics disagree with the admitted profile")
        expected_bytes = Glm4MoeLiteAdapter.router_workspace_bytes(
            payload_dtype=payload_dtype,
            payload_row_capacity=row_capacity,
            hidden_size=hidden_size,
            routed_expert_count=routed_expert_count,
            routed_topk=routed_topk,
        )
        if workspace.numel() != expected_bytes:
            raise ValueError(f"GLM Router workspace must contain exactly {expected_bytes} bytes")
        tensors = (router_weights.weight, correction_bias, workspace, routed_ids, routed_weights)
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("GLM Router tensors must share one CUDA device")
        scratch = workspace.view(torch.float32)
        input_elements = row_capacity * hidden_size
        router_input = scratch[:input_elements].view(row_capacity, hidden_size)
        logits = scratch[input_elements:].view(row_capacity, routed_expert_count)
        # Retain FP32 operands like the pinned gate, without capture-time storage allocation.
        copy_router_input(hidden_states, router_input)
        torch.mm(router_input, router_weights.weight.t(), out=logits)
        typing.cast(typing.Callable[..., None], biased_sigmoid_topk_kernel[(row_capacity,)])(
            logits,
            correction_bias,
            routed_ids,
            routed_weights,
            EXPERT_COUNT=typing.cast(language.constexpr, routed_expert_count),
            TOPK=typing.cast(language.constexpr, routed_topk),
            EXPERT_BLOCK_SIZE=typing.cast(language.constexpr, triton.next_power_of_2(routed_expert_count)),
            TOPK_BLOCK_SIZE=typing.cast(language.constexpr, triton.next_power_of_2(routed_topk)),
            num_warps=2,
        )

    @classmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: Mapping[str, object],
        model_config_digest: str,
    ) -> ffn.FfnModelSpec:
        """Compile GLM main layers and corrected-sigmoid routing semantics."""

        architecture.require_family_profile(
            model_config,
            architecture_name="Glm4MoeLiteForCausalLM",
            model_type="glm4_moe_lite",
            dtype_field="dtype",
        )
        explicit_frequency = model_config.get("moe_layer_freq")
        if explicit_frequency is not None and (
            not isinstance(explicit_frequency, int) or isinstance(explicit_frequency, bool) or explicit_frequency != 1
        ):
            raise ValueError("explicit moe_layer_freq must equal the GLM family constant 1")
        hidden_size = architecture.require_integer(model_config, "hidden_size", minimum=1)
        layer_count = architecture.require_integer(model_config, "num_hidden_layers", minimum=1)
        intermediate_size = architecture.require_integer(model_config, "intermediate_size", minimum=1)
        expert_intermediate_size = architecture.require_integer(model_config, "moe_intermediate_size", minimum=1)
        routed_expert_count = architecture.require_integer(model_config, "n_routed_experts", minimum=1)
        shared_expert_count = architecture.require_integer(model_config, "n_shared_experts", minimum=0)
        routed_topk = architecture.require_integer(model_config, "num_experts_per_tok", minimum=1)
        first_moe_layer = architecture.require_integer(model_config, "first_k_dense_replace", minimum=0)
        expert_group_count = architecture.require_integer(model_config, "n_group", minimum=1)
        selected_expert_group_count = architecture.require_integer(model_config, "topk_group", minimum=1)
        if architecture.require_string(model_config, "topk_method") != "noaux_tc":
            raise ValueError("topk_method must equal 'noaux_tc'")
        if expert_group_count != 1 or selected_expert_group_count != 1:
            raise ValueError("first-production GLM routing requires n_group == topk_group == 1")
        renormalize = architecture.require_boolean(model_config, "norm_topk_prob")
        routed_scaling_factor = architecture.require_positive_number(model_config, "routed_scaling_factor")

        layers: list[ffn.FfnLayerSpec] = []
        for layer_id in range(layer_count):
            if layer_id < first_moe_layer:
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
                        router_correction_bias_key=(f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"),
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
