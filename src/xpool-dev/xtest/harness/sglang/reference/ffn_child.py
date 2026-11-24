"""Isolated pinned-SGLang bootstrap for FFN reference evaluation."""

from __future__ import annotations

import os
import sys
import typing
from multiprocessing.connection import Connection

import torch

from xtest.harness.sglang.reference import ffn_protocol
from xtest.harness.support.environment import qualification_environment

SGLANG_PLUGIN_SENTINEL = "__xpool_sglang_ffn_reference_no_plugins__"
GEMMA4_MODEL_CLASSES = {"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"}
SUPPORTED_MODEL_CLASSES = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GptOssForCausalLM",
        *GEMMA4_MODEL_CLASSES,
        "LlamaForCausalLM",
        "OPTForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
    }
)


def read_routing_output(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the actual upstream carrier on CPU after plugin-isolated bootstrap.

    The process entry imports SGLang only after selecting the reference plugin
    set, so the external carrier types are resolved at this boundary.
    """

    from sglang.srt.layers.moe.topk import StandardTopKOutput, TritonKernelTopKOutput

    if isinstance(output, StandardTopKOutput):
        return (
            output.topk_ids.detach().to(device="cpu", dtype=torch.int32).contiguous(),
            output.topk_weights.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        )
    if not isinstance(output, TritonKernelTopKOutput):
        raise RuntimeError(f"FFN reference received an unsupported TopK carrier: {type(output).__name__}")

    expert_sizes = output.a_ragged_metadata.slice_sizes.detach().cpu()
    expert_ids = torch.repeat_interleave(
        torch.arange(expert_sizes.numel(), dtype=torch.int32), expert_sizes.to(torch.int64)
    )
    scatter = output.scatter_indx.detach().to(device="cpu", dtype=torch.int64)
    token_expert_ids = torch.empty_like(scatter, dtype=torch.int32)
    token_weights = torch.empty_like(scatter, dtype=torch.float32)
    # Invert the upstream expert-major dispatch into its original token slots.
    token_expert_ids[scatter] = expert_ids
    token_weights[scatter] = output.gate_scal.detach().to(device="cpu", dtype=torch.float32)
    return token_expert_ids.view(-1, output.n_expts_act), token_weights.view(-1, output.n_expts_act)


def replace_routing_output(
    output: object,
    *,
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
    """Return the rebuilt SGLang carrier and its verified CPU IDs and weights."""

    from sglang.srt.layers.moe.topk import StandardTopKOutput, TritonKernelTopKOutput, routing

    if isinstance(output, StandardTopKOutput):
        expected_weights = topk_weights.to(output.topk_weights.dtype)
        result = StandardTopKOutput(expected_weights, topk_ids.to(output.topk_ids.dtype), output.router_logits)
    elif isinstance(output, TritonKernelTopKOutput):
        metadata, gather, scatter, _, count = routing(router_logits, topk_ids.shape[1], expt_indx=topk_ids)
        expected_weights = topk_weights.to(output.gate_scal.dtype)
        result = TritonKernelTopKOutput(metadata, gather, scatter, expected_weights.flatten()[scatter.long()], count)
    else:
        raise RuntimeError(f"FFN reference cannot prescribe TopK carrier {type(output).__name__}")
    consumed_ids, consumed_weights = read_routing_output(result)
    if not torch.equal(consumed_ids, topk_ids.cpu().int()) or not torch.equal(
        consumed_weights, expected_weights.cpu().float()
    ):
        raise RuntimeError("FFN reference carrier did not preserve prescribed IDs and weights")
    return result, (consumed_ids, consumed_weights)


def run_ffn_reference_child(
    connection: Connection,
    job: ffn_protocol.FfnReferenceJob,
) -> None:
    """Filter the inherited settings before SGLang or its TP ranks start."""

    environment = qualification_environment()
    os.environ.clear()
    os.environ.update(environment)
    os.environ["SGLANG_PLUGINS"] = SGLANG_PLUGIN_SENTINEL
    if any(
        name == "xpool.integrations.sglang" or name.startswith("xpool.integrations.sglang.") for name in sys.modules
    ):
        raise RuntimeError("xpool SGLang integration was imported before FFN reference isolation")

    import torch.multiprocessing

    rendezvous_path = job.workdir / "torch-distributed-rendezvous"
    rendezvous_uri = rendezvous_path.resolve().as_uri()
    if job.tensor_parallel_size == 1:
        run_ffn_reference_rank(0, job, rendezvous_uri)
    else:
        torch.multiprocessing.spawn(
            run_ffn_reference_rank,
            args=(job, rendezvous_uri),
            nprocs=job.tensor_parallel_size,
            join=True,
        )
    connection.send(ffn_protocol.FfnReferenceCompleted(case_count=len(job.cases)))


def run_ffn_reference_rank(
    tensor_parallel_rank: int,
    job: ffn_protocol.FfnReferenceJob,
    rendezvous_uri: str,
) -> None:
    """Load one TP shard, execute every case, and release SGLang state."""

    os.environ["SGLANG_PLUGINS"] = SGLANG_PLUGIN_SENTINEL
    if any(
        name == "xpool.integrations.sglang" or name.startswith("xpool.integrations.sglang.") for name in sys.modules
    ):
        raise RuntimeError("xpool SGLang integration was imported before FFN reference isolation")

    import gc

    import sglang.srt.configs.device_config
    import sglang.srt.configs.load_config
    import sglang.srt.configs.model_config
    import sglang.srt.distributed
    import sglang.srt.distributed.parallel_state
    import sglang.srt.layers.dp_attention
    import sglang.srt.layers.moe.utils
    import sglang.srt.layers.utils
    import sglang.srt.model_loader
    import sglang.srt.runtime_context
    import sglang.srt.server_args
    import torch
    import torch.distributed

    torch.cuda.set_device(tensor_parallel_rank)
    server_args = typing.cast(typing.Any, sglang.srt.server_args.ServerArgs)(
        model_path=str(job.model_path),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="auto",
        device="cuda",
        load_format="auto",
        tp_size=job.tensor_parallel_size,
        pp_size=1,
        dp_size=job.tensor_parallel_size,
        ep_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_a2a_backend="none",
        moe_runner_backend="auto",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_custom_all_reduce=True,
    )
    sglang.srt.runtime_context.publish(server_args, role="scheduler")
    sglang.srt.layers.moe.utils.initialize_moe_config()
    model_config = sglang.srt.configs.model_config.ModelConfig.from_server_args(server_args)
    sglang.srt.distributed.set_custom_all_reduce(False)

    # The reference uses SGLang's real model loader and TP groups while keeping
    # CrossPool's plugin and dispatcher path absent from the process.
    model = None
    failure: BaseException | None = None
    vllm_parallel_state_patched = False
    try:
        sglang.srt.distributed.init_distributed_environment(
            world_size=job.tensor_parallel_size,
            rank=tensor_parallel_rank,
            distributed_init_method=rendezvous_uri,
            local_rank=tensor_parallel_rank,
            backend="nccl",
            timeout=sglang.srt.runtime_context.get_parallel().dist_timeout,
            moe_a2a_backend="none",
        )
        sglang.srt.distributed.initialize_model_parallel(
            tensor_model_parallel_size=job.tensor_parallel_size,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            attention_data_parallel_size=job.tensor_parallel_size,
            attention_context_model_parallel_size=1,
            moe_data_model_parallel_size=1,
        )
        group_widths = (
            sglang.srt.distributed.parallel_state.get_tensor_model_parallel_world_size(),
            sglang.srt.distributed.parallel_state.get_attn_tensor_model_parallel_world_size(),
            sglang.srt.distributed.parallel_state.get_moe_tensor_parallel_world_size(),
            sglang.srt.distributed.parallel_state.get_pipeline_model_parallel_world_size(),
        )
        expected_group_widths = (job.tensor_parallel_size, 1, job.tensor_parallel_size, 1)
        if group_widths != expected_group_widths:
            raise RuntimeError(
                f"FFN reference parallel group widths {group_widths} do not match {expected_group_widths}"
            )
        sglang.srt.layers.dp_attention.initialize_dp_attention(
            server_args=server_args,
            model_config=model_config,
        )
        load_config = sglang.srt.configs.load_config.LoadConfig(
            load_format=sglang.srt.runtime_context.get_model().load_format,
            download_dir=sglang.srt.runtime_context.get_model().download_dir,
            model_loader_extra_config=sglang.srt.runtime_context.get_model().model_loader_extra_config,
            tp_rank=tensor_parallel_rank,
        )
        sglang.srt.distributed.parallel_state.monkey_patch_vllm_parallel_state()
        vllm_parallel_state_patched = True
        model = sglang.srt.model_loader.get_model(
            model_config=model_config,
            load_config=load_config,
            device_config=sglang.srt.configs.device_config.DeviceConfig("cuda", tensor_parallel_rank),
        )
        sglang.srt.distributed.parallel_state.monkey_patch_vllm_parallel_state(reverse=True)
        vllm_parallel_state_patched = False
        model_class = type(model).__name__
        if model_class not in SUPPORTED_MODEL_CLASSES:
            raise RuntimeError(f"unsupported FFN reference model class: {model_class}")
        model.eval()
        # The pinned loader returns unrelated concrete model classes.
        typed_model = typing.cast(typing.Any, model)
        if model_class in GEMMA4_MODEL_CLASSES:
            layers = typed_model.language_model.layers
        elif model_class == "OPTForCausalLM":
            layers = typed_model.model.decoder.layers
        else:
            layers = typed_model.model.layers

        for case in job.cases:
            if case.layer_id >= len(layers):
                raise RuntimeError(f"FFN reference layer {case.layer_id} is outside the model layer range")
            layer = layers[case.layer_id]
            if isinstance(layer, sglang.srt.layers.utils.PPMissingLayer):
                raise RuntimeError(f"FFN reference TP rank {tensor_parallel_rank} does not own layer {case.layer_id}")
            execute_ffn_reference_case(
                hidden_size=model_config.hidden_size,
                model_dtype=model_config.dtype,
                mlp=layer if model_class == "OPTForCausalLM" else layer.mlp,
                case=case,
                persist_result=tensor_parallel_rank == 0,
            )
            torch.distributed.barrier()
    except BaseException as error:
        failure = error
    finally:
        if vllm_parallel_state_patched:
            sglang.srt.distributed.parallel_state.monkey_patch_vllm_parallel_state(reverse=True)
        model = None
        try:
            sglang.srt.distributed.destroy_model_parallel()
            sglang.srt.distributed.destroy_distributed_environment()
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as cleanup_error:
            if failure is None:
                raise
            failure.add_note(f"FFN reference cleanup also failed: {cleanup_error!r}")
    if failure is not None:
        raise failure


def execute_ffn_reference_case(
    *,
    hidden_size: int,
    model_dtype: torch.dtype,
    mlp: torch.nn.Module,
    case: ffn_protocol.FfnReferenceCaseSpec,
    persist_result: bool,
) -> None:
    """Execute one TP FFN case and persist its rank-zero result."""

    import safetensors.torch
    import sglang.srt.layers.dp_attention
    import sglang.srt.models.opt
    import torch

    tensors = safetensors.torch.load_file(case.input_path, device="cpu")
    if set(tensors) not in ({"hidden_states"}, {"hidden_states", "topk_ids", "topk_weights"}):
        raise RuntimeError(f"FFN reference input has invalid tensor keys: {sorted(tensors)}")
    hidden_states = tensors["hidden_states"]
    if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden_size:
        raise RuntimeError(
            f"FFN reference input shape {tuple(hidden_states.shape)} does not match hidden size {hidden_size}"
        )
    if hidden_states.dtype != model_dtype:
        raise RuntimeError(f"FFN reference input dtype {hidden_states.dtype} does not match model dtype {model_dtype}")
    global_num_tokens = [hidden_states.shape[0]] * torch.distributed.get_world_size()
    sglang.srt.layers.dp_attention.set_dp_buffer_len(
        sum(global_num_tokens),
        hidden_states.shape[0],
        True,
        global_num_tokens,
    )
    sglang.srt.layers.dp_attention.set_is_extend_in_batch(False)

    hook_handle = None
    topk_calls = 0
    natural_routing: tuple[torch.Tensor, torch.Tensor] | None = None
    prescribed_routing: tuple[torch.Tensor, torch.Tensor] | None = None
    if "topk_ids" in tensors:
        topk = getattr(mlp, "topk", None)
        if topk is None:
            raise RuntimeError("Dense FFN reference cannot consume prescribed routes")
        ids = tensors["topk_ids"].to(device="cuda")
        coefficients = tensors["topk_weights"].to(device="cuda")
        shared_count = int(getattr(mlp, "n_shared_experts", getattr(mlp, "num_shared_experts", 0)) or 0)
        fused_shared_count = int(getattr(mlp, "num_fused_shared_experts", 0) or 0)
        if shared_count and not fused_shared_count:
            ids = ids[:, :-shared_count].contiguous()
            coefficients = coefficients[:, :-shared_count].contiguous()
        if getattr(getattr(mlp, "experts", None), "should_fuse_routed_scaling_factor_in_topk", False):
            coefficients = coefficients * float(getattr(mlp, "routed_scaling_factor"))

        def prescribe_topk(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> object:
            nonlocal topk_calls, natural_routing, prescribed_routing
            del module
            if (
                ids.shape != (hidden_states.shape[0], topk.topk_config.top_k)
                or (ids >= inputs[1].shape[1] + fused_shared_count).any()
            ):
                raise ValueError("FFN reference prescribed routes disagree with the loaded model")
            if persist_result:
                natural_routing = read_routing_output(output)
            result, consumed_routing = replace_routing_output(
                output, router_logits=inputs[1], topk_ids=ids, topk_weights=coefficients
            )
            if persist_result:
                prescribed_routing = consumed_routing
            topk_calls += 1
            return result

        hook_handle = topk.register_forward_hook(prescribe_topk)

    try:
        with torch.inference_mode():
            if isinstance(mlp, sglang.srt.models.opt.OPTDecoderLayer):
                projected, _ = mlp.fc1(hidden_states.to(device="cuda"))
                output, _ = mlp.fc2(mlp.activation_fn(projected))
            else:
                output = mlp(hidden_states.to(device="cuda"))
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"FFN reference MLP returned {type(output).__name__}, expected Tensor")
        output = output.detach().to(device="cpu").contiguous()
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if output.shape != hidden_states.shape or output.dtype != hidden_states.dtype:
        raise RuntimeError(
            "FFN reference output shape/dtype does not match its input: "
            f"{tuple(output.shape)}/{output.dtype} versus {tuple(hidden_states.shape)}/{hidden_states.dtype}"
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("FFN reference output contains non-finite values")
    if hook_handle is not None and topk_calls != 1:
        raise RuntimeError(f"FFN reference observed {topk_calls} TopK calls, expected one")
    if not persist_result:
        return

    with case.output_path.open("xb") as output_file:
        output_file.write(safetensors.torch.save({"hidden_states": output}))
    if hook_handle is not None:
        if natural_routing is None or prescribed_routing is None:
            raise RuntimeError("FFN reference did not retain its controlled routing evidence")
        with case.routing_path.open("xb") as routing_file:
            routing_file.write(
                safetensors.torch.save(
                    {
                        "natural_ids": natural_routing[0],
                        "natural_weights": natural_routing[1],
                        "prescribed_ids": prescribed_routing[0],
                        "prescribed_weights": prescribed_routing[1],
                    },
                    metadata={"case_id": case.case_id, "layer_id": str(case.layer_id)},
                )
            )
