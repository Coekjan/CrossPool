"""Isolated pinned-SGLang bootstrap for FFN reference evaluation."""

from __future__ import annotations

import json
import os
import sys
import typing
from multiprocessing.connection import Connection

import torch

from xtest.harness.sglang.reference import ffn_protocol

SGLANG_PLUGIN_SENTINEL = "__xpool_sglang_ffn_reference_no_plugins__"
SUPPORTED_MODEL_CLASSES = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "LlamaForCausalLM",
        "GptOssForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
    }
)


def run_ffn_reference_child(
    connection: Connection,
    job: ffn_protocol.FfnReferenceJob,
) -> None:
    """Run every TP rank after isolating the original SGLang plugin set."""

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
    import sglang.srt.layers.moe.topk
    import sglang.srt.layers.moe.utils
    import sglang.srt.layers.utils
    import sglang.srt.model_loader
    import sglang.srt.model_loader.utils
    import sglang.srt.runtime_context
    import sglang.srt.server_args
    import torch
    import torch.distributed

    torch.cuda.set_device(tensor_parallel_rank)
    source_config = json.loads((job.model_path / "config.json").read_text())
    is_gpt_oss = source_config.get("architectures") == ["GptOssForCausalLM"]
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
        moe_runner_backend="triton" if is_gpt_oss else "auto",
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
        if model_config.hf_config.architectures == ["GptOssForCausalLM"]:
            for case in job.cases:
                # Match the dtype context used by SGLang's full model loader.
                with sglang.srt.model_loader.utils.set_default_torch_dtype(model_config.dtype):
                    mlp = load_gpt_oss_reference_layer(model_config.hf_config, job.model_path, case.layer_id)
                execute_ffn_reference_case(
                    hidden_size=model_config.hidden_size,
                    model_dtype=model_config.dtype,
                    mlp=mlp,
                    case=case,
                    persist_result=tensor_parallel_rank == 0,
                )
                torch.distributed.barrier()
                del mlp
                gc.collect()
                torch.cuda.empty_cache()
            return
        model = sglang.srt.model_loader.get_model(
            model_config=model_config,
            load_config=load_config,
            device_config=sglang.srt.configs.device_config.DeviceConfig("cuda", tensor_parallel_rank),
        )
        sglang.srt.distributed.parallel_state.monkey_patch_vllm_parallel_state(reverse=True)
        vllm_parallel_state_patched = False
        if type(model).__name__ not in SUPPORTED_MODEL_CLASSES:
            raise RuntimeError(f"unsupported FFN reference model class: {type(model).__name__}")
        model.eval()
        # The pinned loader returns unrelated concrete model classes.
        typed_model = typing.cast(typing.Any, model)

        for case in job.cases:
            if case.layer_id >= len(typed_model.model.layers):
                raise RuntimeError(f"FFN reference layer {case.layer_id} is outside the model layer range")
            layer = typed_model.model.layers[case.layer_id]
            if isinstance(layer, sglang.srt.layers.utils.PPMissingLayer):
                raise RuntimeError(f"FFN reference TP rank {tensor_parallel_rank} does not own layer {case.layer_id}")
            execute_ffn_reference_case(
                hidden_size=model_config.hidden_size,
                model_dtype=model_config.dtype,
                mlp=layer.mlp,
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


def load_gpt_oss_reference_layer(config: object, model_path: object, layer_id: int) -> torch.nn.Module:
    """Load one original SGLang layer with upstream MXFP4 decoding on A100.

    The temporary model shell supplies the exact checkpoint namespace to the
    original loader. It contains no attention or unrelated Expert layers and
    imports no CrossPool production materialization or decoding code.
    """

    from pathlib import Path

    from safetensors import safe_open
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
    from sglang.srt.models.gpt_oss import GptOssForCausalLM, GptOssSparseMoeBlock

    cfg = typing.cast(typing.Any, config)
    path = Path(typing.cast(str, model_path))
    quantization = Mxfp4Config.from_config(cfg.quantization_config)
    with torch.device("cuda"):
        mlp = GptOssSparseMoeBlock(
            config=cfg, layer_id=layer_id, quant_config=quantization, prefix=f"model.layers.{layer_id}.mlp"
        )
    shell = GptOssForCausalLM.__new__(GptOssForCausalLM)
    torch.nn.Module.__init__(shell)
    shell.config = cfg
    shell.quant_config = quantization
    shell.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.add_module("mlp", mlp)
    shell.model.add_module("layers", torch.nn.ModuleDict({str(layer_id): layer}))
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        import json

        mapping = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            mapping = dict.fromkeys(f.keys(), "model.safetensors")
    prefix = f"model.layers.{layer_id}.mlp."

    def expert_tensors():
        for key, filename in mapping.items():
            if key.startswith(prefix + "experts."):
                with safe_open(str(path / filename), framework="pt", device="cpu") as f:
                    yield key, f.get_tensor(key)

    shell._load_mxfp4_experts_weights(expert_tensors())
    for suffix, destination in (("weight", mlp.router.weight), ("bias", mlp.router.bias)):
        key = prefix + "router." + suffix
        with safe_open(str(path / mapping[key]), framework="pt", device="cpu") as f:
            typing.cast(torch.Tensor, destination).data.copy_(f.get_tensor(key))
    mlp.experts.quant_method.process_weights_after_loading(mlp.experts)
    return mlp.eval()


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
    import sglang.srt.layers.moe.topk
    import torch

    tensors = safetensors.torch.load_file(case.input_path, device="cpu")
    if set(tensors) != {"hidden_states"}:
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

    captured_routing: list[tuple[torch.Tensor, torch.Tensor]] = []
    shared_expert_calls: list[None] = []
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    # Rank zero records the router's actual carrier rather than reconstructing
    # routing from logits, preserving model-specific correction semantics.
    topk = getattr(mlp, "topk", None)
    if persist_result and topk is not None:

        def capture_topk(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: sglang.srt.layers.moe.topk.StandardTopKOutput | sglang.srt.layers.moe.topk.StandardTopKOutputPacked,
        ) -> None:
            del module, inputs
            if not isinstance(
                output,
                (
                    sglang.srt.layers.moe.topk.StandardTopKOutput,
                    sglang.srt.layers.moe.topk.StandardTopKOutputPacked,
                ),
            ):
                raise RuntimeError(f"FFN reference requires a standard TopK carrier, received {type(output).__name__}")
            captured_routing.append(
                (
                    output.topk_ids.detach().to(device="cpu", dtype=torch.int32).clone(),
                    output.topk_weights.detach().to(device="cpu", dtype=torch.float32).clone(),
                )
            )

        hook_handles.append(topk.register_forward_hook(capture_topk))

    shared_expert_count = int(getattr(mlp, "n_shared_experts", getattr(mlp, "num_shared_experts", 0)) or 0)
    fused_shared_expert_count = int(getattr(mlp, "num_fused_shared_experts", 0) or 0)
    shared_expert = getattr(mlp, "shared_experts", getattr(mlp, "shared_expert", None))
    # Separately executed shared experts are appended to routed evidence; fused
    # implementations already include them in the router's output carrier.
    if persist_result and shared_expert_count > 0 and fused_shared_expert_count == 0:
        if shared_expert is None:
            raise RuntimeError("FFN reference could not find the separately executed shared expert")

        def capture_shared_expert(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs, output
            shared_expert_calls.append(None)

        hook_handles.append(shared_expert.register_forward_hook(capture_shared_expert))

    try:
        with torch.inference_mode():
            output = mlp(hidden_states.to(device="cuda"))
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"FFN reference MLP returned {type(output).__name__}, expected Tensor")
        output = output.detach().to(device="cpu").contiguous()
    finally:
        for handle in hook_handles:
            handle.remove()

    if output.shape != hidden_states.shape or output.dtype != hidden_states.dtype:
        raise RuntimeError(
            "FFN reference output shape/dtype does not match its input: "
            f"{tuple(output.shape)}/{output.dtype} versus {tuple(hidden_states.shape)}/{hidden_states.dtype}"
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("FFN reference output contains non-finite values")
    if not persist_result:
        return

    output_tensors = {"hidden_states": output}
    if topk is not None:
        if len(captured_routing) != 1:
            raise RuntimeError(f"FFN reference observed {len(captured_routing)} TopK calls, expected one")
        topk_ids, topk_weights = captured_routing[0]
        if shared_expert_count > 0 and fused_shared_expert_count == 0:
            if len(shared_expert_calls) != 1:
                raise RuntimeError(
                    f"FFN reference observed {len(shared_expert_calls)} shared-expert calls, expected one"
                )
            routed_expert_count = int(
                getattr(
                    mlp,
                    "num_experts",
                    getattr(getattr(mlp, "config", None), "n_routed_experts"),
                )
            )
            shared_ids = torch.arange(
                routed_expert_count,
                routed_expert_count + shared_expert_count,
                dtype=torch.int32,
            ).expand(topk_ids.shape[0], -1)
            topk_ids = torch.cat((topk_ids, shared_ids), dim=1)
            topk_weights = torch.cat(
                (topk_weights, torch.ones_like(shared_ids, dtype=torch.float32)),
                dim=1,
            )
        output_tensors["topk_ids"] = topk_ids.contiguous()
        output_tensors["topk_weights"] = topk_weights.contiguous()

    with case.output_path.open("xb") as output_file:
        output_file.write(safetensors.torch.save(output_tensors))
