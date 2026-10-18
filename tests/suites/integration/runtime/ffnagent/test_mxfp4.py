from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

import xtest
from xpool import ffn
from xpool.config import XpoolConfig
from xpool.runtime.ffnagent import architecture, execution, loader, operators, registry, weights
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def write_mxfp4_checkpoint(path: Path) -> tuple[ffn.FfnModelSpec, dict[str, torch.Tensor]]:
    """Create two distinct biased Experts layers with nontrivial scale/nibble values."""

    torch.manual_seed(71)
    path.mkdir()
    config = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "hidden_size": 96,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "swiglu_limit": 7.0,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    (path / "config.json").write_text(json.dumps(config))
    spec = GptOssAdapter.compile(model_id=TEST_MODEL_ID, model_config=architecture.FfnSourceConfig(config))
    tensors = {}
    for layer in spec.layers:
        assert isinstance(layer, ffn.MoeFfnSpec) and isinstance(layer.checkpoint, ffn.Mxfp4MoeCheckpointKeys)
        k = layer.checkpoint
        tensors.update(
            {
                k.router_weight_key: torch.randn(4, 96).bfloat16(),
                k.router_bias_key: torch.randn(4).bfloat16(),
                k.gate_up_blocks_key: torch.randint(0, 256, (4, 256, 3, 16), dtype=torch.uint8),
                k.gate_up_scales_key: torch.randint(120, 125, (4, 256, 3), dtype=torch.uint8),
                k.gate_up_bias_key: torch.randn(4, 256).bfloat16(),
                k.down_blocks_key: torch.randint(0, 256, (4, 96, 4, 16), dtype=torch.uint8),
                k.down_scales_key: torch.randint(120, 125, (4, 96, 4), dtype=torch.uint8),
                k.down_bias_key: torch.randn(4, 96).bfloat16(),
            }
        )
    safetensors.torch.save_file(tensors, path / "model.safetensors")
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1], "loader": {"parallelism": 2}},
                "models": [{"id": str(TEST_MODEL_ID), "path": str(path)}],
            }
        )
    )
    return architecture.load(model_id=TEST_MODEL_ID, model_path=path), tensors


def decode_reference(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independently decode all sixteen nibble values on CPU for comparison."""

    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    packed = blocks.flatten(-2)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return (lut[nibbles] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)).bfloat16()


def layer_reference(
    x: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    keys: ffn.Mxfp4MoeCheckpointKeys,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Evaluate biased/clamped gated Experts independently of production kernels."""

    a = decode_reference(tensors[keys.gate_up_blocks_key], tensors[keys.gate_up_scales_key])
    b = decode_reference(tensors[keys.down_blocks_key], tensors[keys.down_scales_key])
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[row, slot])
            if e < 0:
                continue
            gu = (x[row].float() @ a[e].float().T + tensors[keys.gate_up_bias_key][e].float()).bfloat16()
            g = gu[::2].float().clamp(max=7)
            u = gu[1::2].float().clamp(-7, 7)
            activated = (g * torch.sigmoid(1.702 * g) * (u + 1)).bfloat16()
            y = activated.float() @ b[e].float().T + tensors[keys.down_bias_key][e].float()
            result[row] += (y * probabilities[row, slot]).bfloat16().float()
    return result.bfloat16()


@pytest.mark.parametrize("tp_size", (1, 2))
@xtest.requirements(device_count=1)
def test_mxfp4_loader_tp_bias_and_graph(tmp_path: Path, tp_size: int) -> None:
    spec, tensors = write_mxfp4_checkpoint(tmp_path / "model")
    layer = spec.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec) and isinstance(layer.checkpoint, ffn.Mxfp4MoeCheckpointKeys)
    requests = tuple(
        loader.LocalLayerWeightRequest(
            model_path=tmp_path / "model",
            hidden_size=96,
            payload_dtype=torch.bfloat16,
            router_weight_dtype=torch.bfloat16,
            layer=layer,
            tp_rank=rank,
            tp_size=tp_size,
        )
        for rank in range(tp_size)
    )
    local = loader.materialize_local_layer_weights(requests=requests)
    x = torch.randn(19, 96, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (19, 2), device="cuda", dtype=torch.int32)
    ids[-2:] = -1
    p = torch.softmax(torch.randn(19, 2, device="cuda"), dim=-1)
    p[-2:] = 0
    partials = []
    for rank, w in enumerate(local):
        assert isinstance(w, weights.Mxfp4MoeFfnWeights)
        assert (w.down_bias is not None) == (rank == 0)
        assert (w.router is not None) == (rank == 0)
        assert w.gate_up_packed.dtype is torch.uint8 and w.down_packed.dtype is torch.uint8
        expected = tensors[layer.checkpoint.gate_up_blocks_key][
            :, rank * 256 // tp_size : (rank + 1) * 256 // tp_size
        ].flatten(-2)
        torch.testing.assert_close(w.gate_up_packed.cpu(), expected, rtol=0, atol=0)
        sig = execution.MoeFfnExecutionSignature(
            payload_dtype=torch.bfloat16,
            payload_row_capacity=19,
            hidden_size=96,
            local_intermediate_size=128 // tp_size,
            expert_count=4,
            effective_topk=2,
            activation=ffn.ActivationKind.SWIGLU_OAI,
            routed_scaling_factor=1,
            router=None,
            mxfp4=execution.Mxfp4ExecutionConfig(alpha=1.702, limit=7, down_bias_present=rank == 0),
        )
        workspace = registry.allocate_moe_graph_capture_workspace(signature=sig, w13_config={"BLOCK_SIZE_M": 16})
        out = torch.empty_like(x)

        def launch() -> None:
            assert sig.mxfp4 is not None
            operators.compute_mxfp4_partial(
                hidden_states=x,
                layer_weights=w,
                topk_ids=ids,
                topk_weights=p,
                sorted_token_ids=workspace.sorted_token_ids,
                expert_ids=workspace.expert_ids,
                num_tokens_post_padded=workspace.num_tokens_post_padded,
                cumsum_buffer=workspace.cumsum_buffer,
                gate_up=workspace.gate_up,
                activated=workspace.activated,
                route_outputs=workspace.route_outputs,
                output=out,
                config=sig.mxfp4,
            )

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        partials.append(out.cpu().float())
    result = torch.stack(partials).sum(dim=0).bfloat16()
    reference = layer_reference(x.cpu(), tensors, layer.checkpoint, ids.cpu(), p.cpu())
    torch.testing.assert_close(result, reference, rtol=0.025, atol=0.025)
    assert torch.count_nonzero(result[-2:]) == 0


@xtest.requirements(device_count=1)
def test_mxfp4_capture_accounts_every_weight_resource(tmp_path: Path) -> None:
    spec, _ = write_mxfp4_checkpoint(tmp_path / "model")
    layer = spec.layers[0]
    (w,) = loader.materialize_local_layer_weights(
        requests=(
            loader.LocalLayerWeightRequest(
                model_path=tmp_path / "model",
                hidden_size=96,
                payload_dtype=torch.bfloat16,
                router_weight_dtype=torch.bfloat16,
                layer=layer,
                tp_rank=0,
                tp_size=1,
            ),
        )
    )
    assert isinstance(w, weights.Mxfp4MoeFfnWeights)
    router = execution.MoeRouterExecutionSignature(
        compute_routed_topk=GptOssAdapter.compute_routed_topk,
        router_weight_dtype=torch.bfloat16,
        routed_expert_count=4,
        router_workspace_bytes=GptOssAdapter.router_workspace_bytes(
            payload_dtype=torch.bfloat16,
            payload_row_capacity=19,
            hidden_size=96,
            routed_expert_count=4,
            routed_topk=2,
        ),
        correction_bias_present=False,
        renormalize=True,
        bias_present=True,
    )
    sig = execution.MoeFfnExecutionSignature(
        payload_dtype=torch.bfloat16,
        payload_row_capacity=19,
        hidden_size=96,
        local_intermediate_size=128,
        expert_count=4,
        effective_topk=2,
        activation=ffn.ActivationKind.SWIGLU_OAI,
        routed_scaling_factor=1,
        router=router,
        mxfp4=execution.Mxfp4ExecutionConfig(alpha=1.702, limit=7, down_bias_present=True),
    )
    probe = registry.create_control_capture_probe(w)
    assert isinstance(probe, weights.Mxfp4MoeFfnWeights)
    assert probe.down_bias is not None and probe.router is not None and probe.router.bias is not None
    capture = registry.capture_moe_signature(sig, w, probe)
    actual = tuple(
        t.nbytes
        for t in (
            probe.gate_up_packed,
            probe.gate_up_scales,
            probe.down_packed,
            probe.down_scales,
            probe.gate_up_bias,
            probe.down_bias,
            probe.router.weight,
            probe.router.bias,
        )
    )
    assert execution.control_capture_probe_storage_bytes(sig) == actual
    assert capture.projection is not None


@pytest.mark.timeout(240)
@xtest.requirements(device_count=3)
def test_mxfp4_installed_graph_rebinds_layers_and_tp_bias(tmp_path: Path) -> None:
    """Exercise real scale/bias rebinding and TP delivery through installed Lanes."""

    from xkit.network import TcpEndpointReservation, TcpPortSpace
    from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
    from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement
    from xpool.runtime.transport import InstanceRankTransportProfile
    from xtest.harness.native.ffn.protocol import FfnInstanceSpec, FfnInvocationSpec
    from xtest.harness.native.ffn.topology import materialize_ffn_cluster_launch, run_ffn_topology

    spec, tensors = write_mxfp4_checkpoint(tmp_path / "model")
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"ffn_concurrency": 2, "slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": str(tmp_path / "model"), "ffn_tp_size": 2}],
        }
    )
    endpoint = TcpEndpointReservation.reserve(config.daemon.host, port_space=TcpPortSpace.local())
    workdir = tmp_path / "installed"
    launch = materialize_ffn_cluster_launch(config=config, daemon_port=endpoint.port, workdir=workdir)
    x = torch.randn(3, 96).bfloat16()
    input_path = workdir / "input.safetensors"
    safetensors.torch.save_file({"hidden_states": x}, input_path)
    invocations = tuple(
        FfnInvocationSpec(
            case_id=f"exchange-{index}",
            input_path=input_path,
            output_path=workdir / f"output-{index}.safetensors",
            layer_ordinal=layer_id,
            forward_mode=ForwardMode.PREFILL,
            output_requirement=OutputRequirement.PER_RANK_COMPLETE,
            dp_row_layout=DpRowLayout.NONE,
            dp_rank_payload_rows=None,
        )
        for index, layer_id in enumerate((0, 1, 0))
    )
    profile = InstanceFfnProfile(
        payload_dtype=torch.bfloat16,
        hidden_size=96,
        layers=tuple(InstanceFfnLayerProfile(layer_id=entry.layer_id, kind=entry.kind) for entry in spec.layers),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=3,
        group_sum_complete_admitted=False,
    )
    instance = FfnInstanceSpec(
        environment=dict(launch.environment),
        model_id=TEST_MODEL_ID,
        rank=0,
        ffn_tp_size=2,
        ffn_profile=profile,
        transport=InstanceRankTransportProfile(
            hidden_size=96, payload_row_capacity=3, atn_tp_rank=0, atn_tp_size=1, atn_dp_rank=0, atn_dp_size=1
        ),
        invocations=invocations,
    )
    run_ffn_topology(launch=launch, endpoint=endpoint, instance_specs=(instance,), workdir=workdir, timeout_seconds=180)
    for invocation in invocations:
        selected_layer = spec.layers[invocation.layer_ordinal]
        assert isinstance(selected_layer, ffn.MoeFfnSpec)
        assert isinstance(selected_layer.checkpoint, ffn.Mxfp4MoeCheckpointKeys)
        k = selected_layer.checkpoint
        # Original CUDA Linear owns the Router's library-specific BF16 rounding.
        logits = torch.nn.functional.linear(
            x.cuda(), tensors[k.router_weight_key].cuda(), tensors[k.router_bias_key].cuda()
        ).cpu()
        values, ids = torch.topk(logits.float(), 2, dim=-1)
        reference = layer_reference(x, tensors, k, ids, torch.softmax(values, dim=-1))
        actual = safetensors.torch.load_file(invocation.output_path)["hidden_states"]
        torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.03)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtype_field", ("dtype", "torch_dtype", None))
@xtest.requirements(device_count=1)
def test_original_sglang_mxfp4_layer_reference(tmp_path: Path, dtype_field: str | None) -> None:
    """Exercise upstream loading/decoding in an isolated process without a full model replica."""

    from xtest.harness.sglang.reference.ffn import SglangFfnReferenceCase, SglangFfnReferenceRunner

    spec, tensors = write_mxfp4_checkpoint(tmp_path / "model")
    config_path = tmp_path / "model" / "config.json"
    config = json.loads(config_path.read_text())
    dtype = config.pop("torch_dtype")
    if dtype_field is not None:
        config[dtype_field] = dtype
    config_path.write_text(json.dumps(config))
    x = torch.randn(3, 96).bfloat16()
    (result,) = SglangFfnReferenceRunner(workdir=tmp_path / "reference", timeout_seconds=120).run(
        model_path=tmp_path / "model",
        tensor_parallel_size=1,
        cases=(SglangFfnReferenceCase(case_id="layer-zero", layer_id=0, hidden_states=x),),
    )
    assert result.routing is not None
    layer = spec.layers[0]
    assert isinstance(layer, ffn.MoeFfnSpec) and isinstance(layer.checkpoint, ffn.Mxfp4MoeCheckpointKeys)
    expected = layer_reference(x, tensors, layer.checkpoint, result.routing.topk_ids, result.routing.topk_weights)
    torch.testing.assert_close(result.output, expected, rtol=0.025, atol=0.025)
