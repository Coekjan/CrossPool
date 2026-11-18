from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sglang.srt.layers.moe.topk import StandardTopKOutput, TritonKernelTopKOutput, routing

import xtest
from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.quantization import Unquantized
from xtest.harness.sglang.reference.ffn_child import read_routing_output, replace_routing_output
from xtest.harness.sglang.reference.router import RouterReferenceCase, run_router_reference
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@xtest.requirements(device_count=1)
def test_reference_reads_actual_expert_sorted_triton_routing_without_recomputing_it() -> None:
    row = torch.arange(16, dtype=torch.float32, device="cuda")
    logits = torch.stack((row, row.flip(0), torch.roll(row, 5)))
    carrier = TritonKernelTopKOutput(*routing(logits, 2))

    expert_ids, weights = read_routing_output(carrier)

    selected, expected_ids = torch.topk(logits.cpu(), 2, dim=-1)
    torch.testing.assert_close(expert_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.softmax(selected, dim=-1), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("expert_major", (False, True), ids=("standard", "expert-major"))
@xtest.requirements(device_count=1)
def test_reference_carrier_consumes_prescribed_ids_and_weights(expert_major: bool) -> None:
    logits = torch.arange(48, device="cuda", dtype=torch.float32).view(3, 16)
    natural_ids = torch.tensor(((15, 14),) * 3, device="cuda", dtype=torch.int32)
    natural_weights = torch.softmax(logits[:, -2:].flip(1), dim=-1)
    carrier = (
        TritonKernelTopKOutput(*routing(logits, 2))
        if expert_major
        else StandardTopKOutput(natural_weights, natural_ids, logits)
    )
    ids = torch.tensor(((1, 7), (8, 3), (15, 0)), device="cuda", dtype=torch.int32)
    weights = torch.tensor(((0.2, 0.8), (0.6, 0.4), (0.1, 0.9)), device="cuda", dtype=torch.float32)

    replaced = replace_routing_output(carrier, router_logits=logits, topk_ids=ids, topk_weights=weights)
    consumed_ids, consumed_weights = read_routing_output(replaced)

    assert torch.equal(consumed_ids, ids.cpu())
    assert torch.equal(consumed_weights, weights.cpu())
    self_ids, self_weights = read_routing_output(carrier)
    self_replaced = replace_routing_output(
        carrier, router_logits=logits, topk_ids=self_ids.cuda(), topk_weights=self_weights.cuda()
    )
    restored_ids, restored_weights = read_routing_output(self_replaced)
    assert torch.equal(restored_ids, self_ids)
    assert torch.equal(restored_weights, self_weights)


@xtest.requirements(device_count=1)
def test_router_retains_identified_raw_tensors_when_comparison_domain_is_unsupported(tmp_path: Path) -> None:
    layer = ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=7,
        router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
        router_projection_precision=ffn.RouterProjectionPrecision.FP32_LOGITS,
        expert_intermediate_size=32,
        routed_expert_count=16,
        shared_expert_count=0,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key="router"),
        router_correction_bias_key="correction",
        routed_experts=ffn.StackedExpertSource(
            gate_up=ffn.ProjectionSpec(quantization=Unquantized(), values_key="up"),
            down=ffn.ProjectionSpec(quantization=Unquantized(), values_key="down"),
        ),
        shared_expert=None,
    )
    model_spec = ffn.FfnModelSpec(
        model_id=ModelId("test/router-metadata"),
        hidden_size=16,
        activation=ffn.SiluActivation(),
        layers=(layer,),
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_file(
        {"router": torch.zeros((16, 16), dtype=torch.bfloat16), "correction": torch.zeros(16)},
        model_path / "model.safetensors",
    )
    case = RouterReferenceCase(
        case_id="unsupported-softmax",
        layer_id=7,
        hidden_states=torch.zeros((1, 16), dtype=torch.bfloat16),
        execution_capacity=32,
        topk_ids=torch.tensor(((0, 1),), dtype=torch.int32),
        topk_weights=torch.full((1, 2), 0.5),
    )
    workdir = tmp_path / "router-reference"
    with pytest.raises(RuntimeError, match="does not cover corrected Softmax"):
        run_router_reference(
            model_spec=model_spec,
            model_path=model_path,
            tensor_parallel_size=1,
            cases=(case,),
            device=0,
            workdir=workdir,
            timeout_seconds=120,
        )

    with safe_open(str(workdir / "case-0000" / "router.safetensors"), framework="pt") as reader:
        assert reader.metadata() == {
            "case_id": case.case_id,
            "layer_id": "7",
            "execution_capacity": "32",
        }
        assert torch.equal(reader.get_tensor("hidden_states"), case.hidden_states)
        assert reader.get_tensor("ids").shape == (32, 2)
