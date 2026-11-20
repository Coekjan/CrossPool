from pathlib import Path
from types import SimpleNamespace

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
from xtest.harness.sglang.reference.ffn import read_reference_result
from xtest.harness.sglang.reference.ffn_child import (
    execute_ffn_reference_case,
    read_routing_output,
    replace_routing_output,
)
from xtest.harness.sglang.reference.ffn_protocol import FfnReferenceCaseSpec
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

    replaced, verified_routing = replace_routing_output(
        carrier, router_logits=logits, topk_ids=ids, topk_weights=weights
    )
    consumed_ids, consumed_weights = read_routing_output(replaced)

    torch.testing.assert_close(verified_routing, (consumed_ids, consumed_weights), rtol=0, atol=0)
    assert torch.equal(consumed_ids, ids.cpu())
    assert torch.equal(consumed_weights, weights.cpu())
    self_ids, self_weights = read_routing_output(carrier)
    self_replaced, _ = replace_routing_output(
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


@pytest.mark.parametrize(("expert_major", "shared_count", "routed_scale"), [(False, 1, 2.0), (True, 0, 1.0)])
@pytest.mark.parametrize("persist_result", (False, True), ids=("participant", "writer"))
@xtest.requirements(device_count=1)
def test_reference_retains_natural_and_effective_routes_in_the_same_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_major: bool,
    shared_count: int,
    routed_scale: float,
    persist_result: bool,
) -> None:
    logits = torch.arange(48, device="cuda", dtype=torch.float32).view(3, 16)
    carrier = (
        TritonKernelTopKOutput(*routing(logits, 2))
        if expert_major
        else StandardTopKOutput(
            torch.full((3, 2), 0.5, device="cuda", dtype=torch.bfloat16),
            torch.tensor([[15, 14]] * 3, device="cuda", dtype=torch.int32),
            logits,
        )
    )
    natural_ids, natural_weights = read_routing_output(carrier)
    ids = torch.tensor([[1, 7], [8, 3], [15, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    expected_weights = (weights * routed_scale).to(torch.float32 if expert_major else torch.bfloat16).float()

    class TopK(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.topk_config = SimpleNamespace(top_k=2)
            self.calls = 0

        def forward(self, hidden: torch.Tensor, router_logits: torch.Tensor) -> object:
            self.calls += 1
            return carrier

    class Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.topk = TopK()
            self.n_shared_experts = shared_count
            self.num_fused_shared_experts = 0
            self.routed_scaling_factor = routed_scale
            self.experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=routed_scale != 1.0)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            consumed_ids, consumed_weights = read_routing_output(self.topk(hidden, logits))
            assert torch.equal(consumed_ids, ids)
            assert torch.equal(consumed_weights, expected_weights)
            return hidden + consumed_weights.sum(dim=-1, keepdim=True).to(hidden)

    spec = FfnReferenceCaseSpec(
        "carrier-diagnostic", 7, tmp_path / "input.safetensors", tmp_path / "output.safetensors"
    )
    if shared_count:
        input_ids = torch.cat((ids, torch.full((3, 1), 16, dtype=torch.int32)), dim=1)
        input_weights = torch.cat((weights, torch.ones((3, 1))), dim=1)
    else:
        input_ids, input_weights = ids, weights
    save_file(
        {
            "hidden_states": torch.ones((3, 4), dtype=torch.bfloat16),
            "topk_ids": input_ids,
            "topk_weights": input_weights,
        },
        spec.input_path,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    mlp = Mlp()
    execute_ffn_reference_case(
        hidden_size=4, model_dtype=torch.bfloat16, mlp=mlp, case=spec, persist_result=persist_result
    )
    assert mlp.topk.calls == 1
    if not persist_result:
        assert not spec.routing_path.exists() and not spec.output_path.exists()
        return

    result = read_reference_result(spec, expect_routing=True)
    assert result.natural_routing is not None and result.prescribed_routing is not None
    assert torch.equal(result.natural_routing.topk_ids, natural_ids)
    assert torch.equal(result.natural_routing.topk_weights, natural_weights)
    assert torch.equal(result.prescribed_routing.topk_ids, ids)
    assert torch.equal(result.prescribed_routing.topk_weights, expected_weights)
    expected_output = torch.ones((3, 4), dtype=torch.bfloat16) + expected_weights.sum(dim=-1, keepdim=True).bfloat16()
    assert torch.equal(result.output, expected_output)
